from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from mss import MSS


APP_DIR = (
    Path(sys.executable).resolve().parent
    if getattr(sys, "frozen", False)
    else Path(__file__).resolve().parent
)
CONFIG_PATH = APP_DIR / "config.json"


@dataclass(frozen=True)
class Marker:
    profile: str
    role: str
    x: float
    y: float
    area: float
    circularity: float


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Config not found: {path}. Copy config.example.json to config.json and set map_rect."
        )

    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    required = ["map_rect", "meters_per_grid_cell", "marker_detection"]
    missing = [key for key in required if key not in cfg]
    if missing:
        raise ValueError(f"Missing config keys: {', '.join(missing)}")
    return cfg


def save_config(path: Path, cfg: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
        f.write("\n")


def capture_screen() -> np.ndarray:
    with MSS() as sct:
        monitor = sct.monitors[0]
        shot = sct.grab(monitor)
        bgra = np.array(shot)
    return cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)


def capture_map(rect: dict[str, int]) -> np.ndarray:
    region = {
        "left": int(rect["left"]),
        "top": int(rect["top"]),
        "width": int(rect["width"]),
        "height": int(rect["height"]),
    }
    if region["left"] < 0 or region["top"] < 0 or region["width"] <= 0 or region["height"] <= 0:
        raise ValueError("map_rect must contain non-negative left/top and positive width/height")

    with MSS() as sct:
        shot = sct.grab(region)
        bgra = np.array(shot)
    return cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def crop_map(image: np.ndarray, rect: dict[str, int]) -> np.ndarray:
    left = int(rect["left"])
    top = int(rect["top"])
    width = int(rect["width"])
    height = int(rect["height"])
    if left < 0 or top < 0 or width <= 0 or height <= 0:
        raise ValueError("map_rect must contain non-negative left/top and positive width/height")
    if left + width > image.shape[1] or top + height > image.shape[0]:
        raise ValueError(
            f"map_rect {rect} is outside image bounds {image.shape[1]}x{image.shape[0]}"
        )
    return image[top : top + height, left : left + width].copy()


def resolve_grid_cell_px(map_image: np.ndarray, cfg: dict[str, Any]) -> float:
    grid_cell_px = cfg.get("grid_cell_px")
    if grid_cell_px is not None:
        return float(grid_cell_px)

    grid_columns = cfg.get("grid_columns")
    if grid_columns is not None:
        columns = float(grid_columns)
        if columns <= 0:
            raise ValueError("grid_columns must be positive")
        return float(map_image.shape[1]) / columns

    if cfg.get("grid_detection", {}).get("enabled", True):
        detected = detect_grid_step_px(map_image, cfg.get("grid_detection", {}))
        if detected is not None:
            return detected

    raise ValueError("Grid size was not detected. Set grid_cell_px or grid_columns in config.json.")


def validate_minimap(map_image: np.ndarray, cfg: dict[str, Any]) -> None:
    validation_cfg = cfg.get("map_validation", {})
    if not validation_cfg.get("enabled", True):
        return
    columns = cfg.get("grid_columns")
    if columns is None:
        return

    expected_step = map_image.shape[1] / float(columns)
    search_margin = float(validation_cfg.get("search_margin_ratio", 0.3))
    detected_step = detect_grid_step_px(
        map_image,
        {
            "min_cell_px": max(5, int(expected_step * (1.0 - search_margin))),
            "max_cell_px": int(expected_step * (1.0 + search_margin)),
        },
    )
    max_error = float(validation_cfg.get("max_grid_step_error_ratio", 0.15))
    if detected_step is None or abs(detected_step - expected_step) / expected_step > max_error:
        raise ValueError(
            "Minimap grid not detected. Close in-game menus and make sure the minimap is visible."
        )


def detect_grid_step_px(map_image: np.ndarray, grid_cfg: dict[str, Any]) -> float | None:
    gray = cv2.cvtColor(map_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    edges = cv2.Canny(gray, 50, 150)
    vertical_projection = edges.mean(axis=0).astype(np.float32)
    horizontal_projection = edges.mean(axis=1).astype(np.float32)

    min_cell = int(grid_cfg.get("min_cell_px", 25))
    max_cell = int(grid_cfg.get("max_cell_px", 180))

    vertical_step = estimate_period(vertical_projection, min_cell, max_cell)
    horizontal_step = estimate_period(horizontal_projection, min_cell, max_cell)
    steps = [step for step in (vertical_step, horizontal_step) if step is not None]
    if not steps:
        return None
    return float(np.median(steps))


def estimate_period(signal: np.ndarray, min_period: int, max_period: int) -> float | None:
    signal = signal - signal.mean()
    if float(np.abs(signal).sum()) < 1e-6:
        return None

    corr = np.correlate(signal, signal, mode="full")[len(signal) - 1 :]
    upper = min(max_period, len(corr) - 1)
    lower = min(min_period, upper)
    if upper <= lower:
        return None

    window = corr[lower : upper + 1]
    best = int(np.argmax(window)) + lower
    if corr[best] <= 0:
        return None

    # Refine with neighboring bins when possible.
    left = max(best - 1, lower)
    right = min(best + 1, upper)
    weights = corr[left : right + 1]
    positions = np.arange(left, right + 1, dtype=np.float32)
    if float(weights.sum()) <= 0:
        return float(best)
    return float(np.average(positions, weights=weights))


def detect_markers(map_image: np.ndarray, marker_cfg: dict[str, Any]) -> tuple[list[Marker], np.ndarray]:
    hsv = cv2.cvtColor(map_image, cv2.COLOR_BGR2HSV)
    debug_mask = np.zeros(map_image.shape[:2], dtype=np.uint8)
    markers: list[Marker] = []

    min_area = float(marker_cfg.get("min_area_px", 20))
    max_area = float(marker_cfg.get("max_area_px", 2500))

    for profile in marker_cfg.get("profiles", []):
        profile_min_area = float(profile.get("min_area_px", min_area))
        profile_max_area = float(profile.get("max_area_px", max_area))
        min_circularity = profile.get("min_circularity")
        max_circularity = profile.get("max_circularity")
        mask = np.zeros(map_image.shape[:2], dtype=np.uint8)
        for lower, upper in profile.get("hsv_ranges", []):
            lower_np = np.array(lower, dtype=np.uint8)
            upper_np = np.array(upper, dtype=np.uint8)
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower_np, upper_np))

        kernel = np.ones((3, 3), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        debug_mask = cv2.bitwise_or(debug_mask, mask)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < profile_min_area or area > profile_max_area:
                continue
            perimeter = float(cv2.arcLength(contour, True))
            circularity = 0.0 if perimeter <= 0 else 4.0 * math.pi * area / (perimeter * perimeter)
            if min_circularity is not None and circularity < float(min_circularity):
                continue
            if max_circularity is not None and circularity > float(max_circularity):
                continue
            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue
            x = moments["m10"] / moments["m00"]
            y = moments["m01"] / moments["m00"]
            markers.append(
                Marker(
                    str(profile.get("name", "marker")),
                    str(profile.get("role", profile.get("name", "marker"))),
                    x,
                    y,
                    area,
                    circularity,
                )
            )

    markers.sort(key=lambda marker: marker.area, reverse=True)
    return markers, debug_mask


def build_hsv_mask(hsv: np.ndarray, ranges: list[list[list[int]]]) -> np.ndarray:
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in ranges:
        mask = cv2.bitwise_or(
            mask,
            cv2.inRange(
                hsv,
                np.array(lower, dtype=np.uint8),
                np.array(upper, dtype=np.uint8),
            ),
        )
    return mask


def contour_marker(contour: np.ndarray, profile: str, role: str) -> Marker | None:
    area = float(cv2.contourArea(contour))
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        return None
    perimeter = float(cv2.arcLength(contour, True))
    circularity = 0.0 if perimeter <= 0 else 4.0 * math.pi * area / (perimeter * perimeter)
    return Marker(
        profile=profile,
        role=role,
        x=moments["m10"] / moments["m00"],
        y=moments["m01"] / moments["m00"],
        area=area,
        circularity=circularity,
    )


def contour_bright_fraction(
    hsv: np.ndarray, contour: np.ndarray, min_value: int = 220
) -> float:
    outline = np.zeros(hsv.shape[:2], dtype=np.uint8)
    cv2.drawContours(outline, [contour], -1, 255, 1)
    values = hsv[:, :, 2][outline > 0]
    if values.size == 0:
        return 0.0
    return float(np.mean(values >= min_value))


def detect_player_arrow(
    map_image: np.ndarray, marker_cfg: dict[str, Any]
) -> tuple[Marker, np.ndarray]:
    cfg = marker_cfg.get("player_arrow_detection", {})
    hsv = cv2.cvtColor(map_image, cv2.COLOR_BGR2HSV)

    triangle_ranges = cfg.get("triangle_hsv_ranges", [[[0, 0, 170], [179, 100, 255]]])
    triangle_mask = build_hsv_mask(hsv, triangle_ranges)
    triangle_mask = cv2.morphologyEx(
        triangle_mask, cv2.MORPH_CLOSE, np.ones((2, 2), dtype=np.uint8)
    )
    triangle_contours, _ = cv2.findContours(
        triangle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    min_triangle_area = float(cfg.get("min_triangle_area_px", 120))
    max_triangle_area = float(cfg.get("max_triangle_area_px", 500))
    min_triangle_side = int(cfg.get("min_triangle_side_px", 12))
    max_triangle_side = int(cfg.get("max_triangle_side_px", 45))
    min_triangle_solidity = float(cfg.get("min_triangle_solidity", 0.75))
    min_small_triangle_area = float(cfg.get("min_small_triangle_area_px", 7))
    max_small_triangle_area = float(cfg.get("max_small_triangle_area_px", 80))
    min_small_triangle_side = int(cfg.get("min_small_triangle_side_px", 4))
    max_small_triangle_side = int(cfg.get("max_small_triangle_side_px", 18))
    min_small_triangle_solidity = float(cfg.get("min_small_triangle_solidity", 0.6))
    min_triangle_bright_fraction = float(cfg.get("min_triangle_bright_fraction", 0.25))
    approximations = cfg.get("triangle_approximations")
    if approximations is None:
        base_approximation = float(cfg.get("triangle_approximation", 0.04))
        approximations = [base_approximation, 0.06, 0.08]
    approximations = [float(value) for value in approximations]

    triangle_candidates: list[Marker] = []
    for contour in triangle_contours:
        area = float(cv2.contourArea(contour))
        _, (rotated_width, rotated_height), _ = cv2.minAreaRect(contour)
        short_side = min(rotated_width, rotated_height)
        long_side = max(rotated_width, rotated_height)
        is_large_triangle = (
            min_triangle_area <= area <= max_triangle_area
            and min_triangle_side <= short_side
            and long_side <= max_triangle_side
        )
        is_small_triangle = (
            min_small_triangle_area <= area <= max_small_triangle_area
            and min_small_triangle_side <= short_side
            and long_side <= max_small_triangle_side
        )
        if not (is_large_triangle or is_small_triangle):
            continue
        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 0:
            continue
        vertex_counts = [
            len(cv2.approxPolyDP(contour, approximation * perimeter, True))
            for approximation in approximations
        ]
        hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
        solidity = 0.0 if hull_area <= 0 else area / hull_area
        required_solidity = (
            min_triangle_solidity if is_large_triangle else min_small_triangle_solidity
        )
        bright_fraction = contour_bright_fraction(hsv, contour)
        if (
            3 not in vertex_counts
            or solidity < required_solidity
            or bright_fraction < min_triangle_bright_fraction
        ):
            continue
        marker = contour_marker(contour, "player_triangle", "player_arrow")
        if marker is not None:
            triangle_candidates.append(marker)

    if triangle_candidates:
        return max(triangle_candidates, key=lambda marker: marker.area), triangle_mask

    edge_ranges = cfg.get(
        "edge_triangle_hsv_ranges",
        [
            [[0, 0, 180], [179, 130, 255]],
            [[18, 100, 160], [45, 255, 255]],
        ],
    )
    edge_mask = build_hsv_mask(hsv, edge_ranges)
    edge_mask = cv2.morphologyEx(
        edge_mask, cv2.MORPH_CLOSE, np.ones((2, 2), dtype=np.uint8)
    )
    edge_contours, _ = cv2.findContours(
        edge_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    edge_margin = int(cfg.get("edge_triangle_margin_px", 2))
    min_edge_area = float(cfg.get("min_edge_triangle_area_px", 3))
    max_edge_area = float(cfg.get("max_edge_triangle_area_px", 80))
    min_edge_side = float(cfg.get("min_edge_triangle_side_px", 1.5))
    max_edge_side = float(cfg.get("max_edge_triangle_side_px", 20))
    min_edge_solidity = float(cfg.get("min_edge_triangle_solidity", 0.5))
    min_edge_bright_fraction = float(cfg.get("min_edge_triangle_bright_fraction", 0.2))

    edge_candidates: list[Marker] = []
    for contour in edge_contours:
        area = float(cv2.contourArea(contour))
        if not min_edge_area <= area <= max_edge_area:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        touches_edge = (
            x <= edge_margin
            or y <= edge_margin
            or x + width >= map_image.shape[1] - edge_margin
            or y + height >= map_image.shape[0] - edge_margin
        )
        if not touches_edge:
            continue
        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 0:
            continue
        _, (rotated_width, rotated_height), _ = cv2.minAreaRect(contour)
        short_side = min(rotated_width, rotated_height)
        long_side = max(rotated_width, rotated_height)
        if not (min_edge_side <= short_side and long_side <= max_edge_side):
            continue
        vertex_counts = [
            len(cv2.approxPolyDP(contour, approximation * perimeter, True))
            for approximation in approximations
        ]
        hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
        solidity = 0.0 if hull_area <= 0 else area / hull_area
        bright_fraction = contour_bright_fraction(hsv, contour)
        if (
            3 not in vertex_counts
            or solidity < min_edge_solidity
            or bright_fraction < min_edge_bright_fraction
        ):
            continue
        marker = contour_marker(contour, "player_edge_triangle", "player_arrow")
        if marker is not None:
            edge_candidates.append(marker)

    if edge_candidates:
        return max(edge_candidates, key=lambda marker: marker.area), edge_mask

    if not bool(cfg.get("allow_blue_fallback", False)):
        raise ValueError("Player arrow not found: no suitable white triangle was detected.")

    blue_ranges = cfg.get("blue_hsv_ranges", [[[95, 90, 90], [135, 255, 255]]])
    blue_mask = build_hsv_mask(hsv, blue_ranges)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8))
    blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_blue_area = float(cfg.get("min_blue_area_px", 80))
    max_blue_area = float(cfg.get("max_blue_area_px", 1500))
    blue_contours = [
        contour
        for contour in blue_contours
        if min_blue_area <= cv2.contourArea(contour) <= max_blue_area
    ]
    if not blue_contours:
        raise ValueError("Player icon not found: no suitable blue marker was detected.")

    accent_ranges = cfg.get(
        "accent_hsv_ranges",
        [
            [[18, 100, 170], [45, 255, 255]],
            [[0, 0, 210], [179, 100, 255]],
        ],
    )
    accent_mask = build_hsv_mask(hsv, accent_ranges)
    search_padding = int(cfg.get("search_padding_px", 18))
    min_accent_area = float(cfg.get("min_accent_area_px", 5))
    max_accent_area = float(cfg.get("max_accent_area_px", 300))

    best: tuple[float, Marker] | None = None
    for blue_contour in blue_contours:
        x, y, width, height = cv2.boundingRect(blue_contour)
        left = max(0, x - search_padding)
        top = max(0, y - search_padding)
        right = min(map_image.shape[1], x + width + search_padding)
        bottom = min(map_image.shape[0], y + height + search_padding)

        local_mask = np.zeros_like(accent_mask)
        local_mask[top:bottom, left:right] = accent_mask[top:bottom, left:right]
        local_mask = cv2.morphologyEx(
            local_mask, cv2.MORPH_CLOSE, np.ones((2, 2), dtype=np.uint8)
        )
        accent_contours, _ = cv2.findContours(
            local_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        blue_moments = cv2.moments(blue_contour)
        if blue_moments["m00"] == 0:
            continue
        blue_x = blue_moments["m10"] / blue_moments["m00"]
        blue_y = blue_moments["m01"] / blue_moments["m00"]

        for accent_contour in accent_contours:
            marker = contour_marker(accent_contour, "player_arrow", "player_arrow")
            if marker is None or not (min_accent_area <= marker.area <= max_accent_area):
                continue
            distance = math.hypot(marker.x - blue_x, marker.y - blue_y)
            if distance > search_padding + max(width, height) * 0.6:
                continue
            score = marker.area - distance * 0.25
            if best is None or score > best[0]:
                best = (score, marker)

    if best is None:
        # Fall back to the blue icon center if anti-aliasing hides the light arrow.
        largest_blue = max(blue_contours, key=cv2.contourArea)
        fallback = contour_marker(largest_blue, "player_icon", "player_arrow")
        if fallback is None:
            raise ValueError("Player arrow not found near the blue player icon.")
        return fallback, cv2.bitwise_or(triangle_mask, cv2.bitwise_or(blue_mask, accent_mask))

    return best[1], cv2.bitwise_or(triangle_mask, cv2.bitwise_or(blue_mask, accent_mask))


def detect_yellow_target(
    map_image: np.ndarray, marker_cfg: dict[str, Any], player: Marker
) -> tuple[Marker, np.ndarray]:
    cfg = marker_cfg.get("yellow_target_detection", {})
    hsv = cv2.cvtColor(map_image, cv2.COLOR_BGR2HSV)
    ranges = cfg.get("hsv_ranges", [[[20, 140, 160], [45, 255, 255]]])
    mask = build_hsv_mask(hsv, ranges)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = float(cfg.get("min_area_px", 25))
    max_area = float(cfg.get("max_area_px", 1200))
    min_circularity = float(cfg.get("min_circularity", 0.15))
    min_side = int(cfg.get("min_side_px", 7))
    max_side = int(cfg.get("max_side_px", 55))
    min_player_distance = float(cfg.get("min_player_distance_px", 20))
    border_margin = float(cfg.get("border_margin_px", 12))

    candidates: list[Marker] = []
    for contour in contours:
        marker = contour_marker(contour, "yellow_marker", "yellow_marker")
        if marker is None or not (min_area <= marker.area <= max_area):
            continue
        _, _, width, height = cv2.boundingRect(contour)
        if not (min_side <= width <= max_side and min_side <= height <= max_side):
            continue
        aspect = width / float(height)
        if not 0.5 <= aspect <= 2.0 or marker.circularity < min_circularity:
            continue
        if math.hypot(marker.x - player.x, marker.y - player.y) < min_player_distance:
            continue
        if not (
            border_margin <= marker.x <= map_image.shape[1] - border_margin
            and border_margin <= marker.y <= map_image.shape[0] - border_margin
        ):
            continue
        candidates.append(marker)

    if not candidates:
        raise ValueError(
            "Yellow target marker not found. Place the yellow marker on the minimap and try again."
        )

    return max(candidates, key=lambda marker: marker.area * max(marker.circularity, 0.1)), mask


def select_marker(markers: list[Marker], role: str) -> Marker:
    matches = [marker for marker in markers if marker.role == role or marker.profile == role]
    if not matches:
        available = ", ".join(f"{m.profile}/{m.role}" for m in markers[:8]) or "none"
        raise ValueError(f"Could not find marker role '{role}'. Found: {available}")
    return max(matches, key=lambda marker: marker.area)


def analyze_map(map_image: np.ndarray, cfg: dict[str, Any], debug_path: Path | None = None) -> dict[str, Any]:
    validate_minimap(map_image, cfg)
    grid_cell_px = resolve_grid_cell_px(map_image, cfg)
    marker_cfg = cfg["marker_detection"]
    first, player_mask = detect_player_arrow(map_image, marker_cfg)
    second, target_mask = detect_yellow_target(map_image, marker_cfg, first)
    markers = [first, second]
    mask = cv2.bitwise_or(player_mask, target_mask)
    distance_px = math.hypot(first.x - second.x, first.y - second.y)
    meters = distance_px / grid_cell_px * float(cfg["meters_per_grid_cell"])

    if debug_path:
        debug = map_image.copy()
        for marker in markers[:8]:
            center = (int(round(marker.x)), int(round(marker.y)))
            cv2.circle(debug, center, 8, (255, 255, 255), 2)
            cv2.putText(
                debug,
                marker.role,
                (center[0] + 10, center[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        cv2.line(
            debug,
            (int(round(first.x)), int(round(first.y))),
            (int(round(second.x)), int(round(second.y))),
            (255, 255, 255),
            2,
        )
        combined = np.hstack([debug, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)])
        cv2.imwrite(str(debug_path), combined)

    return {
        "distance_m": meters,
        "distance_km": meters / 1000.0,
        "distance_px": distance_px,
        "grid_cell_px": grid_cell_px,
        "from": {"profile": first.profile, "role": first.role, "x": round(first.x, 2), "y": round(first.y, 2)},
        "to": {"profile": second.profile, "role": second.role, "x": round(second.x, 2), "y": round(second.y, 2)},
        "markers": [
            {
                "profile": m.profile,
                "role": m.role,
                "x": round(m.x, 2),
                "y": round(m.y, 2),
                "area": round(m.area, 1),
                "circularity": round(m.circularity, 3),
            }
            for m in markers[:8]
        ],
    }


def analyze(image: np.ndarray, cfg: dict[str, Any], debug_path: Path | None = None) -> dict[str, Any]:
    return analyze_map(crop_map(image, cfg["map_rect"]), cfg, debug_path)


def print_result(result: dict[str, Any]) -> None:
    print(
        f"{result['distance_m']:.1f} m "
        f"({result['distance_km']:.3f} km), "
        f"grid={result['grid_cell_px']:.2f}px, "
        f"from={result['from']['role']}({result['from']['x']},{result['from']['y']}) "
        f"to={result['to']['role']}({result['to']['x']},{result['to']['y']})"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure distance between two War Thunder map markers from a screenshot."
    )
    parser.add_argument("--config", type=Path, default=CONFIG_PATH, help="Path to config.json")
    parser.add_argument("--image", type=Path, help="Use image file instead of live screenshot")
    parser.add_argument("--image-is-map", action="store_true", help="Treat --image as already cropped minimap")
    parser.add_argument("--save-crop", type=Path, help="Save the current minimap crop and exit")
    parser.add_argument("--debug", type=Path, help="Write debug image with crop, markers and mask")
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Wait before live capture so there is time to switch back to the game",
    )
    parser.add_argument("--watch", action="store_true", help="Keep measuring repeatedly")
    parser.add_argument("--interval", type=float, default=0.25, help="Watch interval in seconds")
    parser.add_argument(
        "--cell-size",
        type=float,
        help="Use this many meters per grid cell for the current run",
    )
    parser.add_argument(
        "--ask-cell-size",
        action="store_true",
        help="Ask for meters per grid cell before starting and save the answer",
    )
    parser.add_argument(
        "--hotkey",
        action="store_true",
        help="Wait for F8 to measure; press F9 to exit (Windows)",
    )
    return parser


def raw_debug_path(processed_path: Path) -> Path:
    stem = processed_path.stem
    if stem.endswith("-processed"):
        stem = stem[: -len("-processed")]
    suffix = processed_path.suffix or ".png"
    return processed_path.with_name(f"{stem}-minimap{suffix}")


def save_failed_debug(map_image: np.ndarray, processed_path: Path, error: Exception) -> None:
    debug = map_image.copy()
    overlay = debug.copy()
    cv2.rectangle(overlay, (0, 0), (debug.shape[1], 58), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.72, debug, 0.28, 0, debug)
    cv2.putText(
        debug,
        "DETECTION ERROR",
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    message = str(error)
    cv2.putText(
        debug,
        message[:58],
        (8, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(processed_path), debug)


def measure_once(args: argparse.Namespace, cfg: dict[str, Any]) -> None:
    if args.image:
        image = read_image(args.image)
        map_image = image if args.image_is_map else crop_map(image, cfg["map_rect"])
    else:
        map_image = capture_map(cfg["map_rect"])

    if args.save_crop:
        cv2.imwrite(str(args.save_crop), map_image)
        print(f"Saved minimap crop: {args.save_crop}", flush=True)
        return

    if args.debug:
        cv2.imwrite(str(raw_debug_path(args.debug)), map_image)
    try:
        result = analyze_map(map_image, cfg, args.debug)
    except Exception as exc:
        if args.debug:
            save_failed_debug(map_image, args.debug, exc)
        raise
    print_result(result)
    sys.stdout.flush()


def run_hotkey_mode(args: argparse.Namespace, cfg: dict[str, Any]) -> int:
    if sys.platform != "win32":
        print("Error: global hotkey mode is supported only on Windows.", file=sys.stderr)
        return 2

    import ctypes
    import winsound
    from ctypes import wintypes

    hotkey_message = 0x0312
    no_repeat = 0x4000
    measure_id = 1
    exit_id = 2
    vk_f8 = 0x77
    vk_f9 = 0x78
    user32 = ctypes.windll.user32

    if not user32.RegisterHotKey(None, measure_id, no_repeat, vk_f8):
        print("Error: F8 is already being used by another program.", file=sys.stderr)
        return 2
    if not user32.RegisterHotKey(None, exit_id, no_repeat, vk_f9):
        user32.UnregisterHotKey(None, measure_id)
        print("Error: F9 is already being used by another program.", file=sys.stderr)
        return 2

    print("Ready. Press F8 in War Thunder to measure distance. Press F9 to exit.", flush=True)
    message = wintypes.MSG()
    try:
        while user32.GetMessageW(ctypes.byref(message), None, 0, 0) > 0:
            if message.message != hotkey_message:
                continue
            if message.wParam == exit_id:
                print("Exiting.", flush=True)
                break
            if message.wParam != measure_id:
                continue
            try:
                measure_once(args, cfg)
                winsound.Beep(1000, 90)
            except Exception as exc:
                print(f"Error: {exc}", file=sys.stderr, flush=True)
                winsound.Beep(400, 180)
    finally:
        user32.UnregisterHotKey(None, measure_id)
        user32.UnregisterHotKey(None, exit_id)
    return 0


def ask_cell_size(cfg: dict[str, Any], config_path: Path) -> None:
    current = float(cfg["meters_per_grid_cell"])
    print(f"Текущий размер клетки: {current:g} м")
    while True:
        try:
            answer = input(
                f"Введите размер клетки в метрах или нажмите Enter, чтобы оставить {current:g}: "
            ).strip()
        except EOFError:
            answer = ""

        if not answer:
            print(f"Размер клетки: {current:g} м", flush=True)
            return
        try:
            value = float(answer.replace(",", "."))
        except ValueError:
            print("Введите положительное число, например 275 или 1000.")
            continue
        if value <= 0:
            print("Размер клетки должен быть больше нуля.")
            continue

        cfg["meters_per_grid_cell"] = value
        try:
            save_config(config_path, cfg)
            print(f"Размер клетки сохранён: {value:g} м", flush=True)
        except OSError as exc:
            print(
                f"Warning: размер клетки применяется, но config.json не удалось сохранить: {exc}",
                file=sys.stderr,
                flush=True,
            )
        return


def main() -> int:
    args = build_parser().parse_args()

    if getattr(sys, "frozen", False) and len(sys.argv) == 1:
        args.hotkey = True
        args.ask_cell_size = True
    if args.hotkey and args.debug is None:
        args.debug = APP_DIR / "debug-processed.png"

    try:
        cfg = load_config(args.config)
    except Exception as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 2


    if args.cell_size is not None:
        if args.cell_size <= 0:
            print("Config error: --cell-size must be greater than zero.", file=sys.stderr)
            return 2
        cfg["meters_per_grid_cell"] = args.cell_size
    if args.ask_cell_size:
        ask_cell_size(cfg, args.config)

    if args.hotkey:
        if args.image or args.watch or args.save_crop:
            print(
                "Error: --hotkey cannot be combined with --image, --watch, or --save-crop.",
                file=sys.stderr,
            )
            return 2
        return run_hotkey_mode(args, cfg)

    if not args.image and args.delay > 0:
        delay = max(args.delay, 0.0)
        print(f"Capturing in {delay:g} seconds. Switch to War Thunder now...")
        time.sleep(delay)

    while True:
        try:
            measure_once(args, cfg)
            if args.save_crop:
                return 0
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            if not args.watch:
                return 1

        if not args.watch:
            break
        time.sleep(max(args.interval, 0.05))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
