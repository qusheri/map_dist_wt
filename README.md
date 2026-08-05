# War Thunder map distance

Утилита быстро снимает только миникарту War Thunder, находит стрелочку игрока и желтую метку, затем считает расстояние между ними по клеткам карты.

## Установка

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item config.example.json config.json
```

## Настройка

В `config.example.json` уже стоят примерные координаты миникарты для экрана `2560x1440` по твоему скриншоту:

```json
"map_rect": {
  "left": 2114,
  "top": 994,
  "width": 433,
  "height": 433
}
```

Проверь обрезку:

```powershell
python map_distance.py --save-crop minimap.png
```

Если `minimap.png` захватывает лишнее или режет карту, подстрой `left`, `top`, `width`, `height` в `config.json`.

Главный масштаб:

```json
"meters_per_grid_cell": 1000,
"grid_columns": 7
```

`meters_per_grid_cell` - сколько метров в одной клетке. Это поле ты меняешь сам под карту.

`grid_columns: 7` означает, что программа делит ширину миникарты на 7 клеток и сама получает пиксели клетки. Если захочешь задать пиксели вручную, поставь:

```json
"grid_columns": null,
"grid_cell_px": 61.85
```

## Запуск

Один замер с текущей миникарты:

```powershell
python map_distance.py
```

Быстрый повтор:

```powershell
python map_distance.py --watch --interval 0.2
```

Один расчет расстояния от твоей стрелочки до желтой метки:

```powershell
python map_distance.py
```

Отладочная картинка с найденной стрелочкой и желтой меткой:

```powershell
python map_distance.py --debug debug.png
```

Если игра сворачивается при переходе в консоль, добавь задержку и сразу переключись
обратно в игру:

```powershell
python map_distance.py --delay 3 --debug debug.png
```

При ошибке распознавания `debug.png` всё равно сохранится и будет содержать исходный
кадр области миникарты.

В консоли будет строка такого вида:

```text
526.0 m (0.526 km), grid=61.86px, from=player_arrow(43.2,295.4) to=yellow_marker(92.1,72.6)
```

Проверка на полном сохраненном скриншоте:

```powershell
python map_distance.py --image screenshot.png --debug debug.png
```

Проверка на уже обрезанной миникарте:

```powershell
python map_distance.py --image minimap.png --image-is-map --debug debug.png
```

## Сборка EXE

```powershell
.\build_exe.ps1
```

Готовые файлы появятся в `dist`:

```text
dist\WarThunderDistance.exe
dist\config.json
```

Запусти `WarThunderDistance.exe` двойным кликом. Программа останется в консоли:

- `F8` — сделать снимок миникарты и рассчитать расстояние;
- `F9` — закрыть программу.

При успешном расчёте звучит короткий высокий сигнал, при ошибке — низкий. Последний
кадры сохраняются рядом с программой:

- `debug-minimap.png` — исходный снимок миникарты;
- `debug-processed.png` — найденные точки и маска распознавания.

Оба файла обновляются при каждом нажатии `F8`, в том числе при ошибке распознавания.
