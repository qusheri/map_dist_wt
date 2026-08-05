$ErrorActionPreference = "Stop"

$projectRoot = $PSScriptRoot
$stagingRoot = Join-Path $projectRoot ".exe-build-staging"
$stagingDist = Join-Path $stagingRoot "dist"
$stagingWork = Join-Path $stagingRoot "work"
$stagingSpec = Join-Path $stagingRoot "spec"
$finalDist = Join-Path $projectRoot "dist"

function Remove-StagingDirectory {
    if (-not (Test-Path -LiteralPath $stagingRoot)) {
        return
    }

    $resolvedProject = [System.IO.Path]::GetFullPath($projectRoot).TrimEnd('\')
    $resolvedStaging = [System.IO.Path]::GetFullPath($stagingRoot).TrimEnd('\')
    if (-not $resolvedStaging.StartsWith("$resolvedProject\", [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove staging directory outside the project: $resolvedStaging"
    }
    Remove-Item -LiteralPath $resolvedStaging -Recurse -Force
}

Remove-StagingDirectory

Push-Location $projectRoot
try {
    python -m pip install -r requirements.txt
    python -m pip install -r requirements-build.txt
    python -m PyInstaller `
        --noconfirm `
        --clean `
        --onefile `
        --console `
        --noupx `
        --name WarThunderDistance `
        --distpath $stagingDist `
        --workpath $stagingWork `
        --specpath $stagingSpec `
        map_distance.py

    $stagedExe = Join-Path $stagingDist "WarThunderDistance.exe"
    & $stagedExe --help *> $null
    if ($LASTEXITCODE -ne 0) {
        throw "Built executable failed its startup check."
    }

    New-Item -ItemType Directory -Path $finalDist -Force | Out-Null
    $finalExe = Join-Path $finalDist "WarThunderDistance.exe"
    try {
        Copy-Item -LiteralPath $stagedExe -Destination $finalExe -Force
        $builtExeName = "WarThunderDistance.exe"
    }
    catch [System.IO.IOException] {
        $finalExe = Join-Path $finalDist "WarThunderDistance-updated.exe"
        Copy-Item -LiteralPath $stagedExe -Destination $finalExe -Force
        $builtExeName = "WarThunderDistance-updated.exe"
        Write-Warning "WarThunderDistance.exe is running; wrote the update as $builtExeName"
    }
    $finalConfig = Join-Path $finalDist "config.json"
    if (-not (Test-Path -LiteralPath $finalConfig)) {
        Copy-Item -LiteralPath (Join-Path $projectRoot "config.json") -Destination $finalConfig
    }
}
finally {
    Pop-Location
    Remove-StagingDirectory
}

Write-Host "Built: dist\$builtExeName"
Write-Host "Config: dist\config.json"
