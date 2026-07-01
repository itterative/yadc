<#
.SYNOPSIS
    Build the yadc webui as a Windows desktop .exe (CPU and GPU variants).

.DESCRIPTION
    Drives the full build pipeline:

    1. Installs JS deps and builds the SvelteKit SPA into yadc/webui/build.
    2. Creates two clean Python venvs under build/ (one per variant) so
       CPU and GPU builds don't pollute each other's ONNX runtime.
    3. Installs yadc into each venv — the GPU venv gets the [gpu] extra
       (onnxruntime-gpu) in addition to [packaging] (PyInstaller).
    4. Runs PyInstaller once per variant against the matching spec file.
    5. Prints the absolute paths to the two .exe files.

    All intermediate artifacts (node_modules, venvs, PyInstaller work dir)
    land under build/ which is .gitignored. The shipped binaries land
    under dist/ which is also .gitignored.

    Run from the repo root:

        pwsh scripts/build_windows_exe.ps1

.PARAMETER SkipNpmBuild
    Skip the ``npm ci && npm run build`` step. Use when iterating on the
    spec / entry script and the SPA bundle hasn't changed.

.PARAMETER SkipGpu
    Build only the CPU variant. Useful when you don't have a CUDA
    toolkit installed locally (the GPU build still bundles the
    onnxruntime-gpu wheels without CUDA, but skips the runtime probe
    that sniffs the GPU on first launch).

.EXAMPLE
    pwsh scripts/build_windows_exe.ps1
    pwsh scripts/build_windows_exe.ps1 -SkipNpmBuild
    pwsh scripts/build_windows_exe.ps1 -SkipGpu
#>

[CmdletBinding()]
param(
    [switch]$SkipNpmBuild,
    [switch]$SkipGpu
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Push-Location $repoRoot

try {
    $distDir = Join-Path $repoRoot "dist"
    $buildDir = Join-Path $repoRoot "build"
    New-Item -ItemType Directory -Force -Path $distDir | Out-Null
    New-Item -ItemType Directory -Force -Path $buildDir | Out-Null

    if (-not $SkipNpmBuild) {
        Write-Host "==> Building SvelteKit frontend" -ForegroundColor Cyan
        Push-Location (Join-Path $repoRoot "yadc/webui")
        try {
            npm ci
            if ($LASTEXITCODE -ne 0) { throw "npm ci failed" }
            npm run build
            if ($LASTEXITCODE -ne 0) { throw "npm run build failed" }
        } finally {
            Pop-Location
        }
    } else {
        Write-Host "==> Skipping npm build (using existing yadc/webui/build/)" -ForegroundColor Yellow
    }

    # Hints for the user. PyInstaller pulls in roughly 400 MB - 1.5 GB of
    # platform-locked binaries per venv; reporting progress keeps a long
    # build from looking stuck.
    function Build-Variant {
        param(
            [Parameter(Mandatory)] [string]$Label,
            [Parameter(Mandatory)] [string]$SpecFile,
            [Parameter(Mandatory)] [string]$OutputName,
            [Parameter(Mandatory)] [string[]]$Extras
        )
        $venvDir = Join-Path $buildDir "venv-$Label"
        Write-Host ""
        Write-Host "==> [$Label] creating venv at $venvDir" -ForegroundColor Cyan
        python -m venv $venvDir
        if ($LASTEXITCODE -ne 0) { throw "venv creation failed for $Label" }

        $pip = Join-Path $venvDir "Scripts/pip.exe"
        $pyinstaller = Join-Path $venvDir "Scripts/pyinstaller.exe"

        $extrasArg = ($Extras | ForEach-Object { ".$_" }) -join ","
        Write-Host "==> [$Label] installing yadc[$extrasArg] + pyinstaller" -ForegroundColor Cyan
        & $pip install --upgrade pip | Out-Null
        & $pip install -e "$repoRoot[$extrasArg]"
        if ($LASTEXITCODE -ne 0) { throw "pip install failed for $Label" }

        $workDir = Join-Path $buildDir "pyinstaller-$Label"
        Write-Host "==> [$Label] running PyInstaller" -ForegroundColor Cyan
        & $pyinstaller $SpecFile `
            --distpath $distDir `
            --workpath $workDir `
            --name $OutputName `
            --clean
        if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed for $Label" }
    }

    # CPU build. Order matters: CPU is the default recommendation so we
    # build it first even if the GPU step fails halfway.
    Build-Variant -Label "cpu" -SpecFile (Join-Path $repoRoot "scripts/build_exe.spec") `
                  -OutputName "yadc-webui-cpu" -Extras @("packaging")

    if (-not $SkipGpu) {
        Build-Variant -Label "gpu" -SpecFile (Join-Path $repoRoot "scripts/build_exe_gpu.spec") `
                      -OutputName "yadc-webui-gpu" -Extras @("packaging", "gpu")
    }

    Write-Host ""
    Write-Host "==> Build complete." -ForegroundColor Green
    Write-Host "    CPU binary: $(Join-Path $distDir 'yadc-webui-cpu.exe')"
    if (-not $SkipGpu) {
        Write-Host "    GPU binary: $(Join-Path $distDir 'yadc-webui-gpu.exe')"
    }
}
finally {
    Pop-Location
}
