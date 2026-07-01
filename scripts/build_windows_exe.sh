#!/usr/bin/env bash
# Build the SvelteKit SPA and (on Windows) the desktop .exe variants.
#
# Linux/macOS runners can use this script to validate the SPA build and
# to set up the two Python venvs before delegating to PyInstaller on a
# Windows host. PyInstaller itself does not cross-compile, so this
# script aborts with a clear error at the bundling step on non-Windows
# hosts — running it to completion only happens on Windows.
#
# Usage from the repo root:
#
#     ./scripts/build_windows_exe.sh                # full build
#     ./scripts/build_windows_exe.sh --skip-npm     # reuse existing yadc/webui/build/
#     ./scripts/build_windows_exe.sh --skip-gpu     # CPU variant only
#
# Reuses logic with scripts/build_windows_exe.ps1 — keep them in sync.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

SKIP_NPM=0
SKIP_GPU=0
for arg in "$@"; do
    case "$arg" in
        --skip-npm) SKIP_NPM=1 ;;
        --skip-gpu) SKIP_GPU=1 ;;
        *) echo "unknown arg: $arg" >&2; exit 2 ;;
    esac
done

DIST_DIR="$REPO_ROOT/dist"
BUILD_DIR="$REPO_ROOT/build"
mkdir -p "$DIST_DIR" "$BUILD_DIR"

if [ "$SKIP_NPM" -eq 0 ]; then
    echo "==> Building SvelteKit frontend"
    pushd "$REPO_ROOT/yadc/webui" >/dev/null
    npm ci
    npm run build
    popd >/dev/null
else
    echo "==> Skipping npm build (using existing yadc/webui/build/)"
fi

# Gate on Windows here — PyInstaller can't cross-compile. The SPA
# build above still runs cleanly on any host, which is useful for CI
# that wants to validate the JS side before a Windows runner takes
# over for the bundle step.
case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*) : ;;
    *)
        echo "" >&2
        echo "This script's purpose is to build a Windows .exe." >&2
        echo "PyInstaller does not cross-compile; run this script on" >&2
        echo "Windows (PowerShell: scripts/build_windows_exe.ps1) or" >&2
        echo "in a Windows VM/WSL/CI runner." >&2
        echo "" >&2
        echo "Frontend build completed above. Stopping before PyInstaller." >&2
        exit 0
        ;;
esac

build_variant() {
    local label="$1"
    local spec_file="$2"
    local output_name="$3"
    shift 3
    local extras_csv
    extras_csv=$(IFS=,; echo "$*")
    local venv_dir="$BUILD_DIR/venv-$label"

    echo ""
    echo "==> [$label] creating venv at $venv_dir"
    python -m venv "$venv_dir"

    # shellcheck disable=SC1091
    source "$venv_dir/Scripts/activate"

    echo "==> [$label] installing yadc[$extras_csv] + pyinstaller"
    pip install --upgrade pip >/dev/null
    pip install -e "$REPO_ROOT[$extras_csv]"

    local work_dir="$BUILD_DIR/pyinstaller-$label"
    echo "==> [$label] running PyInstaller"
    pyinstaller "$spec_file" \
        --distpath "$DIST_DIR" \
        --workpath "$work_dir" \
        --name "$output_name" \
        --clean
}

build_variant cpu "$REPO_ROOT/scripts/build_exe.spec" "yadc-webui-cpu" packaging

if [ "$SKIP_GPU" -eq 0 ]; then
    build_variant gpu "$REPO_ROOT/scripts/build_exe_gpu.spec" "yadc-webui-gpu" packaging gpu
fi

echo ""
echo "==> Build complete."
echo "    CPU binary: $DIST_DIR/yadc-webui-cpu.exe"
if [ "$SKIP_GPU" -eq 0 ]; then
    echo "    GPU binary: $DIST_DIR/yadc-webui-gpu.exe"
fi
