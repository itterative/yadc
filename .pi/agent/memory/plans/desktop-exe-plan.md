---
name: desktop-exe-plan
description: Package the yadc webui as a standalone Windows .exe (PyInstaller) that auto-opens a browser tab on launch. Ship two variants — CPU and GPU. Build via a script; no CI workflow in this plan.
last_history: 1
---

# Desktop .exe plan

Bundle the yadc webui into a single Windows `.exe` users can double-click. The exe starts the webui (Quart + uvicorn, SvelteKit SPA from `yadc/webui/build`) and auto-opens it in their default browser. Two variants ship: CPU (default) and GPU.

## Goals

- **One-click UX**: double-click `yadc-webui.exe` → terminal-style console opens → browser tab opens automatically to `http://127.0.0.1:7860`.
- **No Python install required** on the target Windows machine.
- **Two variants shipped**: CPU (smaller, no CUDA) and GPU (with `onnxruntime-gpu`).
- **Existing CLI still works**: `yadc webui serve …` is unchanged when run from a normal install — the .exe is just a thin frozen wrapper.
- **Build is reproducible from source**: a script anyone with Windows + Python 3.13 can run.

## Non-goals (out of scope, not deferred)

- macOS / Linux .app / AppImage packaging.
- A real auto-update mechanism.
- Code signing the binary (left to whoever distributes it).
- A CI workflow that builds the .exe — this plan only ships the build script; CI is a future iteration.

## Design choices

### 1. Tool: PyInstaller

PyInstaller is the de-facto choice for shipping Python apps as single-file binaries and the only one with first-class Windows support out of the box. Nuitka would give a smaller / faster exe but is fragile with `onnxruntime` and `watchdog`. Briefcase is geared toward GUI apps, not server apps.

Cross-compilation is a non-starter — PyInstaller must run on Windows to produce a `.exe`. The build script therefore targets Windows.

### 2. Auto-open browser

A `--browser/--no-browser` flag on `yadc webui serve`. Default `False` from a regular install (current behavior, no surprise browser popups for CLI users). The PyInstaller entry point injects `--browser` when (and only when) the user didn't already pass `--browser` / `--no-browser`, so the flag works from the .exe without changing the source-of-truth default.

Browser opens in a daemon thread that polls the bind socket for ~5 s before calling `webbrowser.open()` — this dodges the classic "browser opens before uvicorn is ready" race.

### 3. Frozen frontend path

`Configuration.app_frontend_build_path` defaults to a `Path(__file__).parent.parent / "webui" / "build"`. In a frozen build, `__file__` points into the unpacked `sys._MEIPASS`, **not** the user's install — so the default already resolves correctly: PyInstaller unpacks data files under `<_MEIPASS>/yadc/webui/build/` and we ship that same path layout via the spec. **No code change needed** — tested and confirmed by inspection. (If the layout ever changes, the resolver is one place to edit.)

CORS is **disabled** when running from the .exe (the SPA is served from the same origin via `AppBlueprint`); the entry point passes `--no-cors` to the underlying CLI.

### 4. Two variants — CPU and GPU

Both share the same spec structure. The build script maintains two venvs:

- `.build-cpu/` → `pip install -e . pyinstaller` → builds `yadc-webui-cpu.exe`
- `.build-gpu/` → `pip install -e ".[gpu]" pyinstaller` → builds `yadc-webui-gpu.exe`

`onnxruntime-gpu` does **not** shadow `onnxruntime` (different dist, same module). The spec includes both as a `hiddenimport` and lets the venv's installed package win. We exclude `uvloop` (Windows doesn't use it; forces pure-`asyncio` uvicorn).

### 5. Reuse `yadc webui serve`

No new click command. The entry point mutates `sys.argv` to inject `["webui", "serve", "--browser"]` (and removes/explicit `--no-cors`) then calls `yadc.cli:cli()`. All existing flags (`--host`, `--port`, `--log-level`, `--access-log-file`, …) keep working.

## Files

### New

| Path | Purpose |
|---|---|
| `scripts/entrypoints/webui_desktop.py` | Tiny PyInstaller entry script. Mutates `sys.argv`, then calls `yadc.cli:cli()`. |
| `scripts/build_exe.spec` | PyInstaller spec for the CPU build. |
| `scripts/build_exe_gpu.spec` | PyInstaller spec for the GPU build. |
| `scripts/build_windows_exe.ps1` | PowerShell driver: builds the SPA, creates two venvs, runs PyInstaller twice. |
| `scripts/build_windows_exe.sh` | Bash equivalent for the same script (so contributors on macOS/Linux can dry-run the front-end build and validate `pip install -e .[packaging]` resolves). The actual PyInstaller step still requires Windows. |
| `docs/desktop-exe.md` | User-facing doc — what the exe is, how to run, how to build, what gets downloaded on first launch. |
| `tests/cli/test_cli_webui_browser.py` | CliRunner tests for `--browser` / `--no-browser`. |
| `tests/test_frozen_paths.py` | Smoke tests for the entry script (importable, mutates argv correctly, resolves `_MEIPASS` path under a fake `sys.frozen`). |

### Modified

| Path | Change |
|---|---|
| `yadc/cli_webui.py` | Add `--browser/--no-browser` flag (default `False`). Add a small `_open_browser_when_ready()` helper that polls the bind socket then calls `webbrowser.open()`. Add `--no-cors`-friendly defaults so the existing flag still works. |
| `pyproject.toml` | Add `packaging` extra (`pyinstaller>=6.10.0`). |
| `docs/README.md` (if present) / `README.md` | Add a "Desktop app" section linking to `desktop-exe.md`. |

## Implementation phases

### Phase 1 — CLI flag

1. Add `--browser/--no-browser` to `yadc webui serve` with default `False`.
2. Add `_open_browser_when_ready(host, port)` helper in `cli_webui.py`: daemon thread that polls the bind socket for up to 5 s (50 × 100 ms) then calls `webbrowser.open(f"http://{host}:{port}")`.
3. Wire the flag into the command body — start the browser thread before `application.run()` returns.

### Phase 2 — Entry point

1. `scripts/entrypoints/webui_desktop.py`:
   - Detect frozen: `getattr(sys, "frozen", False)`.
   - If frozen:
     - Strip the script's own argv (`argv0` is the exe path; rest are user args).
     - If user passed neither `--browser` nor `--no-browser`, default to `--browser`.
     - If user did not pass `--cors/--no-cors`, default to `--no-cors`.
     - Prepend `["webui", "serve"]`.
   - `sys.exit(yadc.cli.cli(standalone_mode=True))`.
2. Write `tests/test_frozen_paths.py`:
   - Patch `sys.frozen = True` and `sys._MEIPASS = "<tmp>"` (via `monkeypatch`).
   - Import the entry script's `_build_argv(raw_argv)` helper (extract as a pure function) and assert it returns the expected argv in three cases (no args, with `--port=9000`, with `--no-browser`).
   - Assert the resolved frontend path is `<tmp>/yadc/webui/build`.

### Phase 3 — PyInstaller spec

1. `scripts/build_exe.spec`:
   - `Analysis(["scripts/entrypoints/webui_desktop.py"], …)`.
   - `datas`: bundles for banner, webui SPA, jinja templates, prompt-generation prompts, migrations. Use `collect_data_files("jinja2")` for Jinja's own internal templates.
   - `hiddenimports` / `collect_submodules`: `uvicorn.logging`, `uvicorn.loops`, `uvicorn.protocols.http.auto`, `quart.*`, `injector`, `watchdog.observers`, `watchdog.observers.inotify_buffer` (no-op on Windows but keep for analyzers), `huggingface_hub`, `keyring_pass`, `cryptography`, `onnxruntime.capi._pybind_state`, plus anything `collect_submodules` flags.
   - `excludes`: `tkinter`, `test`, `unittest`, `uvloop`, `pydoc`.
   - One-file EXE, console=True (so the user sees the uvicorn output).
2. `scripts/build_exe_gpu.spec`: identical except `name="yadc-webui-gpu"` and the venv step installs `.[gpu]`.

### Phase 4 — Build script

1. `scripts/build_windows_exe.ps1`:
   - `Push-Location yadc/webui; npm ci; npm run build; Pop-Location`
   - Spin up two venvs under `build/venv-cpu/` and `build/venv-gpu/`.
   - Install `.[packaging]` into each; install `.[gpu]` into the GPU venv.
   - Run PyInstaller twice: `pyinstaller scripts/build_exe.spec --distpath dist --workpath build/work-cpu -n yadc-webui-cpu --clean` (same for GPU).
   - Print paths to the two resulting `.exe` files.
2. `scripts/build_windows_exe.sh`: bash mirror that does the SPA build and the two venv setups, but **exits with a clear error** at the PyInstaller step unless `uname -s` matches something cross-compileable (it doesn't, so this is the dry-run harness).
3. `.gitignore` updates (root): ignore `build/`, `dist/`, `*.spec.bak` (the `yadc.egg-info/` and `__pycache__/` rules already cover most build artifacts). The spec files **are** tracked.

### Phase 5 — Tests

1. `tests/cli/test_cli_webui_browser.py`:
   - `runner.invoke(cli, ["webui", "serve", "--help"])` shows the new flag.
   - Mock the daemon thread + `webbrowser.open` and assert `_open_browser_when_ready` is a no-op when the flag is off, and *does* call `webbrowser.open` (with the right URL) when the flag is on. Pure unit test, no real network.
2. `tests/test_frozen_paths.py`: as above.
3. CI on Linux/macOS runs both test files; PyInstaller is imported only via the optional `packaging` extra (no PyInstaller import at runtime, so the test suite stays fast).

### Phase 6 — Docs

1. `docs/desktop-exe.md`:
   - What it is (single .exe, no Python install needed).
   - Two files: `yadc-webui-cpu.exe` and `yadc-webui-gpu.exe`. Recommend CPU for most users; only use GPU if you have an NVIDIA card + CUDA 12.x and want faster tagging.
   - Quick start: download, double-click, browser opens at `http://127.0.0.1:7860`.
   - First-launch caveat: the tagger downloads `SmilingWolf/wd-eva02-large-tagger-v3` (~1.3 GB) from HuggingFace Hub to `%USERPROFILE%\.cache\huggingface` on first use.
   - Data locations (already under `~/.cache/yadc`, `~/.local/share/yadc`, `~/.config/yadc` via `platformdirs`).
   - How to build from source (one-liner: `pwsh scripts/build_windows_exe.ps1`).
2. Add a "Desktop app" section near the top of the root `README.md` linking to the doc.

## Test plan (Phases 1, 2, 5)

- `uv run pytest tests/cli/test_cli_webui_browser.py tests/test_frozen_paths.py`
- Manual: build a Linux ELF locally as a smoke test for the spec layout (`pyinstaller` works on Linux too — different output, same spec). On Windows: run the build script, double-click the resulting `.exe`, confirm browser opens.
- Negative tests:
  - `serve --no-browser` does not call `webbrowser.open`.
  - `serve` from a `frozen` argv with `--port=9000` resolves to `http://127.0.0.1:9000`.
  - `--cors` from a frozen argv (no `--no-cors` injected) is preserved — i.e. our entry-point default of `--no-cors` is only injected when the user truly gave no indication.

## Risks / open questions

1. **Exe size**: CPU ~400-600 MB, GPU ~1-2 GB. Uvicorn pulls in `httptools`/`h2` etc. that bloat the binary. We may need to trim later, but ship first, optimize later.
2. **Tagger model download on first launch**: ~1.3 GB. Document loud and clear. A future iteration could ship a "lite" build that disables the tagger entirely.
3. **Hugging Face Hub rate limiting**: in theory an offline machine can't download the model. Fine for v1; offline mode is a future iteration.
4. **Antivirus false positives**: PyInstaller binaries routinely get flagged. Document a note about SmartScreen / Defender ("More info → Run anyway") — actually we can't fix it, but documenting it cuts support burden.
5. **Watchdog on Windows**: watchdog uses `ReadDirectoryChangesW` on Windows (no extra deps). Should "just work", but worth a smoke test once the exe runs.
6. **Code signing**: out of scope.
7. **Click + `--browser` default detection**: click parses `--no-browser` into `browser=False` correctly, but a user passing `--browser` *after* our auto-injected one wins (last one wins). Verify with a small test in the entry-point arg builder — covered in Phase 2.
