# Desktop app (.exe)

yadc ships as a single Windows `.exe` for users who'd rather not install
Python themselves. Double-click it and a browser opens to the webui
automatically.

## What you get

Two binaries, both rebuilt from the same source — only the underlying
ONNX runtime differs:

| File | Use when |
|---|---|
| `yadc-webui-cpu.exe` | Most users. Small (~400 MB), runs on any Windows 10/11 machine. |
| `yadc-webui-gpu.exe` | You have an NVIDIA GPU + CUDA 12.x and want faster tagger inference. Much larger (~1-2 GB) and may fail on machines without CUDA. |

The webui itself is identical between the two; only the tagger inference
backend swaps. If in doubt, download `yadc-webui-cpu.exe`.

## Quick start (Windows users)

1. Download `yadc-webui-cpu.exe` from the project's release page.
2. Double-click it. A console window opens (it's normal — the server log
   lives there) and your default browser navigates to
   `http://127.0.0.1:7860`.
3. Set up an API environment in the webui (`Settings → API
   environments`), load a dataset, and caption.

That's it. No Python install, no `pip`, no Git.

### First launch

The first time the tagger is invoked, yadc downloads the
`SmilingWolf/wd-eva02-large-tagger-v3` ONNX model (~1.3 GB) from
HuggingFace Hub into `%USERPROFILE%\.cache\huggingface`. This is a
one-time cost — subsequent launches are instant. Internet access is
required for this first download; offline operation is not yet
supported by the bundled app.

### Where your data lives

The app uses `platformdirs`, so:

| Data | Path |
|---|---|
| API environments / templates | `%APPDATA%\yadc\` |
| Webui state (datasets, settings, prompt history) | `%LOCALAPPDATA%\yadc\webui.db` |
| Webui access log | `%LOCALAPPDATA%\yadc\webui-access.log` |
| Tagger model cache | `%USERPROFILE%\.cache\huggingface\` |

Delete these to reset the app to a clean state.

### Power-user flags

The `.exe` accepts the same flags as `yadc webui serve`. Power users can
launch it from a shell (`cmd`, PowerShell, Windows Terminal) to override
defaults:

```sh
yadc-webui-cpu.exe --port 9000 --no-browser --log-level debug
```

- `--port` — bind a different port.
- `--no-browser` — start the server without opening the browser.
  The server prints its URL to the console; click it.
- `--access-log-file` — write the per-request access log to a custom
  path (instead of the default rotating log under `LOCALAPPDATA`).
- `--log-level` — `debug` prints uvicorn's request flow.

The defaults auto-set by the desktop launcher are `--browser` (open
the SPA on launch) and `--no-cors` (the SPA is served same-origin, so
CORS isn't needed). Either can be overridden on the command line.

## Building from source

Build the .exe yourself on Windows. Requires:

- **Python 3.13** (any 3.13.x distribution).
- **Node.js 20+** (for the SvelteKit SPA build).
- **Microsoft Visual C++ Build Tools** (PyInstaller needs the Windows
  C runtime).
- About **5 GB of disk space** for the two venvs + PyInstaller work
  trees.

### Platform notes for the resulting binary

- **Windows** — ``console=True`` in the PyInstaller spec gives the
  binary a console window automatically. Double-click works out of
  the box; logs are visible in the window; Ctrl+C stops the server.
- **Linux/macOS** — double-clicking from a file manager creates no
  terminal (the OS just fork-execs with stdout going to a pipe). The
  entry point detects this and re-execs itself inside a terminal
  emulator. The terminal is chosen by:

  1. The ``YADC_DESKTOP_TERMINAL`` env var (full command override,
     e.g. ``YADC_DESKTOP_TERMINAL='my-term -e' ./yadc-webui-cpu``).
  2. ``gsettings get org.gnome.desktop.default-applications.terminal
     {exec,exec-args}`` — the desktop's "Default applications" setting.
     Covers GNOME, MATE, Cinnamon, Pop!_OS, and
     XFCE-with-gnome-settings-daemon. Resolves ``ptyxis`` on
     Fedora 39+/Ubuntu 24.04, ``gnome-terminal`` on Debian,
     ``konsole`` on KDE, etc.
  3. A hardcoded fallback list of well-known binaries on ``$PATH``:
     ``ptyxis``, ``gnome-terminal``, ``konsole``, ``xfce4-terminal``,
     ``mate-terminal``, ``tilix``, ``foot``, ``alacritty``, ``kitty``,
     ``wezterm``, ``deepin-terminal``, ``xterm``,
     ``x-terminal-emulator`` (the freedesktop.org alias).

  The re-spawned binary sees a real TTY and behaves like it was
  launched from a shell. If no terminal emulator is found the server
  still starts — just headless, with logs going to ``/dev/null``;
  run from a shell to see them. The auto-respawn only kicks in when
  ``sys.stdout`` is not a TTY at startup, so launching from a shell
  never spawns a second window. Recursion is guarded by ``YADC_DESKTOP_IN_TERMINAL``.

### One-shot

From a `git clone` of this repo on Windows (PowerShell 7+):

```pwsh
uv sync --extra packaging                  # or: pip install -e ".[packaging]"
pwsh scripts/build_windows_exe.ps1
```

After a few minutes you'll find:

- `dist\yadc-webui-cpu.exe`
- `dist\yadc-webui-gpu.exe`

### Iterating on the bundle

Useful flags:

```pwsh
pwsh scripts/build_windows_exe.ps1 -SkipNpmBuild    # reuse existing yadc/webui/build/
pwsh scripts/build_windows_exe.ps1 -SkipGpu         # CPU build only
```

The script:

1. Builds the SPA (`npm ci && npm run build`).
2. Creates two isolated venvs under `build\venv-cpu\` and
   `build\venv-gpu\`. Each gets `yadc` installed from the current
   working tree, plus PyInstaller.
3. Runs PyInstaller twice — once per spec file — into `dist\`.
4. Prints the final paths.

Both venvs are throwaway; if a build is broken, delete `build\` and
re-run.

### What gets bundled

A regular `pip install yadc` on Windows already produces a working
`yadc.exe` script — the desktop build is one layer above that. It
adds:

1. **All Python dependencies** for the webui (Quart, uvicorn,
   onnxruntime, huggingface_hub, cryptography, …).
2. **The SvelteKit SPA** under `yadc\webui\build\`.
3. **Data files** the API loads at runtime — Jinja2 templates, SQL
   migrations, the startup banner.
4. **A desktop-friendly entry point** (`scripts\entrypoints\webui_desktop.py`)
   that injects `--browser` and `--no-cors` defaults before
   dispatching to the regular `yadc` CLI.

The result is one self-contained file. Move it to another Windows
machine, double-click, and it runs — no Python, no Node, no install.

## Troubleshooting

### "Windows protected your PC" / SmartScreen prompt

PyInstaller binaries lack an Authenticode signature, so SmartScreen
warns on first launch. Click **More info → Run anyway**. Code signing
the binary is out of scope for this build.

### Browser never opens

The auto-open helper polls the bind socket for ~5 s after launch. If
the port is already in use, the helper gives up — the server still
runs, just without the browser popup. Open
`http://127.0.0.1:<port>` (printed to the console at startup)
manually.

To force-skip the browser from a one-off shell launch:

```sh
yadc-webui-cpu.exe --no-browser
```

### Port already in use

The default port (7860) might be taken by another service. Pick
another with `--port`. The console window logs the actual bind
address at startup.

### First tagger request never completes / "downloading model…"

The first tagger invocation pulls ~1.3 GB from
`huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3`. On a slow
connection this takes a few minutes; the webui's progress bar in the
bottom-left reflects download state. Re-launching the .exe cancels
any in-progress tagger jobs; the model is cached, so subsequent
launches are fast.

### Antivirus quarantines the .exe

Some AV vendors flag PyInstaller binaries outright. The build is open
source — feel free to add a code-signing certificate or submit a
false-positive report to your AV vendor. As of this writing we don't
ship a signed binary.

### Build fails with "ModuleNotFoundError: pyinstaller"

PyInstaller is in the `packaging` extra, not `dev`:

```sh
uv sync --extra packaging
# or
pip install -e ".[packaging]"
```

Then re-run `scripts\build_windows_exe.ps1`.

### Build fails with "Bundling yadc/webui/build' ... not found"

The build script runs `npm run build` first; if you passed
`-SkipNpmBuild` without having built the SPA yet, PyInstaller will
fail. Run `npm ci && npm run build` in `yadc\webui\` once and retry.

## Repository layout

```
scripts/
├── entrypoints/
│   └── webui_desktop.py    # PyInstaller entry point — frozen-aware CLI shim
├── build_exe.spec          # CPU PyInstaller spec
├── build_exe_gpu.spec      # GPU PyInstaller spec
├── build_windows_exe.ps1   # Windows build driver
└── build_windows_exe.sh    # bash mirror (SPA build only; gates the
                            # PyInstaller step behind a Windows check)
```
