# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the CPU desktop build.

Produces a one-file ``yadc-webui-cpu.exe`` that bundles the Quart +
SvelteKit webui along with the standard CLI. Double-click it (on
Windows, or run it headless from a shell) and a browser opens to the
webui automatically. The matching GPU spec differs only in the venv
that runs it — see ``scripts/build_windows_exe.ps1``.

**Build prerequisites**
- Python 3.13 on Windows.
- ``pip install -e .[packaging]`` (project + ``pyinstaller>=6.10``).
- ``yadc/webui/build/`` populated via ``npm ci && npm run build``.

**Run prerequisites for the end user**
- Nothing — Python is bundled. The first launch downloads the
  SmilingWolf WD tagger model (~1.3 GB) from HuggingFace Hub.
"""

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Repository layout — the spec runs from the repo root via PyInstaller,
# so all paths are anchored at ``Path('.').resolve()``.
ROOT = Path('.').resolve()
ENTRY = str(ROOT / 'scripts' / 'entrypoints' / 'webui_desktop.py')

# Bundled data files. Each tuple is (source-on-disk, destination-in-bundle).
# PyInstaller preserves the destination path inside the bundle so we
# can resolve it at runtime via ``_MEIPASS/<destination>``.
DATA_FILES = [
    # SvelteKit SPA — Quart's ``AppBlueprint`` reads this at request
    # time. The default ``Configuration.app_frontend_build_path``
    # resolves to ``<package>/webui/build`` and is ``_MEIPASS``-relative
    # in the frozen build; we mirror that path by setting the bundle
    # destination to ``yadc/webui/build`` (NOT ``yadc/webui`` — PyInstaller
    # collapses a source/dest with the same trailing segment and drops
    # the ``build`` part, putting the SPA at ``yadc/webui/_app/...``
    # where :class:`Configuration.app_frontend_build_path` can't find it).
    (str(ROOT / 'yadc' / 'webui' / 'build'), 'yadc/webui/build'),

    # SQLite migrations — discovered by DBMigrations at startup via
    # ``importlib.resources``; PyInstaller's resource tracking is
    # reliable for SQL files but PyInstaller sometimes misses dynamic
    # loading patterns, so we belt-and-brace with explicit datas.
    (str(ROOT / 'yadc' / 'api' / 'migrations'), 'yadc/api/migrations'),

    # Misc textual data shipped via the python package.
    (str(ROOT / 'yadc' / 'api' / 'banner.txt'), 'yadc/api'),
    # Destination must mirror the package path (``yadc/templates/jinja``)
    # because ``yadc/templates/jinja/__init__.py`` makes it a real package
    # and the loader resolves it via ``resources.files('yadc.templates.jinja')``.
    # A plain ``yadc/templates`` destination would copy ``default.jinja`` into
    # ``yadc/templates/`` instead of ``yadc/templates/jinja/``.
    (str(ROOT / 'yadc' / 'templates' / 'jinja'), 'yadc/templates/jinja'),
    (str(ROOT / 'yadc' / 'prompt_generation' / 'prompts'), 'yadc/prompt_generation/prompts'),
]

# Jinja2 keeps templates in its own package — ``collect_data_files``
# grabs everything under ``jinja2`` regardless of layout. Other deps
# handle their own template/data shipping without help.
DATA_FILES += collect_data_files('jinja2')

# Hidden imports. PyInstaller's static analysis covers most of yadc's
# codebase, but a handful of modules use dynamic-import patterns it
# doesn't follow. List them by dotted path; ``collect_submodules`` is
# used where the namespace contains enough modules that hand-listing
# would be brittle.
HIDDEN_IMPORTS = [
    # Quart routes / blueprints are wired by ``yadc.api`` auto-discovery
    # at import time, but individual controller decorators are not always
    # visible to PyInstaller's analysis.
    *collect_submodules('yadc.api.controllers'),
    *collect_submodules('yadc.api.services'),
    *collect_submodules('yadc.api.modules'),

    # uvicorn dynamically imports loop / protocol implementations.
    'uvicorn.logging',
    'uvicorn.loops',
    'uvicorn.loops.asyncio',
    'uvicorn.protocols.http.auto',
    'uvicorn.protocols.websockets.auto',
    'uvicorn.lifespan.on',

    # watchdog picks an observer backend at runtime (ReadDirectoryChangesW
    # on Windows). Listing the ``inotify_buffer`` Linux variant is
    # harmless on Windows; PyInstaller doesn't pull it in otherwise.
    'watchdog.observers',
    'watchdog.observers.inotify_buffer',
    'watchdog.observers.read_directory_changes',

    # huggingface_hub lazy-imports its storage backends on first use.
    'huggingface_hub',
    'huggingface_hub.file_download',

    # Keyring backends — ``keyring-pass`` chains through them.
    'keyring_pass',
    'keyring.backends.fail',
    'keyring.backends.null',

    # onnxruntime's native binding module — PyInstaller's analysis
    # misses the ``capi._pybind_state`` loader.
    'onnxruntime',
    'onnxruntime.capi._pybind_state',

    # cryptography pulls in Rust-built OpenSSL bindings.
    'cryptography',
    'cryptography.hazmat.bindings._rust',
]

# Modules we explicitly don't need in the bundle. Excluding ``uvloop``
# keeps uvicorn on the pure-asyncio loop, which is the only option on
# Windows anyway and avoids importing a C extension that won't load.
EXCLUDES = [
    'tkinter',
    'tkinter.ttk',
    'test',
    'unittest',
    'pydoc',
    'uvloop',
    'PyInstaller',
]

a = Analysis(
    [ENTRY],
    pathex=[str(ROOT)],
    binaries=[],
    datas=DATA_FILES,
    hiddenimports=HIDDEN_IMPORTS,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=EXCLUDES,
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

# One-file exe, console=True so the user sees the uvicorn banner in the
# spawned terminal. ``disable_windowed_subprocess_picker`` keeps the
# SmartScreen prompt behind a single click instead of three.
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='yadc-webui-cpu',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_subprocess_picker=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
