# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the GPU desktop build.

Identical to ``scripts/build_exe.spec`` except for the artifact name;
the only thing that actually differs between CPU and GPU builds is
the Python environment the bundler runs against. Build the GPU venv
with ``pip install -e ".[gpu,packaging]"`` instead of
``pip install -e ".[packaging]"`` — ``onnxruntime-gpu`` shadows
``onnxruntime`` and PyInstaller picks up whichever the venv contains.

Keeping the spec fork-tiny makes it easy to diff between CPU and GPU
without diverging in subtle ways (e.g. hidden imports that apply to
one variant but not the other).
"""

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = Path('.').resolve()
ENTRY = str(ROOT / 'scripts' / 'entrypoints' / 'webui_desktop.py')

# Same data layout as the CPU build. PyInstaller repackages the SPA,
# migrations, jinja templates and prompt-generation text the same way;
# the runtime path resolution is identical.
DATA_FILES = [
    # See the comment in scripts/build_exe.spec for why this is
    # ``yadc/webui/build`` rather than ``yadc/webui`` (a symmetric
    # source/dest would have PyInstaller collapse the ``build``
    # segment, breaking the runtime's frontend path lookup).
    (str(ROOT / 'yadc' / 'webui' / 'build'), 'yadc/webui/build'),
    (str(ROOT / 'yadc' / 'api' / 'migrations'), 'yadc/api/migrations'),
    (str(ROOT / 'yadc' / 'api' / 'banner.txt'), 'yadc/api'),
    (str(ROOT / 'yadc' / 'templates' / 'jinja'), 'yadc/templates'),
    (str(ROOT / 'yadc' / 'prompt_generation' / 'prompts'), 'yadc/prompt_generation/prompts'),
]
DATA_FILES += collect_data_files('jinja2')

HIDDEN_IMPORTS = [
    *collect_submodules('yadc.api.controllers'),
    *collect_submodules('yadc.api.services'),
    *collect_submodules('yadc.api.modules'),
    'uvicorn.logging',
    'uvicorn.loops',
    'uvicorn.loops.asyncio',
    'uvicorn.protocols.http.auto',
    'uvicorn.protocols.websockets.auto',
    'uvicorn.lifespan.on',
    'watchdog.observers',
    'watchdog.observers.inotify_buffer',
    'watchdog.observers.read_directory_changes',
    'huggingface_hub',
    'huggingface_hub.file_download',
    'keyring_pass',
    'keyring.backends.fail',
    'keyring.backends.null',
    # GPU build uses ``onnxruntime-gpu`` instead of ``onnxruntime``;
    # the module path is the same so the import names match the CPU
    # spec — the venv's installed distribution decides which package's
    # native library is bundled.
    'onnxruntime',
    'onnxruntime.capi._pybind_state',
    'cryptography',
    'cryptography.hazmat.bindings._rust',
]

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

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='yadc-webui-gpu',
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
