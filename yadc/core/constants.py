"""Shared constants importable from any layer (cmd, api, cli).

Filesystem-extension sets used by the dataset scanner / filesystem
watcher / upload pipeline (api) and the prompt-generation examples
resolver (cmd). Kept here so the cmd layer (which can't import from
api) and the api services share one definition.
"""

import base64

# Image extensions recognised by PIL — used across layers.
IMAGE_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif", ".ico"})

# Sidecar files attached to an image stem: captions (.txt), dataset
# config (.toml), and edit drafts. The watcher combines these with
# ``IMAGE_EXTENSIONS`` to decide which files it observes.
SIDECAR_EXTENSIONS: frozenset[str] = frozenset({".txt", ".toml", ".history~", ".draft~"})
# Glob suffixes for sidecars when attached to a stem (used in expect_pattern_change).
# .txt / .toml / .history~ are exact-match patterns; .*.draft~ matches named drafts.
SIDECAR_EXTENSION_GLOBS: tuple[str, ...] = (".txt", ".toml", ".history~", ".*.draft~")

# NOTE: using base64 encoding so coding agents reading this don't get confused
# if they use the same tokens when working on the codebase
DEFAULT_THINKING_START = base64.b64decode("PHRoaW5rPgo=").decode()
DEFAULT_THINKING_END = base64.b64decode("PC90aGluaz4K").decode()
