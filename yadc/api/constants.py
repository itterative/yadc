"""Cross-module constants for the yadc API.

Single source of truth for values that multiple services and
modules need to agree on. Keep this module dependency-free
(no imports from other yadc modules) so it can be imported
anywhere without circular-import concerns.
"""

from __future__ import annotations

# Image extensions recognised by the dataset scanner and the
# filesystem watcher. Kept in sync with what PIL can open. Both
# the scanner (``DatasetScanner.scan_image_meta`` /
# ``read_disk``) and the watcher (``DatasetWatcherService`` /
# ``_is_watched``) consult this set; the upload service derives
# its full ``UPLOAD_EXTENSIONS`` from it.
IMAGE_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif", ".ico"})

# Extensions we care about — both images and sidecars. The image
# set is the single source of truth in :mod:`yadc.api.constants`;
# the watcher adds its private sidecar sets on top.
SIDECAR_EXTENSIONS: frozenset[str] = frozenset({".txt", ".toml", ".history~", ".draft~"})
# Glob suffixes for sidecars when attached to a stem (used in expect_pattern_change).
# .txt / .toml / .history~ are exact-match patterns; .*.draft~ matches named drafts.
SIDECAR_EXTENSION_GLOBS: tuple[str, ...] = (".txt", ".toml", ".history~", ".*.draft~")
