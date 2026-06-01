---
date: 2026-06-01
---
# Upload Pipeline Uses Layout Constants

Follow-up polish after history entry 029. The "follow-up" note there flagged
string literals `"images"` and `"folders"` in `dataset_upload.py` (and
matching literals in `dataset_upload_staging.py`) as inconsistent with the
new `managed_paths.py` module. This entry records that the polish was
applied.

## What changed

### `yadc/api/services/dataset_upload.py`

Added the import:

```python
from yadc.api.services.managed_paths import MANAGED_FOLDERS_PREFIX, MANAGED_IMAGES_PREFIX
```

Replaced 11 string-literal usages across all three methods
(`create_dataset_from_upload`, `append_dataset_from_upload`,
`commit_staged_upload`):

- `base_dir / "images"` and `base_dir / "folders"` for the real managed
  layout paths.
- `staging_base / "images"` and `staging_base / "folders"` for the staging
  mirror layout.
- `{"path": "images"}` and `{"path": f"folders/{folder_name}"}` for the
  `[[dataset]]` config entries written to TOML.
- `root_rel = "images"` and `folder_rel = f"folders/{folder_name}"` in the
  commit-time dedup against existing config entries.

The only remaining literal occurrence in the file is one comment that
documents the actual TOML values for the reader — left intentionally.

### `yadc/api/services/dataset_upload_staging.py`

Already updated in entry 029 (the staging helpers use the constants in
`build_staged_groups` and `_staged_path_to_relative`). No further changes
here.

## Verification

- All 253 tests pass.
- `ruff check` and `ruff format --check` clean.
- A repo-wide grep for `"images"` and `"folders"` now returns only the
  two canonical definitions in `managed_paths.py` and one informative
  comment in `dataset_upload.py`.

## Why this matters

`MANAGED_IMAGES_PREFIX` and `MANAGED_FOLDERS_PREFIX` are now the single
source of truth for the managed-dataset layout convention. Every reference
in the upload pipeline — paths, TOML values, staging mirrors — derives
from them. If the convention ever changes, there's exactly one place to
update.
