# Phase 4 Planning: Sidecar File Upload

## Summary

Added Phase 4 to the image-upload-dataset-creation plan: allow uploading `.txt` and `.toml` sidecar files alongside images.

## Rationale

- `.txt` files placed next to images provide existing captions (already supported by dataset scanner).
- `.toml` files placed next to images provide per-image extras/context during captioning (already supported by dataset scanner).
- Currently the upload endpoint only accepts image extensions, which prevents users from uploading datasets that include pre-existing captions or extras.

## Plan Details

### Backend Changes (when implemented)
- Relax extension validation in `DatasetService.create_dataset_from_upload()` to allow `.txt` and `.toml` in addition to image extensions.

### Frontend Changes (when implemented)
- Remove `accept="image/*"` from the file picker input (folder picker already ignores it).
- Update help text to mention `.txt` and `.toml` sidecar support.
- Optionally warn if no image files are detected in the selection.

## Files Modified

- `.pi/agent/memory/plans/archive/image-upload-dataset-creation-plan.md`:
  - Added Phase 4 section
  - Updated API contract to note current vs. Phase 4 allowed file types
