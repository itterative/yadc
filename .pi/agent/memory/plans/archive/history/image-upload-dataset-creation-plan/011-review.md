---
date: 2026-05-28
---
# Code Review: Image Upload Dataset Creation

**Context:** All phases (1–5) implemented across 16 temporary commits. Review requested before merging.

**Scope:** Backend (`api_datasets.py`, `datasets.py`, `configuration.py`, `utils_json.py`), Frontend (`upload.ts`, `api.ts`, `datasetImages.ts`, `FileDropZone.svelte`, `AddDatasetDialog.svelte`).

## Verdict

Solid implementation. All linting (ruff, eslint) and type checking (basedpyright, svelte-check) pass cleanly. The architecture follows project conventions — clean separation of concerns, proper error handling, good UX touches (progress bar, cancel, drag-and-drop, validation warnings).

## Findings

### Must fix before merge

1. **Partial upload cleanup** — `create_dataset_from_upload()` creates `images/` and `folders/` directories before writing files. If the method raises mid-way (bad file, I/O error, disk full) or `_register()` fails after files are written, orphaned data remains in `STATE_PATH/<name>/`. Fix: wrap the entire method body in `try/except` that calls `shutil.rmtree(base_dir, ignore_errors=True)` on any exception, then re-raises. This implements Phase 6 Option B from the plan.

2. **`.` filename edge case** — `PurePosixPath('.').parts == ('.',)` has length 1, so it enters the root file branch. The `resolve()+relative_to()` traversal check passes (a directory is within itself). Then `open(dest_path, "wb")` raises `IsADirectoryError`, which surfaces as an unhandled 500. Fix: guard against `.` and empty filenames before the branch logic.

3. **Inconsistent path sanitization in root branch** — Root files use `(images_dir / filename).resolve()` with raw user input, while folder files use `(folders_dir / str(pure)).resolve()` with the sanitized `PurePosixPath`. Both branches should use `str(pure)` (or `pure.name` for root files) for consistency.

### Should fix

4. **Size limit is best-effort only** — `request.content_length` is the `Content-Length` header: client-controlled and optional. A malicious client can bypass it by omitting the header. The backend writes all files to disk regardless. Fix: add an accumulated-size counter inside the file write loop that raises `ValueError` if `max_upload_size_bytes` is exceeded.

5. **Duplicate `formatBytes()` function** — Identical implementation in `FileDropZone.svelte` and `AddDatasetDialog.svelte`. Extract to a shared utility (e.g., `$lib/format.ts` or add to `$lib/index.ts`).

6. **Tab order vs. default mode mismatch** — The Upload tab renders first in `PillTabs` but `mode` defaults to `'import'`, so the visual first tab doesn't match the selected tab on dialog open. Fix: either change the default `mode` to `'upload'` or reorder the tabs so Import renders first.

### Acknowledged / deferred

- **`UploadResponse` not exported** — Fine for now. `ResponseLike` is the public abstraction for `apiErrorMessage()`.
- **No loading state during drag-parsing** — Recursive `webkitGetAsEntry` traversal on large folders could be slow. Can add a spinner later if users report issues.
- **Phase 5b (distinguish uploaded vs imported datasets)** — Still pending, tracked separately in the plan.

## Things done well

- Path traversal defense-in-depth is correct (`resolve()` + `relative_to()` catches `..`, absolute paths, symlinks, normalization tricks).
- Clean architecture: `upload.ts` (generic XHR), `FileDropZone.svelte` (reusable component), `datasetImages.ts` (API helper), `AddDatasetDialog.svelte` (orchestration).
- Error handling: `AbortError` caught specifically, `ResponseLike` interface decouples from `fetch` Response, backend uses `ErrorCode` enum.
- UX: progress bar, cancel upload, file type validation with warning banner, drag-and-drop with nested element counter.
- Documentation: excellent history trail (10 entries) with every decision recorded.
