---
date: 2026-05-28
---
# Minor / Style Observations

**Context:** Minor observations from the code review that don't require action but are worth recording for future reference.

## Findings

1. **`UploadResponse` class is not exported** — `upload.ts` defines `UploadResponse` as an unexported class. The `ResponseLike` interface in `api.ts` is the public abstraction for `apiErrorMessage()`, which is the right call. If a consumer ever needs to type-narrow or inspect `UploadResponse` directly, it would need to be exported, but there's no current need.

2. **`readDirectoryEntries` batching** — The `do...while` loop in `FileDropZone.svelte` correctly handles `readEntries()` not returning all entries in one call (some browsers return at most 100 entries per call). This is easy to get wrong and was done right.

3. **No loading state during drag-parsing** — Dropping a large folder triggers recursive `webkitGetAsEntry` traversal, which could be slow for deeply nested directories with thousands of files. Currently there's no spinner or "reading files…" indicator during traversal. Low priority — can be added later if users report issues.

4. **Plan status was stale** — The plan index in `plan-management.md` still said "Not started" despite Phases 1–5 being complete. Updated to "Phases 1–5 done, review complete, fixes pending" as part of the review.
