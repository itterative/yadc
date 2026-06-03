---
date: 2026-05-29
---
# Streaming Validation Progress — Approach Selection

**Context:** Phase 13 identifies that large uploads have a silent gap after the HTTP body is received: validation (PIL verify, TOML parse, orphan sidecar checks) and file writing can take several seconds with no UI feedback. The upload progress bar completes but the server is still working.

**Decision:** Use **Option A — chunked streaming NDJSON response via `XMLHttpRequest`**. Extend the existing `upload.ts` XHR wrapper to support streaming response parsing. The upload endpoint returns NDJSON lines, and the frontend reads them incrementally via `xhr.responseText` in download `onprogress` events.

**Options evaluated:**

| Option | Description | Verdict |
|--------|-------------|---------|
| **A: Chunked NDJSON response (XHR)** | Single `POST /api/datasets/upload` streams progress as NDJSON. Extend existing XHR wrapper to parse response text incrementally during download progress events. | **Chosen** — cleanest, keeps upload progress, extends existing code |
| **B: XHR upload → separate progress stream** | Upload returns 202 + job_id, client connects to a second streaming endpoint | Rejected — requires server-side job state management, two requests |
| **C: Polling** | 202 + job_id, client polls status endpoint | Rejected — highest latency, still needs job state |
| **D: Global SSE broadcast** | Dispatch upload progress through existing SSE system | Rejected — broadcast to all clients for a single-client operation; pollutes event namespace |

**Rationale for A with XHR:**
- **Request-scoped:** No job ID, no background task coordination, no cleanup on disconnect. The progress stream lives and dies with the HTTP request.
- **Single request:** Upload + validation + writing are one continuous stream.
- **Keeps upload progress:** XHR `upload.onprogress` already tracks upload bytes. No need for counting ReadableStream hacks.
- **Streaming response via XHR:** `xhr.responseText` is readable incrementally during download `onprogress` events. Track bytes already parsed, extract new NDJSON lines on each progress tick. Well-established pattern.
- **Extends existing code:** The `upload.ts` wrapper already handles XHR lifecycle, abort, progress. Add an `onChunk` callback for streaming NDJSON parsing.
- **Wire format:** NDJSON (`application/x-ndjson`) — one JSON object per line, flushed after each event.
- **Quart support:** Quart already supports async generator responses (same pattern as `api_events.py` SSE endpoint).

**Implementation outline:**

Backend (`dataset_upload.py`):
- `create_dataset_from_upload()` becomes an async generator yielding `UploadProgressEvent` dataclasses (or accepts a callback that the controller wraps).
- Progress events: `{"phase": "validating", "file": "cat.jpg", "index": 5, "total": 100}`, `{"phase": "writing", ...}`, `{"phase": "complete", "dataset": {...}, "warnings": [...]}`, `{"phase": "error", "message": "..."}`.

Backend (`api_datasets.py`):
- Upload endpoint returns a `Response` with an async generator that yields NDJSON lines, `Content-Type: application/x-ndjson`.

Frontend (`upload.ts`):
- Add `onChunk` callback to `UploadOptions`. During XHR download `onprogress`, read `xhr.responseText` from the last parsed offset, split on newlines, parse each complete NDJSON line, call `onChunk(parsedEvent)`.
- Keep existing `onProgress` for upload progress unchanged.

Frontend (`datasetImages.ts`):
- `uploadDataset()` passes `onChunk` to `upload()`, translates progress events into a unified progress type for the caller.

Frontend (`UploadDatasetTab.svelte`):
- Progress bar transitions from "Uploading…" to "Validating…" to "Writing…" based on the `phase` field.
- Shows current file name and N/M count during validation/writing phases.
