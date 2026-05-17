---
name: debug-api-logging
description: YADC_DEBUG_CAPTION_RESPONSES=1 feature for logging caption API request/response pairs to JSONL files.
---

# Debug Caption Response Logging

## Overview

When `YADC_DEBUG_CAPTION_RESPONSES=1` is set (checked at import time in `yadc/core/env.py` as `DEBUG_CAPTION_RESPONSES`), caption API request/response pairs are logged to JSONL files.

`YADC_DEBUG_CAPTION_REQUESTS_BODY` (default `"1"`) controls whether request bodies are included. When off, request body is `[REDACTED]` (avoids logging base64 image data). Response bodies are always logged.

Logging is **opt-in per-request** via `Session.capture_response()` context manager. Only captioning requests (chat/completions, generateContent) are wrapped — model loading, health checks, etc. are not logged.

## Activation Flow

1. `env.py` reads env vars at import time → `DEBUG_CAPTION_RESPONSES`, `DEBUG_CAPTION_REQUESTS_BODY`
2. CLI (`cli_caption.py`) calls `ResponseLogger.from_env(app.CACHE_PATH, dataset_paths)`
3. Logger is passed through `BaseAPICaptioner` → `Session.__init__(response_logger=...)`
4. Captioners (openai.py, gemini.py) wrap `self._session.post(...)` with `self._session.capture_response():`

## Directory Structure

```
~/.cache/yadc/api-debug/
  2026-05-17_14-30-00_my-animals_a1b2c3/
    001.jsonl
    002.jsonl
    ...
```

- **Run directory**: `{date}_{dataset_name}_{path_hash[:8]}`
  - `dataset_name`: first dataset entry path basename, with `dataset_` prefix stripped
  - `path_hash`: SHA-256[:8] of all dataset paths (null-joined)
- **Files**: `{NNN}.jsonl` — sequential counter, **resumed from existing files** on re-runs

## JSONL Envelope

Each file is a single JSON line:
```json
{
  "request": { "method": "POST", "url": "...", "headers": {...}, "body": {...} },
  "response": { "status": 200, "headers": {...}, "body": "..." },
  "stream": false
}
```

- Auth headers (`Authorization`, `x-goog-api-key`) are redacted
- For streaming responses, body is the accumulated SSE lines
- Files are written atomically (`.tmp` → rename)

## Files

- `yadc/core/env.py` — `DEBUG_CAPTION_RESPONSES` + `DEBUG_CAPTION_REQUESTS_BODY` flags
- `yadc/captioners/api/utils/response_logger.py` — `ResponseLogger` class + `DebugStreamProxy`
- `yadc/captioners/api/session.py` — `capture_response()` context manager + logging in `request()`
- `yadc/captioners/api/base.py` — passes `ResponseLogger` to `Session` constructor
- `yadc/captioners/api/openai.py` — wraps captioning POSTs with `capture_response()`
- `yadc/captioners/api/gemini.py` — wraps captioning POSTs with `capture_response()`
- `yadc/cli_caption.py` — creates `ResponseLogger` from env + dataset paths
