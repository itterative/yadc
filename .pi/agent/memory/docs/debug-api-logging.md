---
name: debug-api-logging
description: YADC_DEBUG_CAPTION_RESPONSES=1 feature for logging caption API request/response pairs to JSONL files.
category: workflow
---

# Debug Caption Response Logging

## Overview

When `YADC_DEBUG_CAPTION_RESPONSES=1` is set (checked at import time in `yadc/core/env.py` as `DEBUG_CAPTION_RESPONSES`), caption API request/response pairs are logged to JSONL files.

`YADC_DEBUG_CAPTION_REQUESTS_BODY` (default `"0"`) controls whether request bodies are included. When off, request body is `[REDACTED]` (avoids logging base64 image data). Response bodies are always logged.

Logging is **opt-in per-request** via `Session.capture_response()` context manager. Only captioning requests (chat/completions, generateContent) are wrapped — model loading, health checks, etc. are not logged.

## Activation Flow

1. `env.py` reads env vars at import time → `DEBUG_CAPTION_RESPONSES`, `DEBUG_CAPTION_REQUESTS_BODY`
2. CLI (`cli_caption.py`) checks env vars, calls `cmd_cache.debug_log_dir(toml_path)` to compute the run directory, then creates `ResponseLogger(log_dir, log_body=...)`
3. Logger is passed through `BaseAPICaptioner` → `Session.__init__(response_logger=...)`
4. Captioners (openai.py, gemini.py) wrap `self._session.post(...)` with `self._session.capture_response()`

## Directory Structure

```
~/.cache/yadc/api-debug/
  2026-05-17/
    my_dataset/
      000001_my_photo.jsonl
      000002_another_image.jsonl
    ...
```

- **Date directory**: `{YYYY-MM-DD}` — groups runs by date
- **Run directory**: `{dataset_name}` — TOML filename stem with `dataset_` prefix stripped
  - Path is computed by `cmd_cache.debug_log_dir()` using `app.CACHE_PATH`
  - Created eagerly at `ResponseLogger` construction time
- **Files**: `{NNNNNN}_{image_name}.jsonl` — 6-digit sequential counter + image stem name, **resumed from existing files** on re-runs

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
- `yadc/cmd/cache/cache.py` — `debug_log_dir()` computes the run directory path
- `yadc/llm/response_logger.py` — `ResponseLogger` class (pure writer, accepts `log_dir`) + `DebugStreamProxy`
- `yadc/captioners/api/session.py` — `capture_response()` context manager + logging in `request()`
- `yadc/captioners/api/base.py` — passes `ResponseLogger` to `Session` constructor
- `yadc/captioners/api/openai.py` — wraps captioning POSTs with `capture_response()`
- `yadc/captioners/api/gemini.py` — wraps captioning POSTs with `capture_response()`
- `yadc/cli_caption.py` — reads env vars, creates `ResponseLogger` and passes to captioner
