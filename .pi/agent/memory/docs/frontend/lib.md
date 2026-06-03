---
name: frontend/lib
description: Frontend lib/ root modules — index.ts re-exports, api.ts error parsing, TypedEventSource SSE, async helpers, notifications, upload wrapper, formatters, localStorage store, seeded PRNG.
category: architecture
---

# Frontend: `lib/` Root Modules

Shared modules at `yadc/webui/src/lib/` root (not in sub-folders).

```
yadc/webui/src/lib/
  index.ts             # Re-exports: storable, async helpers, TypedEventSource, API_BASE, random
  api.ts               # API_BASE constant, structured error response parsing (isAPIErrorResponse / isAPIErrorDetail / apiErrorMessage), and user-friendly HTTP status messages. Exports `ResponseLike` interface so `apiErrorMessage()` works with both native `fetch` `Response` and `UploadResponse`.
  events.ts            # TypedEventSource — SSE with Zod validation
  async.ts             # debounce (async debounce+dedupe), sleep, synchronized, delayed helpers
  notifications.ts     # Browser Notification API helpers (permission, sending, first-use prompt)
  storable.js          # localStorage-backed writable store
  random.ts            # Seeded PRNG for deterministic stub layouts
  upload.ts            # Promise-based XMLHttpRequest wrapper with upload progress tracking and AbortSignal support. Returns `UploadResponse` which implements `ResponseLike`.
  format.ts            # Shared formatting utilities — `formatBytes(bytes)` for human-readable file sizes
```

**Cross-references:**
- SSE (TypedEventSource, Zod, resumption): `frontend-patterns` and `dataset-watcher`
- Notifications: `frontend-patterns`
- Upload wrapper (XMLHttpRequest, progress): `dataset-system` (upload pipeline)
