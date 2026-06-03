---
name: captioner-async-migration-plan
description: Migrate captioners and CaptioningService to asyncio for mid-flight cancellation via httpx, while preserving sync CLI paths.
status: Complete
---

# Captioner Async Migration Plan

**Goal:** Enable mid-flight cancellation of API requests when the user clicks **Stop Captioning** in the WebUI, by migrating the captioner HTTP layer and `CaptioningService` to `asyncio` + `httpx`. Sync CLI paths remain untouched.

**Background:** Currently `CaptionJob` runs in a `threading.Thread` and checks `_stop_requested` only *between* images. `requests` blocks the thread during the HTTP call, so stopping has to wait for the current image's API response to complete. `httpx.AsyncClient` supports true cancellation via `task.cancel()`, which closes the in-flight connection immediately.

**Key constraint:** `CaptioningService` is used **only** by the Web API (Quart). The CLI (`cli_caption.py`) instantiates `APICaptioner` directly and calls sync `predict()` / `predict_stream()`. Therefore we can make `CaptioningService` fully async without impacting the CLI, provided we keep sync methods on the captioner hierarchy.

---

## Phase 1: Async HTTP Session (`AsyncSession`)

**New file:** `yadc/captioners/api/async_session.py`

- Create `AsyncSession` class that mirrors `Session`'s public API:
  - `__init__(base_url, headers=..., max_retries=..., cache=..., response_logger=...)`
  - `async def arequest(method, path, *, capture_ctx=..., **kwargs)` → context manager yielding `httpx.Response`
  - `async def aget(path, cache_ttl=..., **kwargs)`
  - `async def apost(path, *, capture_ctx=..., **kwargs)`
  - `capture_response(image_name)` → yields `CaptureContext` (same class, reuse it)
- Use `httpx.AsyncClient` with `httpx.AsyncHTTPTransport` for retries.
- Keep response logging identical to sync `Session` (same `ResponseLogger` interface).
- Cache (`HTTPResponseCache`) is file-based and can be accessed synchronously from async code safely.
- **Dependency change:** Add `httpx` to `pyproject.toml`.
- **Sync `Session` stays untouched.** CLI continues using `requests`.

### Open questions
- Should we eventually replace `requests` with `httpx.Client` in `Session` for uniformity? **Deferred** — not required for this plan.

---

## Phase 2: Async methods on base classes

**Files:** `yadc/core/captioner.py`, `yadc/captioners/api/base.py`

### `Captioner` ABC
Add async methods with **default implementations** that delegate to sync via `asyncio.to_thread()`:

```python
async def apredict(self, image: DatasetImage, **kwargs: Any) -> str:
    return await asyncio.get_running_loop().run_in_executor(None, functools.partial(self.predict, image, **kwargs))

async def apredict_stream(self, image: DatasetImage, **kwargs: Any) -> AsyncGenerator[str, None]:
    # Delegate sync generator via asyncio.to_thread + queue, or skip if unused
    ...
```

This means **backends are not forced to implement async immediately**.

### `BaseAPICaptioner`
- Add optional `_async_session: AsyncSession | None`.
- If `async_session` kwarg is provided, store it; otherwise `apredict` delegates to sync via the default ABC implementation.

---

## Phase 3: True async OpenAI captioner

**File:** `yadc/captioners/api/openai.py`

OpenAI is the reference backend and the parent of most others (Ollama, vLLM, llama.cpp, KoboldCpp, OpenRouter). Migrating it gives true async to ~5 backends at once.

- Add `async def _agenerate_prediction()` — mirrors `_generate_prediction()` but uses `await self._async_session.post(...)`.
- Add `async def _agenerate_stream_prediction()` — mirrors `_generate_stream_prediction()` but iterates the `httpx` stream asynchronously.
- Add `async def apredict()` — calls `_handle_thinking(await self._agenerate_prediction(...))`.
- Add `async def apredict_stream()` — wraps `_agenerate_stream_prediction()` in an async generator.
- **Sync methods stay exactly as they are.**

Since subclasses only override `conversation()` (a pure data-builder), they inherit true async `apredict` / `apredict_stream` automatically.

### Exception mapping
Map `httpx.HTTPStatusError` → `ValueError` the same way sync maps `requests.HTTPError`.

---

## Phase 4: True async Gemini captioner

**File:** `yadc/captioners/api/gemini.py`

- Same pattern as OpenAI: add `_agenerate_prediction`, `_agenerate_stream_prediction`, `apredict`, `apredict_stream`.
- Gemini is independent (does not inherit from OpenAI), so it needs its own async implementations.

---

## Phase 5: `APICaptioner` async support

**File:** `yadc/captioners/api/api_captioner.py`

`APICaptioner.__init__` performs auto-detection via `_infer_api_type()`, which makes sync HTTP probes. This is acceptable to keep sync because:
- It runs once during captioner construction, not per-image.
- The WebUI creates a fresh `APICaptioner` per `CaptionJob`.
- We can add `_ainfer_api_type()` later if needed; **deferred**.

In `__init__`:
- If `_async_session` kwarg is provided, pass it to the inner captioner.
- Otherwise, inner captioners use the default `apredict` (sync delegation).

After Phases 3–5, the matrix looks like:

| Backend | `predict()` | `apredict()` |
|---------|-------------|--------------|
| OpenAI | Sync (requests) | True async (httpx) |
| OpenRouter | Sync (inherited) | True async (inherited) |
| Gemini | Sync (requests) | True async (httpx) |
| Ollama | Sync (inherited) | True async (inherited) |
| vLLM | Sync (inherited) | True async (inherited) |
| llama.cpp | Sync (inherited) | True async (inherited) |
| KoboldCpp | Sync (own) | True async (inherited from OpenAI, unless `_load_model` blocks) |

Note: KoboldCpp `_load_model` has a polling loop with `time.sleep()`. It runs during `load_model()` in `__init__`, not during `apredict()`, so it doesn't affect mid-flight cancellation.

---

## Phase 6: `AsyncCaptionJob`

**File:** `yadc/api/services/captioning.py`

Replace the threaded `CaptionJob` with an async version for the Web API.

### New class: `AsyncCaptionJob`

```python
class AsyncCaptionJob:
    def __init__(self, dataset_name, dataset_service, event_dispatcher, logger, options, on_done, job_id):
        ...
        self._stop_event = asyncio.Event()
        self._task: asyncio.Task | None = None

    async def start(self):
        self._task = asyncio.create_task(self._arun())

    def request_stop(self):
        self._stop_event.set()
        if self._task and not self._task.done():
            self._task.cancel()

    async def _arun(self):
        # Same logic as CaptionJob._do_run but:
        # 1. Uses asyncio.Event for stop signal
        # 2. Awaits model.apredict() instead of model.predict()
        # 3. Catches asyncio.CancelledError to handle mid-flight abort gracefully
        ...
```

### Cancellation behavior
- `request_stop()` sets the event **and** cancels the `asyncio.Task`.
- If the task is inside `await self._async_session.post(...)`, `task.cancel()` raises `asyncio.CancelledError` inside `httpx`, which closes the connection immediately.
- The main loop catches `asyncio.CancelledError`, logs "Captioning stopped", emits final status, and returns.
- If the task is between images (e.g., encoding, saving), the `asyncio.Event` is checked and the loop exits cleanly.

### Keep `CaptionJob`?
Yes — keep the sync `CaptionJob` class unchanged. It is not used by the Web API, but removing it is unnecessary risk. We can mark it with a `# Legacy: used only if CLI batch mode is added` comment.

---

## Phase 7: Async `CaptioningService`

**File:** `yadc/api/services/captioning.py`

Since `CaptioningService` is Web-API-only, convert it to manage async tasks.

```python
class CaptioningService(Service):
    def __init__(self, ...):
        ...
        self._lock = asyncio.Lock()
        self._jobs: dict[str, AsyncCaptionJob] = {}

    async def start_job_async(self, dataset_name: str, options: CaptionJobOptions) -> JobInfo:
        ...

    async def stop_job_async(self, dataset_name: str) -> bool:
        ...

    async def get_status_async(self, dataset_name: str) -> JobInfo:
        ...

    async def caption_single_async(self, dataset_name: str, image_id: int, options: CaptionJobOptions) -> JobInfo:
        ...
```

- `_cleanup()` becomes an async delayed task (`asyncio.sleep(5)` instead of `time.sleep(5)` + thread).
- `DatasetService` calls inside `_arun()` (e.g., `get_image`, `refresh_image_index`) are sync but fast SQLite/file ops. Call them directly or wrap in `asyncio.to_thread()` if needed.

---

## Phase 8: Web API controller update

**File:** `yadc/api/controllers/api_captioning.py`

Switch the controller to await the new async service methods:

```python
@app.post("/datasets/<name>/caption")
async def start_captioning(name: str):
    ...
    info = await captioning.start_job_async(name, options)
    ...

@app.delete("/datasets/<name>/caption")
async def stop_captioning(name: str):
    stopped = await captioning.stop_job_async(name)
    ...

@app.post("/datasets/<name>/images/<int:image_id>/caption")
async def caption_single_image(name: str, image_id: int):
    ...
    info = await captioning.caption_single_async(name, image_id, options)
    ...
```

`get_captioning_status` can remain sync (it just reads a snapshot), or be made async for consistency.

---

## Phase 9: Testing

### Unit tests
- Mock `AsyncSession` and verify `AsyncCaptionJob` handles `asyncio.CancelledError` correctly.
- Test that `request_stop()` during `apredict()` aborts immediately without waiting for a mock delay.
- Verify sync `CaptionJob` still works (existing tests).

### Integration tests
- Start a captioning job against a slow mock server (e.g., `respx` or `pytest-httpx` with delay).
- Call `stop_job_async()` mid-request and assert the job ends within ~1s rather than waiting for the full delay.

### Files to update/create
- `tests/api/test_captioning.py` — update mocks to async service methods.
- `tests/captioners/test_async_session.py` — new.
- `tests/captioners/test_openai_async.py` — new, or extend existing.

---

## Phase 10: Frontend (no changes required)

The frontend already calls `DELETE /api/datasets/{name}/caption`. The endpoint behavior stays the same — it just becomes actually effective mid-flight. No JS/TS changes needed.

---

## Dependency changes

Add to `pyproject.toml`:
```toml
dependencies = [
    ...
    "httpx>=0.27",
]
```

Keep `requests` — CLI still uses it.

---

## Rollback / risk mitigation

- **If `httpx` causes issues in a specific backend**, that backend can fall back to the default `apredict` (sync delegation) while keeping the rest async.
- **If `AsyncCaptionJob` has bugs**, the old `CaptionJob` is still in the file and can be wired back in the controller temporarily.
- **Feature flag idea:** We could add an env var `YADC_ASYNC_CAPTIONING=0` to force the sync path. Not required for initial implementation, but easy to add.

---

## Phase 11: CLI async migration ✅ Complete

**File:** `yadc/cli_caption.py`

The CLI was the last sync user of `predict()` / `predict_stream()`. Since click doesn't natively support async commands, the entry point wraps an async helper in `asyncio.run()`:

```python
@click.command(...)
def caption(dataset: TextIO, **kwargs: Any):
    return asyncio.run(_caption_async(dataset, **kwargs))
```

Inside `_caption_async`:
- Creates an `AsyncSession` and passes it to `APICaptioner`
- Calls `model.load_model()` (still sync — auto-detection)
- Uses `await model.apredict()` / `async for token in model.apredict_stream()` for all predictions
- Closes the async session with `await async_session.aclose()` when done

All helper functions (`_caption`, `_predict_caption_one_shot`, `_predict_caption_rounds`) became `async def`.

Sync file I/O (`click.prompt`, `click.edit`, `open(...).write(...)`) was left as-is because the CLI runs a single task — blocking the event loop on user input is harmless.

---

## Phase 12: Test migration ✅ Complete

**Files:** `tests/captioners/api/conftest.py`, `tests/captioners/api/test_*.py`

All captioner tests were migrated from sync `predict()` / `predict_stream()` to async `apredict()` / `apredict_stream()`.

### Approach

Instead of adding a new dependency like `respx`, a lightweight `MockAsyncSession` helper was added to the conftest:

```python
class MockAsyncSession(AsyncSession):
    def __init__(self, response_text: str):
        super().__init__("http://test")
        self._response_text = response_text

    @asynccontextmanager
    async def arequest(self, method, path, *, capture_ctx=None, **kwargs):
        request = httpx.Request(method, f"http://test/{path.lstrip('/')}")
        yield httpx.Response(200, text=self._response_text, request=request)
```

This yields pre-baked `httpx.Response` objects with a dummy request attached (required for `response.raise_for_status()`). The existing `requests_mock` setup was kept for sync construction (`load_model`, auto-detection).

Tests changed from:
```python
def test_openai_o4_mini(openai, load_test_data):
    captioner = openai("nonstreaming/openai_o4_mini.txt", "o4-mini")
    captioner.load_model("o4-mini")
    got = captioner.predict(mock_image)
    assert got == expected
```

To:
```python
@pytest.mark.asyncio
async def test_openai_o4_mini(openai, load_test_data):
    captioner = openai("nonstreaming/openai_o4_mini.txt", "o4-mini")
    captioner.load_model("o4-mini")
    got = await captioner.apredict(mock_image)
    assert got == expected
```

---

## Phase 13: `HTTPResponseCache` httpx support ✅ Complete

**File:** `yadc/captioners/api/utils/cache.py`

The cache now stores responses in a backend-neutral format (`_ResponseCacheEntry`) and reconstructs either `requests.Response` or `httpx.Response` on demand.

### Changes

- `set()` accepts `requests.Response | httpx.Response`
- `get()` returns `requests.Response | None` (unchanged signature, for sync callers)
- `get_httpx()` added — returns `httpx.Response | None`
- `_read_entry()` extracts shared deserialization logic
- File I/O remains synchronous (deferred to Phase 15)

`AsyncSession.aget()` now uses `get_httpx()` and passes `httpx.Response` directly to `set()` — no more building fake `requests.Response` objects from `httpx` data.

---

## Phase 14: Async API type inference + factory ✅ Complete

**File:** `yadc/captioners/api/api_captioner.py`

The last sync I/O in the captioning stack was `APICaptioner._infer_api_type()`, which probes the remote API during construction to determine which backend to instantiate. This was blocking the event loop if called from an async context.

### Changes

1. **Extracted `_infer_api_type_async()`** — a standalone async function that uses `AsyncSession` to perform the same probes non-blocking.
2. **Added `APICaptioner.create()` async factory** — does async inference, then constructs `APICaptioner` with a pre-inferred `api_type`:
   ```python
   model = await APICaptioner.create(
       api_url=url,
       async_session=session,
       ...
   )
   await model.aload_model("model-name")
   ```
3. **Modified `__init__`** — accepts optional `api_type: APITypes | None = None`. When `None` (legacy), falls back to sync inference. When provided (factory path), skips inference entirely.
4. **Added `aload_model()`** to `APICaptioner`, `OpenAICaptioner`, `GeminiCaptioner`, `KoboldcppCaptioner`, and `Captioner` ABC (default delegates via `run_in_executor`).

### Call sites updated

| File | Before | After |
|------|--------|-------|
| `yadc/api/services/captioning.py` | `APICaptioner(...)` + `model.load_model()` | `await APICaptioner.create(...)` + `await model.aload_model()` |
| `yadc/cli_caption.py` | same | same |

---

## Phase 15: Remove sync captioner methods ✅ Complete

Deleted all remaining sync methods and the sync `Session` class. The entire captioning stack is now async-only.

### Deleted

| File | What was removed |
|------|-----------------|
| `yadc/captioners/api/session.py` | **Entire file** (sync `Session`) |
| `yadc/captioners/api/openai.py` | `load_model`, `_load_model`, `_generate_prediction`, `_generate_stream_prediction`, `_generate_stream_prediction_inner`, `predict`, `predict_stream` |
| `yadc/captioners/api/gemini.py` | Same pattern as OpenAI |
| `yadc/captioners/api/koboldcpp.py` | `_load_model`, `unload_model` |
| `yadc/captioners/api/openrouter.py` | `_log_api_information()` from `__init__` → moved to async `_alog_api_information()` called from `aload_model` |
| `yadc/core/captioner.py` | `predict`, `predict_stream` ABC methods; `load_model` is no longer `@abstractmethod` |
| `yadc/captioners/api/base.py` | `self._session` (sync), `requests` import, `Session` import |
| `yadc/captioners/api/api_captioner.py` | `_infer_api_type`, `load_model`, `predict`, `predict_stream`; `api_type` is now required in `__init__` |
| `yadc/api/services/captioning.py` | `CaptionJob` (legacy threaded class), `_cleanup` sync helper, sync `start_job`/`stop_job`/`get_status`/`caption_single` on `CaptioningService` |

### Tests updated

- `tests/captioners/api/conftest.py` — `MockAsyncSession` now handles `aload_model` paths (`models`, `api/v1/model`, `api/admin/list_options`, etc.) so no `requests_mock` is needed.
- All `test_*.py` fixtures now pass `api_type=...` explicitly and use `await captioner.aload_model(...)`.
- Bad-model tests are now async. 

---

## Summary of implemented file changes

| File | Change |
|------|--------|
| `pyproject.toml` | Add `httpx` dependency |
| `yadc/captioners/api/async_session.py` | **New** — `AsyncSession` with `httpx.AsyncClient` |
| `yadc/core/captioner.py` | Add `apredict` / `apredict_stream` defaults |
| `yadc/captioners/api/base.py` | Add `_async_session` support |
| `yadc/captioners/api/openai.py` | Add `_agenerate_prediction`, `_agenerate_stream_prediction`, `apredict`, `apredict_stream`; keep sync methods |
| `yadc/captioners/api/gemini.py` | Add async equivalents; keep sync methods |
| `yadc/captioners/api/api_captioner.py` | Pass `_async_session` to inner captioners; add `apredict`/`apredict_stream` delegation |
| `yadc/captioners/api/utils/thinking.py` | Add `_handle_thinking_streaming_async` |
| `yadc/captioners/api/utils/error_normalization.py` | Handle `httpx.HTTPStatusError` |
| `yadc/api/services/captioning.py` | Add `AsyncCaptionJob`; async service methods; keep legacy `CaptionJob` |
| `yadc/api/controllers/api_captioning.py` | Await async service methods |
| `yadc/cli_caption.py` | CLI uses async captioning via `asyncio.run()` |
| `yadc/captioners/api/utils/cache.py` | `get_httpx()` + `set()` accepts `httpx.Response` |
| `yadc/captioners/api/api_captioner.py` | `_infer_api_type_async()`, `create()` factory, `aload_model()` |
| `tests/api/test_captioning.py` | Update mocks for async service |

---

## Estimation (actual)

| Phase | Effort |
|-------|--------|
| 1–9 — Core async migration | ~1 day |
| 11 — CLI async migration | ~30m |
| 12 — Test migration | ~1h |
| 13 — Cache httpx support | ~15m |
| 14 — Async inference + factory | ~30m |
| 15 — Delete sync methods | ~1h |

---

## Phase 16: Async cache I/O (deferred)

The cache file I/O (`open()`, `f.read()`, `f.write()`) is still synchronous. While the files are small and access is rare, a future refactor could use `aiofiles` to make `get()` / `set()` fully async. This is low priority since the cache is already fast enough in practice.
