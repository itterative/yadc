---
name: prompt-generator-plan
description: Meta-prompting feature — generate high-quality Jinja2 prompt templates from an intent (and optional few-shot examples) using any configured env's model, with streaming + cancel. Extracts a generic LLM client layer (yadc/llm/) from the existing captioners so captioners delegate and the new prompt generator uses the client directly.
---

# Prompt Generator Plan

## Goal

Add a new yadc feature that uses an existing env's model to **generate
high-quality Jinja2 prompt templates** from a user-stated intent and
optional few-shot examples. The output is a Jinja2 template (with
`system_prompt` + `user_prompt` blocks) that can be saved via the
existing `/api/templates` endpoint and immediately used for captioning.

The prompt generator also pulls a **LLM client layer** (`yadc/llm/`)
out of the existing captioners, so the chat-completion mechanics
(message construction, HTTP transport, reasoning handling, streaming)
live in one place and image encoding stays where it belongs — in the
captioners.

## Decisions (from design discussion)

- **Output**: Jinja2 template, savable via the existing
  `EditTemplateDialog` flow.
- **LLM client layer**: New top-level package `yadc/llm/` with:
  - `BaseLLMClient` — abstract base; owns `AsyncSession`,
    `HTTPResponseCache`, `ResponseLogger`; exposes
    `async def predict_next_message_stream(...) -> MessageStream`
    (awaited — performs the POST + opens the streaming response
    before returning), `list_models`, `load_model`, `unload_model`,
    `offload_model`. Stream-only — no separate non-streaming method.
  - `OpenAICompatibleLLMClient` — base class for any
    chat/completions-style server (openai.com, openrouter, llama.cpp,
    koboldcpp, vllm, ollama). Owns the message translation +
    API-type probing that previously lived in `APICaptioner._infer_api_type()`
    (as a detection/logging signal, not as a dispatch decision).
  - Per-server thin subclasses of `OpenAICompatibleLLMClient`:
    `OpenAILLMClient`, `OpenRouterLLMClient`, `VllmLLMClient`,
    `LlamacppLLMClient`, `KoboldcppLLMClient`, `OllamaLLMClient`.
    Each is a thin wrapper that mostly just exists to give the right
    type identity to `create_client()`; the main reason for keeping
    them separate is that `load_model` may have server-specific
    behavior (local APIs actually load GGUF/etc. into VRAM; cloud
    APIs use it to validate model existence).
  - `GeminiLLMClient` — handles Gemini's `generateContent` API.
  - `create_client(env: UserConfig, **overrides)` factory in
    `yadc/llm/factory.py` — takes the `UserConfig` returned by
    `cmd_envs.load_env`, infers backend type from the API URL,
    returns the right client. Kept because URL-based type inference
    is needed (OpenAI-compatible vs Gemini); may be removable later
    if the type can be determined from env config directly.
- **Single captioner**: `APICaptioner` handles **all** backends
  (OpenAI-compatible + Gemini). Holds any `BaseLLMClient` subclass —
  the client interface is unified enough that one captioner class
  works for both. No separate `GeminiCaptioner` needed. The
  captioner-side factory (`APICaptioner.create()`) goes away —
  callers do `client = create_client(...); captioner =
  APICaptioner(client=client, ...)`.
- **`Message` type**: OpenAI-style Pydantic model
  (`role` + `content`), with content being `str` for plain text or
  `list[TextPart | ImageUrlPart]` for multimodal. Both backends
  translate to/from their native shape.
- **Captioners** hold a client instance (composition). Their
  `predict(image)` becomes a thin wrapper:
  build prompts + encode image + delegate to
  `client.predict_next_message_stream(messages)` (then `.collect()`
  to get the final `Message`). Stream-only — no separate non-streaming
  `predict_next_message` method, because some backends (notably
  llama.cpp) don't cleanly stop decoding in non-streaming mode, so
  streaming is the safe default that gives a natural signal for "the
  model is done".
- **Stream chunks**: Structured `StreamChunk` with `text`,
  `reasoning`, `reasoning_encrypted` (and `reasoning_summary`)
  fields — preserves reasoning for multi-turn conversations and for
  forwarding back to OpenAI-compatible APIs that support
  `reasoning_details`.
- **Examples**: Both sources, user picks — dataset sampling and
  manual `(subject → caption)` pairs.
- **UX**: newline-delimited JSON streaming response (fetch +
  `ReadableStream` + `AbortController` on the frontend) with
  mid-stream cancel via `controller.abort()`.

## Architecture Sketch

### New module: `yadc/llm/`

```
yadc/llm/
  __init__.py                # Re-exports
  types.py                   # Message, TextPart, ImageUrlPart,
                             # ImageUrl, StreamChunk, MessageStream
  base.py                    # BaseLLMClient — HTTP session, cache,
                             # logger; abstract predict_next_message_stream,
                             # list_models, load_model, unload_model,
                             # offload_model
  openai_compatible.py       # OpenAICompatibleLLMClient — chat/completions
                             # translation + API-type probing (shared by
                             # all OpenAI-compatible subclasses)
  openai.py                  # OpenAILLMClient — thin
  openrouter.py              # OpenRouterLLMClient — thin
  vllm.py                    # VllmLLMClient — thin (may override load_model)
  llamacpp.py                # LlamacppLLMClient — thin (may override load_model)
  koboldcpp.py               # KoboldcppLLMClient — thin
  ollama.py                  # OllamaLLMClient — thin
  gemini.py                  # GeminiLLMClient — generateContent translation
  factory.py                 # create_client(env) — URL → per-server client
  utils/                     # (optional) shared helpers if needed
```

### `Message` / `StreamChunk` shapes

```python
# yadc/llm/types.py

class TextPart(pydantic.BaseModel):
    type: Literal["text"] = "text"
    text: str

class ImageUrl(pydantic.BaseModel):
    url: str                          # data: URL or https URL
    detail: Literal["auto", "low", "high"] | None = None

class ImageUrlPart(pydantic.BaseModel):
    type: Literal["image_url"] = "image_url"
    image_url: ImageUrl

ContentPart = TextPart | ImageUrlPart

class Message(pydantic.BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str | list[ContentPart]   # str for plain, list for multimodal
    # For assistant messages coming back from the model:
    reasoning: str | None = None
    reasoning_summary: str | None = None
    reasoning_encrypted: list[dict[str, Any]] | None = None  # opaque

class StreamChunk(pydantic.BaseModel):
    """A single chunk from predict_next_message_stream.

    At least one of `text` / `reasoning` will be set per chunk.
    `reasoning_encrypted` is set once at end-of-stream (if applicable).
    """
    text: str | None = None
    reasoning: str | None = None
    reasoning_summary: str | None = None
    reasoning_encrypted: list[dict[str, Any]] | None = None
```

### `MessageStream` shape

The return type of every prediction call. **`predict_next_message_stream`
is an `async def`** — the caller awaits it. The await performs the
backend POST and obtains the open streaming response (preflight /
connection errors surface here, including the friendly
"Connection closed unexpectedly" wrapper for
`httpx.RemoteProtocolError`/`ReadError`), then returns a
`MessageStream` wrapping that already-open response. Because the
response is already open at return time, cancellation is a concrete
operation — **no background task or queue is needed**:

- `async for chunk in stream` decodes `StreamChunk`s from the open
  response and accumulates them into `.message`.
- `await stream.collect()` drains remaining chunks and returns the
  accumulated `Message` (no-op if already drained).
- `.message` is the partial-so-far accumulated `Message` (read
  mid-iteration for live preview).
- `await stream.cancel()` closes the underlying streaming response
  (httpx `aclose()`). Idempotent. Use this when the consumer needs to
  stop without throwing into its own iterating task (the controller's
  client-disconnect path).

Cancellation also propagates the normal asyncio way: cancelling the
task that is `async for`-ing the stream raises `CancelledError` inside
`__anext__`, which httpx propagates to the request. `cancel()` is the
explicit, non-throwing variant.

```python
class MessageStream:
    """Async iterable of StreamChunks that self-accumulates into a Message.

    Wraps an already-open streaming response (see predict_next_message_stream).

    Use cases:
        # Non-streaming result
        stream = await client.predict_next_message_stream(messages)
        message = await stream.collect()

        # Stream chunks, then get the final message
        stream = await client.predict_next_message_stream(messages)
        async for chunk in stream:
            render(chunk)
        message = await stream.collect()  # already drained, returns immediately

        # Cancel mid-stream (e.g. user clicked Cancel)
        stream = await client.predict_next_message_stream(messages)
        async for chunk in stream:
            if cancelled:
                await stream.cancel()   # close the upstream response
                break
    """

    def __aiter__(self) -> AsyncIterator[StreamChunk]: ...
    async def __anext__(self) -> StreamChunk: ...

    async def collect(self) -> Message:
        """Drain remaining chunks (if any) and return the accumulated Message."""

    @property
    def message(self) -> Message:
        """Currently accumulated Message. Partial if iteration is in progress."""

    async def cancel(self) -> None:
        """Close the underlying streaming response (httpx aclose). Idempotent."""
```

Stream-only API: no separate non-streaming method. Some backends
(notably llama.cpp) don't cleanly stop decoding in non-streaming
mode, so streaming is the safe default that gives a natural signal
for "the model is done". Callers that don't care about chunks just
`await stream.collect()`.

### `BaseLLMClient` surface

```python
class BaseLLMClient(abc.ABC):
    def __init__(self, *, api_url: str, api_token: str,
                 cache: HTTPResponseCache | None = None,
                 response_logger: ResponseLogger | None = None,
                 **_kwargs): ...

    @abc.abstractmethod
    async def predict_next_message_stream(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 4096,
        model: str | None = None,            # None = client default
        reasoning: Literal["low", "medium", "high"] | None = None,
        **opts,
    ) -> MessageStream:
        """POST the request, open the streaming response, return a MessageStream.

        Awaiting this performs the backend POST and obtains the open
        streaming response (preflight/connection errors surface here,
        including the friendly 'Connection closed unexpectedly'
        wrapper for httpx.RemoteProtocolError/ReadError). The returned
        MessageStream wraps that already-open response, so cancellation
        is a concrete operation (no background task/queue needed).
        """

    @abc.abstractmethod
    async def list_models(
        self, *, cache_ttl: float | None = DEFAULT_MODELS_CACHE_TTL_SECONDS,
    ) -> list[str]: ...

    async def load_model(self, model_repo: str, **kwargs) -> None:
        """Validate the model exists (cloud) or load it into VRAM (local).

        Default impl lists models and checks `model_repo` is in the list.
        Per-server clients override for server-specific loading (e.g.
        llama.cpp pulls GGUF into VRAM).
        """

    def unload_model(self) -> None:
        """Free resources associated with the loaded model."""

    def offload_model(self) -> None:
        """Offload model to CPU to free GPU memory."""

    async def aclose(self) -> None: ...
```

### `OpenAICompatibleLLMClient`

- Absorbs the substantive machinery currently in `OpenAICaptioner`
  (the real OpenAI-compatible base — 657 lines — that
  `Llamacpp/Ollama/Vllm/Koboldcpp/OpenRouterCaptioner` subclass):
  its `conversation()` builder and the `predict`/`predict_stream`
  HTTP mechanics, minus image encoding and Jinja prompt rendering
  (those stay with the captioner).
- Talks to any `/v1/chat/completions`-style endpoint.
- Translates `list[Message]` → OpenAI chat/completions body verbatim
  (since the message shape already matches).
- Parses the response back into `Message` (with `reasoning_details`
  captured into `reasoning` / `reasoning_encrypted`).
- The API-type probing (URL heuristics + `/models` + `/health` +
  `/api/version` etc.) moves here from the module-level
  `_infer_api_type_async()` in `api_captioner.py` (a function, not
  a method) but is used only for **logging / detection metadata**,
  not for dispatch — there's only one implementation regardless of
  which OpenAI-compatible server it turns out to be.

### `GeminiLLMClient`

- Translates `list[Message]` → Gemini's `contents` + `system_instruction` shape.
- Translates response → `Message` (with `thought: true` parts captured
  as `reasoning`, encrypted reasoning if any goes to
  `reasoning_encrypted`).

### Captioner refactor (composition, public API unchanged)

**Single captioner class** — `APICaptioner` handles **all** backends
(OpenAI-compatible + Gemini). Holds any `BaseLLMClient` subclass.
The per-server captioner classes (`OpenAICaptioner`,
`OpenRouterCaptioner`, `VllmCaptioner`, `LlamacppCaptioner`,
`KoboldcppCaptioner`, `OllamaCaptioner`, `GeminiCaptioner`) all
collapse — the client interface is unified enough that one
captioner works for both OpenAI-compatible and Gemini.

**Public API stays identical** — same `predict(image) -> str`,
`predict_stream(image) -> AsyncGenerator[str, None]`,
`load_model(model_repo, **kwargs)`, `unload_model()`,
`offload_model()` signatures, same `extra_messages` and
`prediction_context` kwargs. `AsyncCaptionJob`, refine, and other
callers don't move.

```python
class APICaptioner(Captioner):
    """Captioner that delegates to any BaseLLMClient (OpenAI-compatible or Gemini)."""

    def __init__(self, *, client: BaseLLMClient, **kwargs):
        self._client = client

    async def predict(self, image: DatasetImage, *, prediction_context=None,
                      extra_messages=None, **kwargs) -> str:
        messages = self._build_messages(image, extra_messages=extra_messages)
        stream = await self._client.predict_next_message_stream(
            messages, prediction_context=prediction_context, **kwargs)
        message = await stream.collect()
        if prediction_context is not None:
            self._populate_prediction_context(prediction_context, message)
        return self._strip_thinking(message.content)

    async def predict_stream(self, image: DatasetImage, *, prediction_context=None,
                             extra_messages=None, **kwargs) -> AsyncGenerator[str, None]:
        messages = self._build_messages(image, extra_messages=extra_messages)
        stream = await self._client.predict_next_message_stream(
            messages, prediction_context=prediction_context, **kwargs)
        async for chunk in stream:
            if chunk.text and not self._in_thinking(chunk):
                yield self._strip_thinking(chunk.text)
        # After stream completes, populate reasoning into prediction_context
        if prediction_context is not None:
            self._populate_prediction_context(prediction_context, stream.message)

    # load_model / unload_model / offload_model proxy straight to the
    # client. This keeps the captioner's public method surface identical
    # so AsyncCaptionJobRunner, the CLI runner, and existing captioner
    # tests are untouched by the refactor — only the client carries the
    # real load/unload/offload logic.
    async def load_model(self, model_repo: str, **kwargs) -> None:
        await self._client.load_model(model_repo, **kwargs)

    def unload_model(self) -> None:
        self._client.unload_model()

    def offload_model(self) -> None:
        self._client.offload_model()

    def _build_messages(self, image: DatasetImage, *, extra_messages=None) -> list[Message]:
        """Build the OpenAI-style message list from the image + extra_messages.

        Encodes the image (via _encode_image), builds the system + user
        prompts (via prompts_from_image), and packages everything into
        Message objects with text + image_url content parts.
        """
```

The captioner-side factory (`APICaptioner.create()`) goes away.
Callers do the two-step explicitly:

```python
user_config = cmd_envs.load_env(env, password=password)
client = create_client(user_config, **overrides)
captioner = APICaptioner(client=client, **kwargs)
```

The factory for the **client** (`create_client`) is kept because
URL-based type inference is needed (which OpenAI-compatible server
are we talking to? OpenAI vs Gemini?). May be removable later if
the type can be determined elsewhere (e.g. from env config).



### `PromptGenerationService`

- New service in `yadc/api/services/prompt_generation.py`.
- Stateless (one-shot generation, no job lifecycle / DB persistence).
- Uses `OpenAICompatibleLLMClient` / `GeminiLLMClient` **directly**
  (not through a captioner) — there's no image involved.

```python
class ExamplePair(pydantic.BaseModel):
    subject: str
    caption: str
    source: Literal["dataset", "manual"] = "manual"

class PromptGenerationService:
    async def generate(
        self,
        *,
        env: str,
        intent: str,
        examples: list[ExamplePair] | None = None,
        focus: Literal["system", "user", "both"] = "both",
        api_model_name: str | None = None,
        password: str | None = None,
    ) -> AsyncIterator[StreamChunk]: ...
```

Internally:
- `cmd_envs.load_env(env, password=password)` → decrypts token
  (surfaces `PasswordRequiredError` for the 403 pre-flight).
- `create_client(user_config)` → instantiates the right client.
- `client.load_model(user_config.api.model_name)` → ensures the
  model is loaded before predicting. Required for local backends
  (llama.cpp/koboldcpp/ollama actually load into VRAM); an existence
  check for cloud backends. Mirrors what the captioning runner does.
- Builds the meta-conversation:
  - System: "You are a prompt engineer. Emit a Jinja2 template with
    `{% set system_prompt %}…{% endset %}` and
    `{% set user_prompt %}…{% endset %}` blocks…"
  - User: intent + formatted few-shot examples.
- `stream = await client.predict_next_message_stream(...,
    reasoning=None)` — reasoning is explicitly **off** for
    generation (the artifact is the template body; thinking wastes
    tokens). The `MessageStream` is consumed by the controller's
    chunked HTTP response — one newline-delimited JSON line per
    `StreamChunk`, plus a final `{"type": "done"}` or
    `{"type": "error"}` line. Cancellation propagates: if the
    client disconnects, the underlying `asyncio.Task` is cancelled,
    which calls `stream.cancel()` on the `MessageStream`, closing the
    upstream response.

### Routes — `yadc/api/controllers/api_prompts.py`

- `POST /api/prompts/generate` (newline-delimited JSON streaming
  response — fetch + `ReadableStream` on the frontend):
  - Body: `{ env, api_model_name?, intent, examples: ExamplePair[],
    focus }`
  - Returns a chunked HTTP response. Each chunk is one JSON object
    per line: `{"type": "token", "text": "...", "reasoning": "..."}`
    or `{"type": "done", "message": {...}}` or
    `{"type": "error", "message": "..."}`.
  - The controller wraps the `MessageStream` from
    `PromptGenerationService.generate()` and emits one line per
    `StreamChunk` (or aggregated batches, TBD). Cancellation
    propagates: if the client disconnects, the underlying
    `asyncio.Task` is cancelled, which calls `stream.cancel()` on
    the `MessageStream`, which cancels the upstream HTTP request.
  - Pre-flight: `cmd_envs.load_env(env, password=...)` to surface
    `PasswordRequiredError` as 403 (mirrors the captioning endpoint).
- No cancel endpoint needed — cancel is implicit in client
  disconnect.

### Frontend

**Stores** — `yadc/webui/src/lib/stores/prompts/`

- `types.ts` — `ExamplePair`, `PromptGenRequest`, `PromptGenFocus`,
  re-export `StreamChunk`.
- `api.ts` — POST generate (streaming, via `fetch` + `ReadableStream`).
  `controller.abort()` is the cancel mechanism — no separate
  cancel endpoint.
- `store.svelte.ts` — current generation state (idle / streaming /
  complete / cancelled / error), accumulated tokens. Uses runes
  (`.svelte.ts`) rather than plain `store.ts`: the streaming state
  is naturally reactive (`$state` status flag + accumulated token
  string updated as chunks arrive), and there are no cross-cutting
  writables to share — consistent with the top-level
  `topbar.svelte.ts` rune precedent.
- `actions.ts` — `startGeneration()`, `cancelGeneration()`.
- `index.ts` — barrel.

**Components** — `yadc/webui/src/lib/components/prompts/`

- `PromptGenerator.svelte` — host.
- `PromptForm.svelte` — env picker, model picker (existing
  `/api/envs/<name>/models`), intent textarea, focus toggle.
- `DatasetExamplesPanel.svelte` — dataset picker + N slider.
- `ManualExamplesPanel.svelte` — textarea for pairs.
- `GenerationPreview.svelte` — streaming preview with monospace
  font, reasoning collapsible, variable extraction (reuse
  `api_templates._JINJA_VAR_RE`), Save-as-template button.
- Feature folder pattern (≥3 components).

> **Name collision cleanup**: a `PromptPreview.svelte` already
> exists in `lib/components/ui/` — but its intent is unrelated (it
> renders the caption *prompt* that would be sent for a dataset
> image, not a generated template). It is used by exactly one
> consumer, `lib/components/dataset/detail/Preview.svelte` (a
> 14-line wrapper). Move its code into `Preview.svelte`, delete the
> UI-level component, and name the generator's preview distinctly
> (`GenerationPreview.svelte`) to avoid the collision.

**Route** — `yadc/webui/src/routes/prompts/+page.svelte`

- Mirrors `/templates`.
- Uses fetch + `ReadableStream` + `AbortController` (no SSE event
  bus) since generation is single-request and doesn't need history
  replay. The Cancel button calls `controller.abort()`.
- Save button → reuses `EditTemplateDialog`.

**Layout** — `yadc/webui/src/routes/+layout.svelte`

- Add "Prompts" nav item next to "Templates".

### CLI

**`yadc/cli_prompts.py`** + **`yadc/cmd/prompts/`**

- `yadc prompts generate --env <name> --intent "..." [--examples dataset:<name>:<n>] [--examples-manual "<subject>","<caption>" ...]`
  — streams the **generated Jinja2 template body** to **stdout**
  (so it pipes cleanly into `prompts save`). Progress / status /
  reasoning go to **stderr**, never stdout, so the pipe stays pure.
- `yadc prompts save <name>` — reads the template body from stdin
  and saves via `cmd_templates.save_user_template`.
- Mirror the split: click in `cli_prompts.py`, pure logic in
  `cmd/prompts/`.

## Phased Implementation

The original single Phase 1 is **split** so the feature can land on a
minimal client extraction without betting the whole refactor. The
captioner collapse is the risky part (it rewrites every backend's
hot path); it is decoupled so it can ship — and be bisected —
independently after the feature works.

### Phase 1a — Extract `yadc/llm/` client layer (additive, no user-visible change) — ✅ DONE

Goal: a standalone client package the prompt generator uses directly.
**Captioners are untouched** and keep working exactly as before.

**Implemented** (see `yadc/llm/` + `tests/llm/`). Key realization during
impl: the existing `AsyncSession.request(stream=True)` was **fake
streaming** — it popped `stream` and never passed it to httpx, so the
whole body was buffered before any line was yielded (verified with a
slow SSE server). The client layer needed TRUE streaming for
`MessageStream.cancel()` to be meaningful, so a new additive
`AsyncSession.open_stream()` (built on `client.send(req, stream=True)`)
was added; the legacy `request()`/`get()`/`post()` paths are untouched,
so captioners keep their existing (buffered) behavior.

1. Define `Message`, `StreamChunk`, content parts, `MessageStream` in
   `yadc/llm/types.py`.
2. Define `BaseLLMClient` in `yadc/llm/base.py` — owns `AsyncSession`,
   `HTTPResponseCache`, `ResponseLogger`; declares the abstract
   `async def predict_next_message_stream`, `list_models`,
   `load_model`, `unload_model`, `offload_model`.
3. Implement `OpenAICompatibleLLMClient` in
   `yadc/llm/openai_compatible.py` — **copy** the chat-completion
   body-building, response parsing, and streaming mechanics out of
   `OpenAICaptioner` (the real OpenAI-compatible base). Image
   encoding + Jinja prompt rendering stay behind in the captioner.
   API-type probing moves here as a logging/detection concern
   (ported from the module-level `_infer_api_type_async()` in
   `api_captioner.py`). Temporary duplication vs. the captioner is
   expected and intentional — it is removed in 1b.
4. Add thin per-server subclasses in `yadc/llm/openai.py`,
   `openrouter.py`, `vllm.py`, `llamacpp.py`, `koboldcpp.py`,
   `ollama.py`. Most are pure pass-throughs; `load_model` may be
   overridden where the server has a custom loading endpoint.
5. Implement `GeminiLLMClient` in `yadc/llm/gemini.py` — port from
   `gemini.py`, including reasoning translation.
6. Implement `create_client(env: UserConfig, **overrides)` factory in
   `yadc/llm/factory.py`.
7. Add unit tests for `predict_next_message_stream` on each client
   (yielding chunks, accumulating `Message`, `cancel()` closing the
   response, CancelledError propagation) with mocked async sessions,
   mirroring existing `tests/captioners/` patterns. Captioner tests
   are **not** touched in 1a — they keep constructing
   `APICaptioner(api_type=APITypes.X, ...)` and passing.

After 1a, Phases 2–4 (backend prompt generation, frontend, CLI) can
build on the client directly. **Phase 1b can land separately,
anytime.**

### Phase 1b — Collapse captioners onto the client (refactor, separately shippable) — ✅ DONE

1. Collapse all per-server captioner classes (`OpenAICaptioner`,
   `OpenRouterCaptioner`, `VllmCaptioner`, `LlamacppCaptioner`,
   `KoboldcppCaptioner`, `OllamaCaptioner`, `GeminiCaptioner`) into
   a single `APICaptioner` holding a `BaseLLMClient`. **Public API
   stays identical** (`predict`, `predict_stream`, `load_model`,
   `unload_model`, `offload_model`; `extra_messages` and
   `prediction_context` kwargs). `load_model`/`unload_model`/
   `offload_model` proxy to the client so `AsyncCaptionJobRunner`,
   the CLI runner, and their tests are untouched.
2. The `APICaptioner.create()` factory goes away — callers do
   `client = create_client(user_config)` + `APICaptioner(client=...)`.
3. Remove the duplicated mechanics from the captioners (the copies
   made in 1a).
4. Rewrite captioner tests: they currently construct
   `APICaptioner(api_type=APITypes.KOBOLDCPP, ...)`, which no longer
   exists. Replace with `client = create_client(...)` +
   `APICaptioner(client=client, ...)`.
5. Verify the Open Questions below (thinking-strip preservation,
   per-server `load_model` overrides).

### Phase 2 — Backend prompt generation

1. `PromptGenerationService` + `ExamplePair` + meta-prompt
   construction in `yadc/api/services/prompt_generation.py`.
2. `POST /api/prompts/generate` in
   `yadc/api/controllers/api_prompts.py` — chunked HTTP response
   (newline-delimited JSON), no SSE event bus needed.
3. Tests: service-level with mocked client; controller-level with
   `requests-mock` patterns, including client-disconnect cancel.

### Phase 3 — Frontend

1. `stores/prompts/` (types, api, store, actions).
2. `components/prompts/` feature folder.
3. `/prompts` route.
4. Sidebar nav item.
5. ESLint + prettier + svelte-check.

### Phase 4 — CLI parity

1. `cmd/prompts/` pure logic.
2. `cli_prompts.py` click commands.
3. Click in main group (`cli.py`).
4. Tests.

### Phase 5 (optional) — Polish

- Persist recent generations in `SettingsService` (KV store).
- Copy-to-clipboard for the generated template body.
- Show extracted Jinja variables inline as chips.
- A/B test variations of the meta-system-prompt.

## Key Files Touched

### New
- `yadc/llm/__init__.py`
- `yadc/llm/types.py` (`Message`, `TextPart`, `ImageUrlPart`,
  `ImageUrl`, `StreamChunk`, `MessageStream`)
- `yadc/llm/base.py` (`BaseLLMClient`)
- `yadc/llm/openai_compatible.py` (`OpenAICompatibleLLMClient`)
- `yadc/llm/openai.py`, `openrouter.py`, `vllm.py`, `llamacpp.py`,
  `koboldcpp.py`, `ollama.py` (thin per-server subclasses)
- `yadc/llm/gemini.py` (`GeminiLLMClient`)
- `yadc/llm/factory.py` (`create_client`)
- `yadc/api/services/prompt_generation.py`
- `yadc/api/controllers/api_prompts.py`
- `yadc/cli_prompts.py`
- `yadc/cmd/prompts/__init__.py`
- `yadc/cmd/prompts/prompts.py`
- `yadc/webui/src/routes/prompts/+page.svelte`
- `yadc/webui/src/lib/components/prompts/` (feature folder)
- `yadc/webui/src/lib/stores/prompts/` (sub-folder)

### Refactored / Removed
- `yadc/captioners/api/openai.py`, `gemini.py`, `openrouter.py`,
  `vllm.py`, `llamacpp.py`, `koboldcpp.py`, `ollama.py` →
  **deleted** in favor of a single `APICaptioner` class. All
  per-server API-specific logic moves to the corresponding LLM
  client (`yadc/llm/`).
- `yadc/captioners/api/api_captioner.py` → slimmed: single
  `APICaptioner` class with `client: BaseLLMClient` composition.
  The `APICaptioner.create()` factory goes away.
- `yadc/captioners/api/base.py` → HTTP infra (session, cache,
  logger) moves to `yadc/llm/base.py`; captioner base becomes thin.
- `yadc/captioners/api/__init__.py` → re-exports the collapsed
  `APICaptioner` from `yadc/llm/...` consumers.
- `yadc/api/events.py` → no new event types (fetch + ReadableStream
  is the chosen transport, no SSE event bus).

### Updated
- `yadc/cli.py` → register `prompts` subcommand.
- `yadc/webui/src/routes/+layout.svelte` → nav item.
- Documentation: `.pi/agent/memory/docs/backend/captioners.md`
  (clients live in `yadc/llm/` now), `.pi/agent/memory/docs/backend/api.md`
  (new controller), `.pi/agent/memory/docs/captioner-architecture.md`
  (hierarchy collapses; auto-detect facade gone; mixin pattern
  relocates to the client), plan memories.

## Open Questions

1. **Cancel propagation (mechanism resolved, verify end-to-end)**:
   `predict_next_message_stream` is awaited, so the returned
   `MessageStream` wraps an already-open response — `stream.cancel()`
   just closes it (httpx `aclose()`), and cancelling the iterating
   task raises `CancelledError` inside `__anext__`. No background
   task/queue. Remaining work: verify the frontend
   `AbortController` → `fetch.abort()` → server task cancel →
   `stream.cancel()`/`CancelledError` → upstream-close chain
   end-to-end (httpx should propagate cleanly).
2. **Per-server `load_model` overrides**: Some backends (llama.cpp,
   vllm, ollama) have server-specific model loading endpoints.
   Decide per-backend whether `load_model` overrides the default
   (model-existence via `/models`) or extends it. Phase 1 work.
3. **Reasoning in captioners**: Captioners currently strip thinking
   from streaming output via `_handle_thinking_streaming_async`.
   Confirm this behavior is preserved after the refactor — the
   `MessageStream` carries `chunk.reasoning` separately so we
   should be able to drop it cleanly. Phase 1 verification.
4. **Meta-prompt stability**: The system prompt that instructs the
   model to emit Jinja2 templates is itself a critical piece of UX.
   Worth tuning — defer to Phase 5.

## Resolved Decisions (from design discussion)

- **Transport**: fetch + `ReadableStream` + `AbortController`. No SSE
  event bus for the prompt generator — single-request, no history
  replay needed.
- **Per-server thin clients kept**: `OpenAILLMClient`,
  `VllmLLMClient`, `LlamacppLLMClient`, etc., all subclasses of
  `OpenAICompatibleLLMClient`. The reason is `load_model` — local
  APIs actually load models into VRAM, cloud APIs use it to
  validate model existence; the per-server split gives a clean
  place for that server-specific behavior.
- **Single `APICaptioner`** for all backends (OpenAI-compatible +
  Gemini). The per-server captioner classes collapse — the client
  interface is unified enough that one captioner class works for
  everything. No separate `GeminiCaptioner`. Holds any
  `BaseLLMClient` instance via composition.
- **Captioner public API unchanged**: `predict`,
  `predict_stream`, `load_model`, `unload_model`, `offload_model`
  signatures stay the same, as do the `extra_messages` and
  `prediction_context` kwargs. The refactor is purely internal —
  `AsyncCaptionJob`, refine, and other callers don't move.
- **Captioner factory removed**: `APICaptioner.create()` goes away.
  Callers do `client = await create_client(...)` then
  `captioner = APICaptioner(client=client, ...)`. The client factory
  is kept because URL-based type inference is still needed for
  choosing the right LLM client.
- **Stream-only API**: no separate non-streaming
  `predict_next_message` method. Callers who don't care about
  chunks use `await stream.collect()`. Stream-only is also the
  safe default because some backends (notably llama.cpp) don't
  cleanly stop decoding in non-streaming mode.

## Status

**Phase 1a + 1b + shared-infra relocation COMPLETE.** The captioner
layer is now a single `APICaptioner` that composes a `BaseLLMClient` from
`yadc/llm/`, and `yadc/llm/` is fully self-contained — the shared HTTP
infra (`async_session.py`, `cache.py`, `response_logger.py`,
`error_normalization.py`, `response_models.py`, `units.py`,
`DEFAULT_MODELS_CACHE_TTL_SECONDS`) lives there, and the
`captioners.api → llm` direction is unidirectional. Next up: Phase 2
(backend prompt generation) builds on `create_client` +
`predict_next_message_stream` directly.

Proposed — design refined after review:
- `predict_next_message_stream` is `async def` (caller awaits);
  `MessageStream` wraps an already-open response, so `cancel()` is a
  concrete `aclose()` — no background task/queue.
- Phase 1 split into **1a** (extract standalone `yadc/llm/` client,
  captioners untouched) and **1b** (collapse captioners onto the
  client, separately shippable). The feature builds on 1a; 1b is
  decoupled so the risky captioner rewrite can ship/bisect on its own.
- `APICaptioner.load_model`/`unload_model`/`offload_model` proxy to
  the client to keep the runner and existing tests untouched.
- `PromptGenerationService` calls `client.load_model(api_model_name)`
  before predicting and passes `reasoning=None`.
- Existing `PromptPreview.svelte` (single-use, unrelated intent) gets
  inlined into its consumer `Preview.svelte`; the generator's preview
  is `GenerationPreview.svelte`.
- CLI `generate` writes the template body to stdout (progress → stderr)
  so it pipes into `prompts save`.

## User feedback
* several docstrings reference the current plan; long-term, these docstrings are not relevant and should be focused on the current state of the files rather than on the referencing the refactoring done in this plan
