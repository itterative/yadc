---
name: prompt-generator-plan
description: Meta-prompting feature — generate (or refine) high-quality Jinja2 prompt templates from an intent (and optional few-shot examples) using any configured env's model, with streaming + cancel. Extracts a generic LLM client layer (yadc/llm/) from the existing captioners so captioners delegate and the new prompt generator uses the client directly.
last_history: 12
---

# Prompt Generator Plan

## Goal

Add a new yadc feature that uses an existing env's model to **generate
or refine high-quality Jinja2 prompt templates** from a user-stated
intent and optional few-shot examples. The output is a Jinja2
template (with `system_prompt` + `user_prompt` blocks) that can be
saved via the existing `/api/templates` endpoint and immediately
used for captioning.

The generator runs in one of two modes:

- **Generate** (Phase 2/3) — produce a new template from a stated
  intent and optional style-reference examples. The model invents
  the `system_prompt` + `user_prompt` content from scratch.
- **Refine** (Phase 6, planned) — the user picks an existing
  template, edits it in the form, and the LLM applies the user's
  refinement intent to it. Few-shot examples still work as a
  style reference. The mode is selected at the top of the form
  via a pill tab; the streaming preview + save flow are shared.

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
- **Refine mode (Phase 6)**: a second tab in the prompt generator
  ("Generate" / "Refine") that takes an existing template (user-
  picked + user-editable in the form), a refinement intent, and
  optional few-shot examples, and asks the LLM to apply the
  refinement. The backend switches modes via an optional
  `template_content` field on `PromptGenerationRequest` — no
  separate `mode` flag needed; the presence of the field is the
  signal. System prompt, message structure, and (lightly) the final
  "go" message differ between modes; everything else (controller,
  streaming, preview, save flow) is shared. See Phase 6 below.

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
- IndexedDB-backed persistence for the few-shot examples
  (image data URLs) — currently dropped on page reload because
  they don't fit in localStorage and we don't track per-example
  provenance for re-fetch. See
  `history/prompt-generator-plan/003-prompt-form-settings-persistence.md`.

### Phase 5b — Backend cleanup (done, prerequisite for Phase 6a)

**Goal**: small backend cleanup that lands **before** Phase 6a so
the refine mode work has a clean foundation. Three pieces, all
local to `yadc/api/services/prompt_generation.py`:

1. **Refactor file → package** (no behaviour change). Convert
   the single `prompt_generation.py` module into a `prompt_generation/`
   package. Same import path (`yadc.api.services.prompt_generation`)
   so blast radius is minimal — only the on-disk shape changes:

   ```
   yadc/api/services/prompt_generation/
     __init__.py         # Re-exports the public surface. Also
                          # re-exports ``cmd_envs`` + ``create_client``
                          # at the package level so the existing test
                          # patches keep resolving.
     service.py          # ``PromptGenerationService`` +
                          # ``_build_messages`` + ``_EXAMPLE_ACK``
                          # (everything currently in the .py file).
                          # Also loads the system prompt .txt files
                          # via :mod:`importlib.resources` and
                          # defines ``_GENERATE_SYSTEM_PROMPT`` (see
                          # point 2).
     prompts/            # Data directory (no __init__.py).
       generate.txt      # Current system prompt text.
       refine.txt        # Refine-mode system prompt text (staged for
                          # Phase 6a, file exists but is NOT loaded
                          # in Phase 5b — the loading line is added
                          # in Phase 6a when the constant is actually
                          # used).
   ```

   Test patches target the **package** path
   (`yadc.api.services.prompt_generation.cmd_envs`,
   `yadc.api.services.prompt_generation.create_client`), which the
   `__init__.py` re-exports explicitly so they keep working.

2. **System prompts as `.txt` files, loaded as package
   resources in `service.py`** (no behaviour change for now —
   just data → file). The actual prompt text moves from the
   module-level `_SYSTEM_PROMPT` Python string into
   `prompts/generate.txt`, loaded **once at import time inside
   `service.py`** (the file that uses the constant — no separate
   loader module, no `__init__.py` involvement, no import cycle):

   ```python
   # yadc/api/services/prompt_generation/service.py (top of file)
   from importlib.resources import files

   _GENERATE_SYSTEM_PROMPT: str = (
       files("yadc.api.services.prompt_generation")
       .joinpath("prompts")
       .joinpath("generate.txt")
       .read_text(encoding="utf-8")
       .strip()
   )
   ```

   `service.py` is the right home for the constant because it's
   the only consumer — putting the constant in `__init__.py`
   (or a separate `_prompts.py`) would force `service.py` to
   import the package back, which creates a circular import
   (`__init__.py` imports from `service.py` to re-export
   `PromptGenerationService`; `service.py` importing from the
   package re-enters `__init__.py` mid-execution).

   `prompts/refine.txt` is created as part of this cleanup (the
   structure is set up) but the constant + loading line are
   **deferred to Phase 6a** — that file is only meaningful once
   the refine-mode branching in `_build_messages` exists, and
   loading a file that nothing references would be dead code at
   import time. Reasons for `txt` over a Python constant:
   - No multi-line string quoting / indentation hazards.
   - Easy to diff in code review when we tune wording.
   - Easy to add new modes later (just add a new file + a new
     constant in `service.py`).
   - `importlib.resources` makes the prompts part of the
     package distribution — works the same in dev (uv editable
     install), wheel installs, and zip-imports. The
     `__file__`-relative approach would break in the latter
     two.

3. **Wire the system prompt into the message list** (bug fix).
   The current `_SYSTEM_PROMPT` is **dead code** — defined at the
   bottom of `prompt_generation.py` but never referenced anywhere.
   `_build_messages` returns a list of user/assistant turns only;
   the system prompt never reaches the LLM. Phase 5b fixes this
   by prepending `Message(role="system", content=system_prompt)`
   to the list returned by `_build_messages`. The function gains
   a `system_prompt: str` keyword-only parameter (so the
   constant in `service.py` can be injected without a global
   lookup) and `PromptGenerationService.generate` passes
   `_GENERATE_SYSTEM_PROMPT` (and, after Phase 6a,
   `_REFINE_SYSTEM_PROMPT` when `request.template_content` is
   set). The LLM client layer already maps `role="system"` to
   the native shape (OpenAI's `system` message, Gemini's
   `system_instruction`), so no client changes are needed.

**Test impact**: the existing 5 `TestBuildMessages` tests that
assert message indices (`messages[0]`, `messages[1]`, etc.)
**all shift by 1** because the system message now occupies
index 0. `test_final_user_turn_says_go` uses `messages[-1]` so
it's index-agnostic. One new test
(`test_system_message_is_first`) verifies the system message is
at index 0 and its content matches `_GENERATE_SYSTEM_PROMPT` as
loaded by `service.py` (which loads it from `prompts/generate.txt`
via `importlib.resources`) — keeps the test honest about the
txt-file indirection so a future refactor can't silently drop
the loading code.

**Out of scope for Phase 5b** (deferred to Phase 6a proper):

- The `template_content` field on `PromptGenerationRequest`.
- Branching in `_build_messages` on `request.template_content is not None`.
- Picking the refine system prompt vs the generate system prompt
  based on mode.
- New tests for the refine-mode message structure.

Phase 5b just sets up the structure; Phase 6a adds the
generate-vs-refine branching on top.

### Phase 6 — Refine mode (planned, not yet implemented)

**Goal**: a second mode of the prompt generator that takes an
existing template (user-picked, user-editable in the form) and
asks the LLM to apply a refinement intent to it. Few-shot
examples still work as a style reference. Reuses the existing
streaming preview and the "Save as template" save flow.

**Split**: Phase 6a (backend) + Phase 6b (frontend), matching the
Phase 2 + Phase 3 pattern.

#### Phase 6a — Backend (done)

1. `PromptGenerationRequest.template_content: str | None = None`
   — optional. When provided, the service runs in refine mode.
   Pydantic validator: if provided, must be a non-empty string.
   No separate `mode` field — the presence of `template_content`
   is the signal (keeps the API surface small and avoids a
   generate/refine flag mismatch).
2. `GeneratePromptBody` in the controller picks up the same field
   (it's a mirror of the request, `extra="forbid"`, so the Pydantic
   pass-through handles it for free).
3. `PromptGenerationService._build_messages` branches on
   `request.template_content is not None`:
   - **Refine mode, 0 examples** — fold the existing template
     content + intent into a single user message (A-style). No
     priming, no separate template message. The 0-example case
     doesn't have a "before the examples" slot to insert into,
     so the template naturally lives in the first user message.
   - **Refine mode, N ≥ 1 examples** — reframe the existing
     first user message to mention "refine an existing template";
     then insert a new user message with the existing template
     content + a new fake assistant ack (`"Got it. Send the next
     item."`), positioned right after the priming and before
     the first example. The rest of the structure (examples,
     final "go" message) is unchanged but with "refined" wording
     in the final "go" message.
   - **Generate mode** — unchanged. Preserves the existing
     multi-turn priming pattern verbatim, so Phase 2/3 tests
     keep passing without modification.
4. New `_REFINE_SYSTEM_PROMPT` constant. Loaded from
   `prompts/refine.txt` via `importlib.resources` in
   `service.py`, following the same loading pattern as
   `_GENERATE_SYSTEM_PROMPT` from Phase 5b (the `refine.txt`
   file itself was created in Phase 5b so the on-disk
   structure is already in place — Phase 6a just adds the
   loading line + the constant next to its consumer to
   avoid an import cycle via `__init__.py`). Emphasises:
   - The user has provided an existing template — apply the
     requested changes to it (don't invent a new one from
     scratch).
   - Preserve parts of the template that work and don't need
     to change.
   - Output the full refined template (not a diff). The model
     produces a complete, save-ready template.
   - Same Jinja2 structure rules as the generate system prompt
     (`{% set system_prompt %}` + `{% set user_prompt %}` blocks,
     variable defaults, etc.).
5. Streaming response, error handling, and empty-stream
   detection are unchanged.
6. Tests: 2-3 new service tests covering refine mode for
   0 / 1 / N examples + the `template_content` validation.
   Controller tests pick up the new field automatically (the
   Pydantic body is a mirror).

**Refine mode message structure** (for reference, will be
implemented in code):

- 0 examples, refine:
  ```
  User: "Refine my existing Jinja2 prompt template for image
         captioning.\n\nMy current template:\n<content>\n\n
         Intent for refinement:\n<user_intent>\n\n
         <focus-specific format hints>.\n\n
         Output only the template, without any markdown..."
  ```
- N ≥ 1 examples, refine:
  ```
  User:      "I need to refine my existing Jinja2 prompt template
              for image captioning. Intent: <user_intent>. I'll
              send you N example(s) — each as a subject + caption
              + image — plus my current template, then ask you to
              generate. Briefly acknowledge each item. When I say
              'now generate', emit the refined Jinja2 template."
  Assistant: "Understood. Send your items."        (fake priming)
  User:      "My current template (please refine it according
              to the intent):\n\n<content>"
  Assistant: "Got it. Send the next item."        (new fake ack)
  User:      example 1
  Assistant: "Got it. Send the next example."     (existing ack)
  User:      example 2
  ... (existing pattern continues) ...
  User:      "Now generate the refined Jinja2 template. Output
              only the template, without any markdown..."
  ```

#### Phase 6b — Frontend

1. `PromptFormSettings` (in `stores/prompts/settings.ts`):
   - Add `mode: 'generate' | 'refine'` (default `'generate'`).
   - Bump `$version` to 2. Migration: if the stored `$version` is
     1, the existing `storable(...)` factory will fall back to
     defaults on a version mismatch (it's the documented failure
     path). To avoid losing the user's existing env / apiUrl /
     intent / focus settings on upgrade, ship a small
     `migrate(v1) → v2` helper in the `storable(...)` call that
     re-adds the new `mode: 'generate'` field on top of the v1
     object. The same `migrate` argument pattern is already
     supported by `storable` (see `storable.js`); just unused
     in the current settings stores.
2. `PromptGenerator.svelte`:
   - Add a `PillTabs` at the top of the form column with two
     tabs: "Generate" and "Refine". Binds `mode` via the
     controlled-component pattern (local `mode = $state(s.mode)`
     initialised from `$promptSettings`, single `$effect` to
     sync back via `promptSettings.update`). Avoid
     `bind:mode={$promptSettings.mode}` (the `$store` write
     doesn't propagate — see the rule of thumb in
     `003-prompt-form-settings-persistence.md`).
   - Switching tabs changes the form content but preserves
     the env / model / focus / examples / intent state.
   - The preview stays on the right (always visible, always
     streaming). Tabs do not affect the preview.
3. `PromptForm.svelte`:
   - Add `mode: 'generate' | 'refine'` prop (default
     `'generate'`).
   - Add `templateContent: string` `$bindable` prop (default
     `''`).
   - In refine mode, render a "Template" section between
     Environment and Intent:
     - **Template picker** — `<select>` of `$templates.items`
       (already maintained by the templates store). On change,
       fetch the selected template's content via
       `fetchTemplate(name)` and write it into the bound
       `templateContent`. Show a loading spinner while
       fetching; show an inline error on failure. An empty
       option at the top is the default (user can paste their
       own content without picking from the list).
     - **`JinjaEditor`** — bound to `templateContent` (with
       its `variables` bindable for a chip strip below the
       editor, matching the preview's variable chip styling).
   - The "Intent" label becomes "Refinement intent" in refine
     mode. The underlying `intent` field is the same
     persisted `promptSettings.intent` — we just relabel.
   - The button label in the host becomes "Refine" in refine
     mode (host-owned, so this is in `PromptGenerator.svelte`).
4. `PromptGenRequest` (in `stores/prompts/types.ts`):
   - Add `template_content?: string | null` (matches the
     backend field name; the frontend's camelCase state maps
     to this snake_case wire field via the existing
     `startGeneration` mapping in `actions.ts`).
5. `startGeneration` in `actions.ts`:
   - Pass `template_content: s.templateContent || null` (the
     `s` snapshot is the controlled-component's local form
     values, same as the current `intent` / `focus` mapping).
6. `PromptGenerator.handleGenerate` (in
   `PromptGenerator.svelte`):
   - Validates `templateContent.trim().length > 0` when
     `mode === 'refine'` (in addition to the existing env +
     intent checks). Frontend validation is in addition to
     the backend Pydantic validator — the backend is the
     source of truth, but a friendly client-side error
     avoids an unnecessary round-trip.
7. No new files in `lib/components/prompts/` (everything
   slots into the existing files).
8. Tests: no new vitest tests for Phase 6b. The streaming
   reader tests already cover the wire surface; the
   `template_content` field doesn't change the streaming
   protocol. UI tests are deferred to a future pass (matches
   the existing pattern — none of the Phase 3 components
   have vitest coverage either).

**Persistence summary** for the refine fields:

- `mode` — persisted in `promptSettings` ($version 2).
- `templateContent` — NOT persisted. Ephemeral local state in
  the host. On reload, the user re-selects a template (or
  pastes) to populate the editor.
- Template picker selection — NOT persisted. On reload, the
  picker is empty.
- User's edits to the source template — NOT persisted.
  Ephemeral local state.

**Out of scope (deferred)**:

- "Save over original" flow for the refined output — uses
  the current "Save as new" flow. The user can manually
  overwrite via `/templates` if they want to replace the
  source. Adding an "Overwrite original" option to the
  `EditTemplateDialog` is a separate cross-cutting change
  (would apply to both generate and refine modes), not a
  Phase 6 deliverable.
- Persisting the user's edits to the source template
  (similar to the deferred IndexedDB follow-up for
  examples).
- Persisting `lastTemplateName` (so the user comes back to
  the same selection).
- UI tests for the new form (matches the existing
  pattern — Phase 3 components have no vitest coverage).
- CLI parity for refine mode (defer until Phase 4 lands
  for the base generate flow, then add `--refine <name>` +
  `--template-content <file>` flags mirroring the
  `dataset:<name>:<n>` examples flag).

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
- `yadc/cli.py` → register `prompts` subcommand. *(Phase 4)*
- `yadc/webui/src/routes/+layout.svelte` → nav item (added in
  Phase 3).
- `yadc/webui/src/lib/components/dataset/detail/Preview.svelte`
  → expanded (absorbed the old `ui/PromptPreview` body in
  Phase 3).
- `yadc/webui/src/lib/components/templates/EditTemplateDialog.svelte`
  → gained `initialContent?` prop (Phase 3).
- `yadc/webui/src/lib/icons/SvgSave.svelte` → new icon (Phase 3).
- Documentation: `.pi/agent/memory/docs/backend/captioners.md`
  (clients live in `yadc/llm/` now), `.pi/agent/memory/docs/backend/api.md`
  (new controller), `.pi/agent/memory/docs/captioner-architecture.md`
  (hierarchy collapses; auto-detect facade gone; mixin pattern
  relocates to the client), plan memories. Frontend docs under
  `.pi/agent/memory/docs/frontend/` were updated for Phase 3
  (`stores.md`, `components-domain.md`, `routes.md`).

### Phase 6 (refine mode, planned)

- `yadc/api/services/prompt_generation.py` — add
  `template_content` field + branch in `_build_messages` + new
  `_REFINE_SYSTEM_PROMPT` constant.
- `yadc/api/controllers/api_prompts.py` — `GeneratePromptBody`
  picks up the new field via the existing Pydantic body
  pattern (no body-side changes needed beyond the field
  being added to the inner `PromptGenerationRequest`).
- `tests/api/test_prompt_generation.py` — new tests for refine
  mode (0 / 1 / N examples + `template_content` validation).
- `yadc/webui/src/lib/stores/prompts/types.ts` — add
  `template_content` to `PromptGenRequest`.
- `yadc/webui/src/lib/stores/prompts/settings.ts` — add
  `mode` to `PromptFormSettings`, bump `$version` to 2 with a
  `migrate` helper.
- `yadc/webui/src/lib/components/prompts/PromptGenerator.svelte` —
  add `PillTabs` host + `mode` controlled-component wiring.
- `yadc/webui/src/lib/components/prompts/PromptForm.svelte` —
  add `mode` + `templateContent` props, render the template
  section (picker + `JinjaEditor`) in refine mode, relabel
  the intent textarea.
- `yadc/webui/src/lib/stores/prompts/actions.ts` — pass
  `template_content` in `startGeneration`.
- Memory docs (after impl): `docs/frontend/stores.md`,
  `docs/frontend/components-domain.md`,
  `docs/frontend/components-ui.md` (PillTabs mention in
  PromptGenerator), `docs/backend/api.md` (refine endpoint
  field).

### Phase 5b (backend cleanup, planned)

Refactor + bug fix, all in `yadc/api/services/prompt_generation/`.

- Delete `yadc/api/services/prompt_generation.py`.
- Create the `yadc/api/services/prompt_generation/` package:
  - `__init__.py` (re-exports the public surface; also re-exports
    `cmd_envs` + `create_client` at the package level for the
    existing test patches — nothing else, to avoid an import
    cycle with `service.py`).
  - `service.py` (extracted `PromptGenerationService` +
    `_build_messages` + `_EXAMPLE_ACK`; new `system_prompt` kwarg
    on `_build_messages`; prepends `Message(role="system", ...)`
    to the returned list; loads `prompts/generate.txt` via
    `importlib.resources` and defines `_GENERATE_SYSTEM_PROMPT` at
    module level — the constant lives next to its only consumer
    to avoid an import cycle via `__init__.py`).
  - `prompts/generate.txt` (current `_SYSTEM_PROMPT` text).
  - `prompts/refine.txt` (refine system prompt text — file is
    created in Phase 5b so the structure is set up, but the
    constant + loading line are **added in Phase 6a** when
    `_build_messages` actually branches on it).
- `tests/api/test_prompt_generation.py`:
  - Shift `messages[i]` index assertions in 5 tests by 1 (the
    system message now occupies index 0).
  - New `test_system_message_is_first` verifying the system
    message is at index 0 and its content matches the loaded
    `_GENERATE_SYSTEM_PROMPT` (the test exercises the same
    loading path as production by reading the constant directly
    from `service.py`).
  - `_PATCH_CMD_ENVS` + `_PATCH_CREATE_CLIENT` paths unchanged
    (the `__init__.py` re-export keeps them resolving).
- Memory docs (after impl): `docs/backend/services.md` (or
  whichever doc covers the services tree) needs the file → package
  update; `docs/backend/api.md` may want a note about the
  system-prompt wire-up if it discusses the message shape.

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
   Worth tuning — defer to Phase 5. Complementary: Phase 8 would
   externalize the meta-prompt scaffolding (the inlined user/assistant
   turns in `_build_messages`) into a `messages.jinja` so the wording
   is maintainable and (later) user-overridable — separate from tuning
   the content itself.

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

### Phase 7 — Prompt-history persistence (planned)

Server-side persistence of the artifact (intent + focus + examples
+ mode + template content) so the user can save and restore the
full prompt across sessions/machines. Defers / supersedes the
IndexedDB TODO entry.

- **DB migration `0008_prompt_history_{up,down}.sql`**: two tables —
  `prompt_history` (mode, intent, focus, template_content, created_t)
  and `prompt_example_images` (raw image BLOBs, one row per example
  image, FK to the parent with `ON DELETE CASCADE`). The DB stores
  uncompressed bytes — the service base64-translates at the wire
  boundary (see the design history entry for the full schema +
  `CHECK` constraints).
- **`yadc/api/services/prompt_history_repository.py`**:
  `PromptHistoryEntry` (entry-level row) + `PromptExample`
  (per-image child row) dataclasses + all SQL for both tables.
  `list_entries_with_counts` is a single `LEFT JOIN ... GROUP BY`
  so `example_count` comes back attached (no per-row JSON parse).
  `get_entry_with_examples` returns `(entry, examples)` from one
  connection. `insert_entry` + `insert_examples` + `prune_old`
  run inside one transaction for save atomicity; `delete_entry`
  cascades to image rows via the FK — no service-level cleanup.
  Inline row→dataclass construction per the repo convention.
- **`yadc/api/services/prompt_history.py`**: `PromptHistoryService`
  with `save_entry` (insert entry + insert examples + prune to 20
  inside one transaction, matching `ConfigHistoryService.save_snapshot`'s
  pattern), `list_history` (opaque `next` cursor), `get_entry`,
  `delete_entry`. The service owns the **base64 ↔ BLOB translation
  at the wire boundary**: `ExamplePair.image_data_url` is the wire
  format; the repo stores raw bytes (decode on save, re-encode on
  get). Malformed data URLs are silently skipped on save.
- **`yadc/api/controllers/api_prompts_history.py`**: four
  endpoints (`POST` / `GET` list / `GET` id / `DELETE` id).
  Pydantic body model for save reuses the existing
  `ExamplePair` from the `prompt_generation` service.
  List response includes server-derived `intent_preview`
  (first 2 lines) + `example_count` + `had_template` so the
  frontend doesn't have to compute them.
- **`yadc/webui/src/lib/stores/prompts/{types,api,index}.ts`**:
  `PromptHistoryListItem` + `PromptHistoryEntry` types; four
  API helpers (`fetchHistoryList`, `fetchHistoryEntry`,
  `saveHistoryEntry`, `deleteHistoryEntry`).
- **`yadc/webui/src/lib/components/prompts/PromptHistoryPanel.svelte`**:
  list view with the user-specified layout (intent preview,
  badges for focus / example count / mode / "with template",
  relative timestamp, restore + delete actions, confirm
  dialog for delete).
- **`yadc/webui/src/lib/components/prompts/PromptGenerator.svelte`**:
  third tab in the form-card `PillTabs`
  (`Generate | Refine | History`), "Save to history" button
  in the form footer, restore handler that switches tab +
  populates `mode` / `intent` / `focus` / `examples` /
  `templateContent` from the entry + toasts confirmation.
- Tests: `tests/api/test_prompt_history_repository.py` +
  `tests/api/test_prompt_history_service.py` +
  `tests/api/test_api_prompts_history.py`.
- Doc updates: `backend/api.md`, `frontend/components-domain.md`,
  `frontend/stores.md`, `todo.md` (remove the superseded
  IndexedDB entry).
- **Out of scope**: env / model / template picker selection
  (workspace-level, not part of the artifact); per-entry
  pin/favorite; search/filter; CLI parity (wait for Phase 4).

See `history/prompt-generator-plan/008-phase-7-design.md` for
the full design rationale (scope decisions, prune policy,
schema, API surface, list-item layout).

### Phase 8 — Meta-prompt Jinja templating (proposed, not yet implemented)

**Goal**: move the meta-prompt **scaffolding** — the inlined
user/assistant message bodies in `_build_messages` (priming acks,
intro turns, the final "go" message, the per-example label, the focus
format hints) and the `_EXAMPLE_ACK` / `_REFINE_TEMPLATE_ACK`
constants — out of `service.py` into a Jinja2 template, so the wording
lives in data, not Python. Motivation: maintainability + enabling a
later feature where users override these instructions with their own
(via a `ChoiceLoader` resolution chain mirroring caption templates).
That override UX is a separate later feature — Phase 8 only does the
externalization.

**Decisions (from the design pass)**:

- **Macros throughout** (`{% macro x(args) %}`), not `{% set %}`
  blocks. The per-example label varies inside the example loop, and
  `{% set %}` blocks evaluate once with the call's globals (can't
  vary per iteration), so a macro is forced there; uniformity then
  favours macros for the rest. (The captioner uses `{% set %}` blocks
  because its templates are user-authored output blocks, not
  per-call-arg scaffolding.)
- **Loading**: one module-level `Environment`
  (`trim_blocks`/`lstrip_blocks` matching `PromptRenderer`);
  `get_template("messages.jinja").module` cached once; macros called
  with explicit args. Verified that `get_template(name, globals=…)`
  on a cached template **sticks** globals from the first call, so the
  globals-based form is avoided in favour of explicit-arg macros.
- **Focus hints**: live in `messages.jinja` as a `focus_hints(focus)`
  macro. They contain literal `{% set … %}` example text, so those
  passages are wrapped in `{% raw %}…{% endraw %}`. (Two existing
  `"block.Use"` typos in `_format_hints` get fixed in the move.)
- **System prompts stay as `.txt`**: `generate.txt` / `refine.txt`
  are unchanged (no variables → nothing to gain from Jinja). Phase 8
  moves only the scaffolding.

**File layout**: new
`yadc/api/services/prompt_generation/prompts/messages.jinja` (~12
macros + shared `focus_hints`); `service.py` `_build_messages` becomes
pure orchestration (every `Message(...)` body is a `_MSGS.<macro>(…)`
call; the example turn wraps `_MSGS.example_label(…)` in a `TextPart`
+ `ImageUrlPart`). `_format_hints`, `_EXAMPLE_ACK`,
`_REFINE_TEMPLATE_ACK` are deleted.

**Test impact**: `tests/api/test_prompt_generation.py` deliberately
doesn't pin scaffolding wording (only the two system-prompt constants
+ the example structure), so the refactor is low-risk. Add one smoke
test asserting `_build_messages` yields the right roles/turn-count per
scenario so a future template edit can't silently drop a turn.

**Status**: proposed — awaiting a go/no-go. See
`history/prompt-generator-plan/010-meta-prompt-jinja-templating.md`
for the full design (incl. the Jinja-mechanics findings that drove the
macro choice, and the full `messages.jinja` + slimmed `_build_messages`
sketches).

## Status

**Phase 1a + 1b + 2 + 3 COMPLETE.** Phase 3 lands the web UI
end-to-end:

- `lib/stores/prompts/` (types, streaming api, runes-based state,
  high-level actions, vitest tests for the streaming reader)
- `lib/components/prompts/` feature folder (PromptGenerator host,
  PromptForm, ExamplesPanel with manual + from-dataset modes,
  GenerationPreview with streaming/copy/save)
- `routes/prompts/+page.svelte` and a new "Prompts" nav entry
  in `routes/+layout.svelte` (between Datasets and Templates)
- The pre-existing `lib/components/ui/PromptPreview.svelte`
  (one-use wrapper around the dataset image prompt preview) was
  inlined into `dataset/detail/Preview.svelte` and deleted, freeing
  the name for the new `prompts/GenerationPreview` (no collision).
- `EditTemplateDialog` got an `initialContent?` prop so the
  generator's "Save as template" button pre-fills the editor.
- 13 new vitest tests for the streaming reader, 50 frontend tests
  total all passing. Backend 726 tests still pass. svelte-check,
  eslint, prettier, ruff, basedpyright, build all clean.

Phase 4 (CLI parity) and Phase 5 (meta-prompt tuning) remain.
**Phase 8 (meta-prompt Jinja templating) — proposed, awaiting a
go/no-go** (see the Phase 8 section above +
`010-meta-prompt-jinja-templating.md`).
**Phase 5b (backend cleanup, DONE)**,
**Phase 6a (refine-mode backend, DONE)**,
**Phase 6b (refine-mode frontend, DONE)**,
**Phase 7 (prompt-history persistence, DONE)**. See
`history/prompt-generator-plan/002-phase-3-frontend.md` for the
detailed Phase 3 record,
`history/prompt-generator-plan/004-refine-mode-design.md` for the
Phase 6 design rationale,
`history/prompt-generator-plan/005-backend-cleanup-design.md`
for the Phase 5b design rationale,
`history/prompt-generator-plan/006-phase-5b-implementation.md`
for the Phase 5b implementation record (including the
constant-location and test-patch-path corrections that
happened during the work), and
`history/prompt-generator-plan/007-phase-6a-implementation.md`
for the Phase 6a (refine-mode backend) implementation record
(including the wording decisions that came up during
implementation), and
`history/prompt-generator-plan/008-phase-6b-implementation.md`
for the Phase 6b (refine-mode frontend) implementation record
(including the `storable.js` migrate-bug fix, the form-in-both-tabs
trade-off (later replaced — see `011-combined-form-mode-toggle.md`),
and the ESLint `argsIgnorePattern` config), and
`history/prompt-generator-plan/008-phase-7-design.md` /
`009-phase-7-implementation.md` for the prompt-history
persistence (server-side replacement for the deferred
IndexedDB plan; the implementation entry covers the
`activeTab`/`mode` split, the `untrack(() => mode)` init
pattern (removed in the combined-form revision — see `011-combined-form-mode-toggle.md`),
the save/restore flow, and the `mode + examples +
template_content` JSON contract).

**Phase 6b revised (2026-06-20): the Generate/Refine tab switcher
was replaced with a single combined form + inline New/Refine pill
toggle.** The original form-in-both-tabs approach mounted the env
selector (+ model/template fetches) twice and lost per-instance
state on tab switch; the form is now rendered once, with the
New/Refine mode as an inline pill toggle between Focus and the
examples. See `history/prompt-generator-plan/011-combined-form-mode-toggle.md`.

**2026-06-20 — Settings tab + configurable max_tokens / image_quality.**
The env selector moved out of the Generate form into a dedicated
**Settings tab** (Generate / Settings / History), which also adds two
previously-missing generation knobs: `max_tokens` (was hardcoded to
25000 in `service.py`; client default of 4096 truncated templates) and
`image_quality` (wasn't plumbed in at all). Both are full-stack
(`image_quality` is FE+BE — OpenAI reads it from `image_url.detail`,
Gemini from a client opt). `promptSettings` → `$version: 3`. See
`history/prompt-generator-plan/012-settings-tab-max-tokens-image-quality.md`.

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
* 2026-06-20 — after testing the MVP (Phases 1–3), the user
  asked to add a "refine" mode: take an existing template, edit
  it in the form, and ask the LLM to apply a refinement intent.
  LLM-side: insert a new user message with the existing template
  + a fake assistant ack, before the example images (0-example
  case folds into the first user message). Frontend: pill tabs
  at the top of the form column for "Generate" / "Refine",
  reusing the existing `PillTabs` component. Editing of the
* 2026-06-20 — during Phase 5b implementation, the user
  corrected the constant location: `importlib.resources`
  loading goes in `service.py` (next to its only consumer),
  NOT in `__init__.py` or a separate `_prompts.py` loader
  module. Putting it in the package would create a circular
  import (`__init__.py` imports from `service.py` to
  re-export; `service.py` importing the package back re-enters
  `__init__.py` mid-execution). Same argument rules out a
  dedicated `_prompts.py` re-export through `__init__.py`.
  Kept simple: one `_GENERATE_SYSTEM_PROMPT` private
  constant in `service.py`, loaded at import time. Also
  the user said: "don't rewrite the service and just use
  mkdir and mv" for the file → package refactor (i.e. don't
  try to retype the whole file, just move it and make
  targeted edits on top). And switched the `..modules.*`
  relative imports in `service.py` to absolute `yadc.api.modules.*`
  imports (matching the convention in the other
  `services/*.py` files).
  source template happens in the form (via `JinjaEditor`), not
  in the backend. Captured as **Phase 6** in this plan; see
  `history/prompt-generator-plan/004-refine-mode-design.md` for
  the design rationale and the deferred items.
* 2026-06-20 — in the same session, the user asked to "start
  cleaning up the backend" before adding the refine system
  prompt. Concrete asks: convert the single
  `prompt_generation.py` into a `prompt_generation/` package
  (same import path → reduce blast radius) and move the system
  prompts out of the Python file into `.txt` files. While
  planning this, we also discovered that the existing
  `_SYSTEM_PROMPT` constant is **dead code** — defined at the
  bottom of the file but never referenced by `_build_messages`
  (which only returns user/assistant turns). The cleanup phase
  wires the system prompt back in as a `Message(role="system",
  ...)` prepended to the returned list, which is a real
  behaviour change (the LLM client already maps `role="system"`
  to its native shape — OpenAI `system`, Gemini
  `system_instruction` — so no client changes are needed).
  Captured as **Phase 5b** in this plan; see
  `history/prompt-generator-plan/005-backend-cleanup-design.md`
  for the full design rationale.
