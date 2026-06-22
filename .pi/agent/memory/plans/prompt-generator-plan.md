---
name: prompt-generator-plan
description: Meta-prompting feature — generate/refine Jinja2 prompt templates from an intent + optional few-shot examples using any configured env's model, with streaming + cancel + server-side prompt history. Built on a generic LLM client layer (yadc/llm/) extracted from the captioners.
last_history: 14
---

# Prompt Generator Plan

## Goal

Use an existing env's model to **generate or refine high-quality
Jinja2 prompt templates** from a user-stated intent and optional
few-shot examples. The output is a Jinja2 template (`system_prompt` +
`user_prompt` blocks) that can be saved via the existing
`/api/templates` endpoint and immediately used for captioning.

The feature runs in two modes — **Generate** (new template from an
intent + optional style-reference examples) and **Refine** (apply a
refinement intent to an existing template the user picks + edits in
the form) — plus a server-side **history** of past artifacts the user
can restore. Streaming preview + cancel are shared across both modes.

The work also pulled a generic **LLM client layer** (`yadc/llm/`) out
of the old per-backend captioners, so the chat-completion mechanics
(message construction, HTTP transport, reasoning handling, streaming)
live in one place and image encoding stays in the captioners.

> The original full design — every decision, the architecture sketch
> with code, the planned phase shape, resolved decisions, and the
> status / user-feedback notes as they stood — is preserved in
> [`history/prompt-generator-plan/013-design-and-done-phase-detail.md`](history/prompt-generator-plan/013-design-and-done-phase-detail.md).
> Per-phase *implementation* records are `001`–`012`. This plan is the
> current-state tracker; reach for those entries when you need detail.

## Current State

Generation, refine mode, prompt-history persistence, and the settings
tab (configurable `max_tokens` / `image_quality`) are all shipped
end-to-end. **Remaining:** Phase 4 (CLI parity), Phase 5 (optional
polish — partly superseded by Phase 7), and Phase 8 (meta-prompt
Jinja templating, proposed — awaiting a go/no-go).

## Architecture (what was built)

### `yadc/llm/` client layer

A top-level package holding the unified chat-completion mechanics,
extracted from the old per-backend captioners:

- **Types** (`types.py`): `Message` (OpenAI-style `role` + `content`,
  where `content` is `str` or `list[TextPart | ImageUrlPart]`; carries
  `reasoning` / `reasoning_summary` / `reasoning_encrypted` for
  assistant turns), `StreamChunk` (`text` / `reasoning` /
  `reasoning_summary` / `reasoning_encrypted`), `MessageStream`.
- **`MessageStream`** — the return type of every prediction call.
  `predict_next_message_stream` is `async def`: awaiting it performs
  the POST and returns a stream wrapping an *already-open* response,
  so `cancel()` is a concrete `aclose()` (no background task/queue).
  `async for chunk in stream` self-accumulates into `.message`;
  `await stream.collect()` drains + returns the final `Message`;
  `await stream.cancel()` closes the upstream (idempotent).
  **Stream-only** — no separate non-streaming method (llama.cpp
  doesn't cleanly stop in non-streaming mode).
- **`BaseLLMClient`** (`base.py`) — owns `AsyncSession`,
  `HTTPResponseCache`, `ResponseLogger`; declares
  `predict_next_message_stream`, `list_models`, `load_model`,
  `unload_model`, `offload_model`. `load_model` validates model
  existence (cloud) or loads into VRAM (local).
- **`OpenAICompatibleLLMClient`** (`openai_compatible.py`) — the real
  machinery for any `/v1/chat/completions` server: message translation,
  response parsing (with `reasoning_details` capture), streaming.
  API-type probing lives here as a logging/detection concern only
  (one implementation regardless of which server it is).
- **Thin per-server subclasses** (`openai.py`, `openrouter.py`,
  `vllm.py`, `llamacpp.py`, `koboldcpp.py`, `ollama.py`) — exist for
  type identity + server-specific `load_model` (local APIs actually
  load GGUF/etc. into VRAM).
- **`GeminiLLMClient`** (`gemini.py`) — `generateContent` translation
  incl. reasoning.
- **`create_client(env, **overrides)`** (`factory.py`) — URL-based type
  inference → the right client.

Key impl note: the existing `AsyncSession.request(stream=True)` was
**fake streaming** (it dropped `stream` before reaching httpx, so the
whole body buffered). A new additive `AsyncSession.open_stream()`
(built on `client.send(req, stream=True)`) gives the client layer
**true** streaming, needed for `MessageStream.cancel()` to be
meaningful; the legacy `request()`/`get()`/`post()` paths are
untouched.

### Captioner collapse (Phase 1b)

All per-server captioner classes collapsed into a single
**`APICaptioner`** that holds any `BaseLLMClient` via composition.
The client interface is unified enough that one captioner works for
OpenAI-compatible + Gemini. **Public API is unchanged**
(`predict`, `predict_stream`, `load_model`, `unload_model`,
`offload_model`; `extra_messages` + `prediction_context` kwargs) —
`AsyncCaptionJob`, the CLI runner, and existing tests don't move.
`load_model`/`unload_model`/`offload_model` proxy to the client. The
old `APICaptioner.create()` factory is gone; callers do
`client = create_client(user_config)` + `APICaptioner(client=...)`.

### Prompt generation (Phase 2)

`PromptGenerationService` (`yadc/api/services/prompt_generation/`)
uses the client **directly** (no captioner — no image). It
`load_model`s, builds a multi-turn meta-conversation (system prompt +
intent + few-shot examples as text+image parts + a final "now
generate" turn), and yields `StreamChunk`s from
`predict_next_message_stream(messages, reasoning=None)` (reasoning
off — the artifact is the template body).

`POST /api/prompts/generate` (`api_prompts.py`) returns a streaming
**NDJSON** response (`{"type":"token",...}` / `{"type":"done"}` /
`{"type":"error",...}`), with synchronous preflight (env load → 403
on password-required) before the stream starts. Cancel is implicit in
client disconnect → Quart cancels the response coroutine →
`CancelledError` → `MessageStream.cancel()` → upstream close.

The service is a **package**: `__init__.py` re-exports the public
surface (+ `cmd_envs` / `create_client` for the test patches);
`service.py` holds `PromptGenerationService` + `_build_messages`;
`prompts/generate.txt` + `prompts/refine.txt` are the system prompts,
loaded once at import time via `importlib.resources` (the constant
lives in `service.py` next to its only consumer to avoid an import
cycle through `__init__.py`). Phase 5b also fixed a latent bug: the
old `_SYSTEM_PROMPT` was dead code — it's now wired in as
`Message(role="system", ...)` prepended to the message list.

### Refine mode (Phase 6)

A second mode that takes an existing template (user-picked +
user-editable in the form) and asks the LLM to apply a refinement
intent. The backend signals it via an optional `template_content`
field on `PromptGenerationRequest` (no separate `mode` flag —
presence is the signal); `_build_messages` branches on it: the
0-example case folds template + intent into the first user message,
the N≥1 case inserts the template + a fake assistant ack after the
priming. `_REFINE_SYSTEM_PROMPT` (loaded from `prompts/refine.txt`)
emphasises applying changes to the existing template, preserving what
works, and outputting the full refined template (not a diff).

Frontend: the Generate/Refine **tab switcher was replaced** with a
single combined form + an inline New/Refine pill toggle (the original
tab approach mounted the env selector twice and lost per-instance
state on switch — see `011`). The streaming preview + "Save as
template" flow are shared. `promptSettings` carries `mode`.

### Prompt-history persistence (Phase 7)

Server-side persistence of the full artifact (mode + intent + focus +
template content + example images) so the user can save/restore across
sessions — supersedes the earlier IndexedDB TODO. DB migration
`0008_prompt_history` adds `prompt_history` + `prompt_example_images`
(raw image BLOBs, FK `ON DELETE CASCADE`); the service does the
base64↔BLOB translation at the wire boundary.
`prompt_history_repository.py` (entry + per-image dataclasses; one
`LEFT JOIN ... GROUP BY` for counts) + `prompt_history.py`
(`PromptHistoryService`: save with prune-to-20 in one transaction,
opaque-cursor list, get, delete) + `api_prompts_history.py` (4
endpoints; list response carries server-derived `intent_preview` /
`example_count` / `had_template`). Frontend: a **History** tab +
"Save to history" button + `PromptHistoryPanel.svelte` (intent
preview, badges, relative timestamp, restore + delete with confirm).

### Settings tab + generation knobs (2026-06-20)

The env selector moved out of the Generate form into a dedicated
**Settings tab** (Generate / Settings / History), which also plumbed
through two previously-missing knobs: `max_tokens` (was hardcoded to
25000; client default 4096 truncated templates) and `image_quality`
(FE+BE — OpenAI reads it from `image_url.detail`, Gemini from a
client opt). `promptSettings` → `$version: 3`.

### UI side panel + two fixes (2026-06-22)

The dataset route's `SidePanel.svelte` was hoisted into a reusable
**generic `ui/SidePanel.svelte`** primitive (`children` snippet +
`bind:open` + `class` + `onclose`, FAB toggle built in, mobile
drawer). The dataset's wrapper became `DatasetSidePanel.svelte`. A
new `PromptSidePanel.svelte` wraps the primitive and hosts the three
tabs + footer actions; `PromptGenerator.svelte` flipped to a
full-page `GenerationPreview` + side-panel layout (form state stays
in the host; `startGeneration` is called with a snapshot of form
values so mid-stream edits don't leak). Two latent bugs were fixed
in the process: the outside-click-to-close check moved from
`panelRef.contains(target)` to `e.composedPath().includes(panelRef)`
(Svelte 5 flushes `$state` synchronously inside click handlers, so
the target can be detached by the time the bubble-phase listener
runs) and the preview wrapper gained `min-w-0` so unbreakable
content (long reasoning lines) can no longer push the panel
rightward. ReasoningCard's expanded `<pre>` also gained
`break-words`. See
[`history/prompt-generator-plan/014-prompt-ui-side-panel-and-fixes.md`](history/prompt-generator-plan/014-prompt-ui-side-panel-and-fixes.md)
for the full record.

## Phases

### Done

| Phase | What | Records |
|-------|------|---------|
| **1a** | Extract `yadc/llm/` client layer (additive; captioners untouched). | `013` |
| **1b** | Collapse per-server captioners onto the client (single `APICaptioner`). | `013` |
| **2** | Backend prompt generation (service + `POST /api/prompts/generate`, NDJSON streaming). | `001`, `013` |
| **3** | Frontend (stores, components, `/prompts` route, nav item; `EditTemplateDialog.initialContent`). | `002`, `003`, `013` |
| **5b** | Backend cleanup (file→package, `.txt` system prompts, wire-the-system-prompt bug fix). | `005`, `006`, `013` |
| **6** | Refine mode (6a backend `template_content` branching + `_REFINE_SYSTEM_PROMPT`; 6b frontend — later revised to a combined form + inline pill toggle). | `004`, `007`, `011`, `013` |
| **7** | Prompt-history persistence (DB + repository + service + controller + History tab/panel). | `008`, `009`, `013` |

Plus the 2026-06-20 settings-tab + `max_tokens`/`image_quality` work (`012`).

### Remaining

#### Phase 4 — CLI parity

1. `cmd/prompts/` pure logic.
2. `cli_prompts.py` click commands.
3. Register in the main group (`cli.py`).
4. Tests.

CLI `generate` writes the **template body to stdout** (progress /
status / reasoning → stderr) so it pipes cleanly into `prompts save`.
`prompts save <name>` reads the body from stdin and saves via
`cmd_templates.save_user_template`. Mirror the cmd/cli split: click in
`cli_prompts.py`, pure logic in `cmd/prompts/`.

#### Phase 5 (optional) — Polish

- Persist recent generations in `SettingsService` (KV store).
- Copy-to-clipboard for the generated template body.
- Show extracted Jinja variables inline as chips.
- A/B test variations of the meta-system-prompt.
- ~~IndexedDB-backed persistence for the few-shot examples~~ —
  superseded by Phase 7's server-side history. See
  `003-prompt-form-settings-persistence.md` for the original
  localStorage plan.

#### Phase 8 — Meta-prompt Jinja templating (proposed, awaiting go/no-go)

Move the meta-prompt **scaffolding** — the inlined user/assistant
message bodies in `_build_messages` (priming acks, intro turns, the
final "go" message, the per-example label, the focus format hints)
and the `_EXAMPLE_ACK` / `_REFINE_TEMPLATE_ACK` constants — out of
`service.py` into a Jinja2 template, so the wording lives in data,
not Python. Motivation: maintainability + enabling a later feature
where users override these instructions (via a `ChoiceLoader` chain
mirroring caption templates). That override UX is a separate later
feature — Phase 8 only does the externalization.

Decisions from the design pass:

- **Macros throughout** (`{% macro x(args) %}`), not `{% set %}`
  blocks. The per-example label varies inside the example loop and
  `{% set %}` blocks can't vary per iteration, so a macro is forced
  there; uniformity favours macros for the rest.
- **Loading**: one module-level `Environment`
  (`trim_blocks`/`lstrip_blocks` matching `PromptRenderer`);
  `get_template("messages.jinja").module` cached once; macros called
  with explicit args (`get_template(name, globals=…)` sticks globals
  from the first call on a cached template, so the globals form is
  avoided).
- **Focus hints**: live in `messages.jinja` as a `focus_hints(focus)`
  macro, with literal `{% set … %}` example text wrapped in
  `{% raw %}…{% endraw %}`. (Two existing `"block.Use"` typos in
  `_format_hints` get fixed in the move.)
- **System prompts stay as `.txt`**: `generate.txt` / `refine.txt`
  are unchanged (no variables → nothing to gain from Jinja). Phase 8
  moves only the scaffolding.

File layout: new
`yadc/api/services/prompt_generation/prompts/messages.jinja` (~12
macros + shared `focus_hints`); `service.py` `_build_messages` becomes
pure orchestration (every `Message(...)` body is a `_MSGS.<macro>(…)`
call). `_format_hints`, `_EXAMPLE_ACK`, `_REFINE_TEMPLATE_ACK` are
deleted.

Test impact: `tests/api/test_prompt_generation.py` deliberately
doesn't pin scaffolding wording, so the refactor is low-risk; add one
smoke test asserting `_build_messages` yields the right roles /
turn-count per scenario. See
[`history/prompt-generator-plan/010-meta-prompt-jinja-templating.md`](history/prompt-generator-plan/010-meta-prompt-jinja-templating.md)
for the full design (Jinja-mechanics findings + the full
`messages.jinja` + slimmed `_build_messages` sketches).

## Key Files

**New (`yadc/llm/`)** — `__init__.py`, `types.py`, `base.py`,
`openai_compatible.py`, `openai.py`, `openrouter.py`, `vllm.py`,
`llamacpp.py`, `koboldcpp.py`, `ollama.py`, `gemini.py`, `factory.py`.

**New (prompt generation)** — `yadc/api/services/prompt_generation/`
(package: `__init__.py`, `service.py`, `prompts/{generate,refine}.txt`);
`yadc/api/controllers/api_prompts.py`; `yadc/cli_prompts.py` +
`yadc/cmd/prompts/` *(Phase 4 — not yet)*.

**New (prompt history)** — DB migration `0008_prompt_history_*`;
`yadc/api/services/prompt_history_repository.py`;
`yadc/api/services/prompt_history.py`;
`yadc/api/controllers/api_prompts_history.py`.

**Refactored** — `yadc/captioners/api/`: the per-server
`openai.py`/`gemini.py`/`openrouter.py`/`vllm.py`/`llamacpp.py`/
`koboldcpp.py`/`ollama.py` are deleted in favour of a single
`APICaptioner` holding a `BaseLLMClient`; `api_captioner.py` is
slimmed (composition); `base.py` HTTP infra moved to `yadc/llm/base.py`.

**Frontend** — `routes/prompts/+page.svelte` + "Prompts" nav item;
`lib/components/prompts/` feature folder (`PromptGenerator`,
`PromptForm`, `ExamplesPanel`, `GenerationPreview`,
`PromptHistoryPanel`); `lib/stores/prompts/` (types, api, store,
actions, settings). `EditTemplateDialog` gained `initialContent?`;
the old one-use `ui/PromptPreview.svelte` was inlined into
`dataset/detail/Preview.svelte` and deleted.

**Docs** — `docs/backend/captioners.md` (clients live in `yadc/llm/`
now), `docs/backend/api.md` (new controllers),
`docs/captioner-architecture.md` (hierarchy collapses),
`docs/frontend/{stores,components-domain,routes,components-ui}.md`.

## Open Questions

1. **Meta-prompt stability / maintainability** — the system prompt +
   scaffolding wording is a critical piece of UX and worth tuning
   (Phase 5). Phase 8 externalizes the scaffolding into
   `messages.jinja` to make it maintainable (and later
   user-overridable); tuning the *content* itself is separate from
   that externalization.

*(The original Open Questions — cancel-propagation end-to-end,
per-server `load_model` overrides, reasoning-in-captioners — were all
resolved during Phase 1/3; see `013`.)*

## Follow-ups

- **Docstring cleanup** — several docstrings still reference "the
  plan" / the refactoring narrative. Long-term they should describe
  the *current* state of their file, not the migration. (User note;
  not tied to a phase.)
