---
date: 2026-06-20
---
# Settings tab + configurable max_tokens & image_quality

**Context:** Two gaps in the prompt generator after the combined-form
revision (`011-combined-form-mode-toggle.md`):

1. The env selector (and model picker) lived in the Generate form. The
   user wanted it in a dedicated **Settings tab** instead, alongside
   two generation knobs that had no UI at all.
2. `max_tokens` was **hardcoded to 25000** in `service.py` because the
   client default (4096) truncated longer templates. There was no way
   to set it from the frontend. `image_quality` was likewise unsettable
   (and not even plumbed into the service — `_build_messages` hardcoded
   `detail="auto"`).

**Scope decision (with the user):** ship **both** `max_tokens` and
`image_quality` (the marginal cost is small once the Settings tab
exists). Default `max_tokens` to **16384** (matches the captioning
config ceiling; safe for most models; well above the 4k that bit the
user — bump higher in the field if a model allows). Range 100–65536.

**image_quality is FE + BE (forced by architecture):** OpenAI carries
image quality on the message (`ImageUrl.detail`); Gemini reads it from
the `predict_next_message_stream` opts (`→ mediaResolution`). The
captioner already does both (sets `detail` on the part + passes the
opt). The prompt generator now mirrors that: `_build_messages` sets
`detail=image_quality` on each example `ImageUrlPart` **and** `generate`
passes `image_quality` as an opt to `predict_next_message_stream`.

**Frontend restructure — three tabs:**
- `PromptGenerator.svelte`: `PillTabs` is now Generate / Settings /
  History (genuinely different content per tab — the legitimate
  PillTabs use). New host `$state` `maxTokens` + `imageQuality`
  (persisted); the footer action bar (Save-to-history + Generate) now
  renders only on the Generate tab.
- **New `PromptSettings.svelte`** (Settings tab): the Environment
  section (env `<select>` + API URL + token + model picker with
  refresh, auto pre-filled from the env) **moved here** from
  `PromptForm`, plus a Generation-limits section (`max_tokens` number
  input + `image_quality` auto/low/high pills). All inputs disable
  while a generation is streaming.
- `PromptForm.svelte` (Generate tab): slimmed to Intent + Focus +
  Mode pills + Template (refine) + Examples. The env/model logic
  (`loadEnvs`, `loadModels`, `envInfo`, the pre-fill `$effect`) and
  the `SvgSpinner`/`SvgRefresh` imports moved to `PromptSettings`.

**Persistence:** `promptSettings` bumped to **`$version: 3`** with a
single shape-agnostic `migrateToCurrent` (overlays `DEFAULTS` on any
prior stored object — `maxTokens`/`imageQuality` (added v3) and `mode`
(added v2) fall back to defaults; existing fields preserved). The old
per-version `migrateToV2` is replaced. History entries are unchanged
(they store the artifact, not generation params — max_tokens /
image_quality are workspace settings, not part of the saved prompt).

**Backend:**
- `PromptGenerationRequest` + `GeneratePromptBody`: added
  `max_tokens: int = Field(16384, ge=100, le=65536)` and
  `image_quality: Literal["auto","low","high"] = "auto"`.
- Controller passes both through to `request_model`.
- `service.py`: `_build_messages` gains an `image_quality` kwarg
  (used for `ImageUrl(..., detail=image_quality)`); `generate` passes
  `max_tokens=request.max_tokens` + `image_quality=request.image_quality`
  to `predict_next_message_stream` (removes the hardcoded 25000).
- `Literal` import added to the controller.

**Tests:** added `test_max_tokens_bounds`,
`test_image_quality_rejects_invalid`,
`test_max_tokens_forwarded_to_client`,
`test_image_quality_forwarded_to_client_and_message` (asserts both the
opt and the `image_url.detail` on the example parts). `_request` helper
gained `max_tokens` / `image_quality` params. Controller tests use
minimal bodies so the new optional fields don't break them.

**Checks:** backend 21/21 prompt-generation + 18/18 controller tests
pass; basedpyright clean; frontend ESLint + Prettier + svelte-check +
`npm run build` + 13/13 vitest all clean.

**Files touched:**
- `yadc/api/services/prompt_generation/service.py`
- `yadc/api/controllers/api_prompts.py`
- `tests/api/test_prompt_generation.py`
- `yadc/webui/src/lib/stores/prompts/types.ts` (added `PromptImageQuality`)
- `yadc/webui/src/lib/stores/prompts/actions.ts` (`maxTokens` /
  `imageQuality` in `StartGenerationArgs` + request)
- `yadc/webui/src/lib/stores/prompts/settings.ts` (v3 + new fields)
- `yadc/webui/src/lib/components/prompts/PromptSettings.svelte` (new)
- `yadc/webui/src/lib/components/prompts/PromptForm.svelte` (slimmed)
- `yadc/webui/src/lib/components/prompts/PromptGenerator.svelte`
  (3 tabs + new state)
- `yadc/webui/src/lib/components/prompts/index.ts` (export
  `PromptSettings`)
- Memory docs: `docs/frontend/components-domain.md`,
  `docs/frontend/stores.md`, `docs/backend/api.md`.
