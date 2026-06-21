---
date: 2026-06-20
---
# Phase 3 — Frontend prompt generator (complete)

**Context:** The prompt-generator plan's Phase 3 — the web UI
side. The backend plumbing (Phase 2) is in place
(`POST /api/prompts/generate` returns streaming NDJSON: token /
done / error lines). The frontend needs stores, components, a
route, and a sidebar entry.

**Decision:** Implemented as designed, with one consolidation
approved by the user during the plan check-in:

- The plan called for two separate example sub-panels
  (`DatasetExamplesPanel` + `ManualExamplesPanel`). Confirmed
  with the user that a **single `ExamplesPanel.svelte` with two
  add modes** (`+ Manual` inline, `+ From dataset` as an inline
  picker) is the right shape for v1. Both modes share the same
  added-examples list — manual rows and dataset-sourced rows are
  identical once added. Simpler, fewer files, same UX.

## Files

### New: stores (`yadc/webui/src/lib/stores/prompts/`)
- `types.ts` — `ExamplePair`, `PromptGenFocus`, `PromptGenRequest`,
  `StreamEvent`, `GenerationStatus`.
- `api.ts` — `generatePrompt(request, callbacks, signal)` reads
  the response body via `ReadableStream.getReader()` and parses
  NDJSON line-by-line. The final-line-without-trailing-newline
  case is handled by flushing the decoder + buffer at EOF. Also
  `fetchImageAsDataUrl()` (browser fetch → FileReader) and
  `readFileAsDataUrl()` (local File → data URL) for the
  multimodal examples panel.
- `store.svelte.ts` — Svelte 5 `$state` rune module owning the
  streaming state (status / body / reasoning / error /
  AbortController) + state mutators
  (`beginGeneration`, `appendToken`, `appendReasoning`,
  `completeGeneration`, `cancelGenerationState`, `failGeneration`,
  `reset`). The AbortController is owned by the state so
  `cancelGenerationState()` can abort it cleanly.
- `actions.ts` — `startGeneration()` orchestrates the API call
  and wires streaming callbacks to the state mutators. Errors
  surface via toasts and the state.
- `api.test.ts` — Vitest tests (13) for the streaming reader:
  happy path, reasoning/reasoning_summary dispatch, mid-stream
  error, malformed lines, chunked streaming (split mid-JSON),
  no-trailing-newline EOF, HTTP errors, no-body, abort
  propagation, FileReader helper.
- `index.ts` — barrel.

### New: components (`yadc/webui/src/lib/components/prompts/`)
- `PromptGenerator.svelte` — host: two-column layout (form on
  left, streaming preview on right). Owns the form bindings +
  the Save-as-template dialog state.
- `PromptForm.svelte` — env selector (with auto model fetch via
  `withPasswordRetry`), model picker, intent textarea, focus
  toggle, examples section, Generate button. Mirrors
  `caption/EnvSelector` but bindable + reacts to streaming
  status (inputs disabled mid-stream).
- `ExamplesPanel.svelte` — the consolidated panel. Examples
  list (thumbnail + subject + caption + remove). `+ Manual` opens
  a file picker, reads each file via `readFileAsDataUrl()`, and
  appends a new row. `+ From dataset` opens an inline picker
  with a dataset selector + image grid; clicking a thumbnail
  auto-fills subject = `image.file_name` (sans extension) and
  caption = `fetchCaption()` result (best-effort; if the image
  has no caption yet, the field starts empty).
- `GenerationPreview.svelte` — streaming monospace preview with
  blinking caret, copy-to-clipboard, Save-as-template button,
  extracted Jinja variables as chips, reasoning collapsible.
  Renders the generation store state directly (no props).
- `index.ts` — barrel.

### New: route
- `yadc/webui/src/routes/prompts/+page.svelte` — Topbar with
  title + description, full-height host of `PromptGenerator`.

### Updated
- `yadc/webui/src/routes/+layout.svelte` — added "Prompts" nav
  item (with `SvgSparkle` icon) between Datasets and Templates.
- `yadc/webui/src/lib/components/templates/EditTemplateDialog.svelte`
  — added `initialContent?` prop to pre-fill the editor for new
  templates. Used by `PromptGenerator`'s "Save as template"
  button so the generated body becomes the new template's
  starting content.
- `yadc/webui/src/lib/icons/SvgSave.svelte` — new icon for the
  save button.

### Cleanup (per plan)
- Inlined the body of the now-removed
  `lib/components/ui/PromptPreview.svelte` into its only
  consumer, `lib/components/dataset/detail/Preview.svelte`
  (was a 14-line wrapper, now the host). Deleted
  `lib/components/ui/PromptPreview.svelte` to free the name for
  the new `prompts/GenerationPreview` (no collision).

## Design choices

- **Single examples panel, two modes** (per user). Simpler
  component tree, identical UX. The mode picker is just two
  buttons in the empty state and the list footer.
- **State controller in the store** (not a module-local ref in
  actions.ts). Cleaner — `cancelGenerationState` aborts the
  state-owned controller directly. No cross-module state
  plumbing.
- **NDJSON reader in production code, not a separate utility**:
  the reader is small and tied to this endpoint's specific
  wire format. Pulling it into `lib/` would be premature
  abstraction. It IS tested (13 cases) so the test pattern
  is reusable if other endpoints need streaming later.
- **`EditTemplateDialog.initialContent`** is the minimum-invasive
  way to thread the generated body into the save flow. The
  dialog already supports new-template mode; the new prop just
  overrides the default stub content.
- **`SvgSparkle` for the nav icon**: it's the only icon in
  the existing set that conveys "AI generation" / "magic", and
  it's already used by `Caption.svelte` (refine caption) so the
  visual vocabulary is consistent.

## Verification

- All 50 frontend tests pass (13 new + 37 pre-existing).
- All 726 backend tests still pass.
- `ruff check`, `ruff format --check`, `basedpyright` clean.
- `svelte-check` clean, `eslint` clean, `prettier --check`
  clean, `npm run build` succeeds.

## Open Questions status

- **Cancel propagation (end-to-end)**: Phase 2 verified the
  server-side path (Quart cancel → `CancelledError` →
  `client.aclose()` in the service's `finally`). Phase 3 adds
  the client-side path: `controller.abort()` on the
  `AbortController` passed to `fetch` cancels the request, and
  the `ReadableStream` reader stops yielding. Not exercised by
  tests yet (would need a fake long-running stream), but the
  mechanism is in place and the streaming abort test verifies
  the propagation.
