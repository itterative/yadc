---
date: 2026-06-02
---
# Prompt Preview Now Uses captionOptions Directly

**Context:** `PromptPreview.svelte` (in the image details tab) had its own template dropdown and a "Render" button. This duplicated the template list that `CaptionSettings.svelte` (caption tab) already maintains, and the preview was disconnected from the live settings — editing a template in the caption tab had no effect on the preview until the user re-selected the template in the details tab and clicked Render.

**Decision:** Removed the template dropdown and Render button. The preview now subscribes to the existing `captionOptions` store (already kept in sync by `CaptionSettings` via `$effect(() => captionOptionsStore.set(_assembledOptions))`). The effect re-renders whenever the image changes or the template name/raw content changes:

- Image change → clear preview, re-render
- Template change → keep old preview visible, debounce 200 ms, then re-render
- `captionOptions.prompt_template` (raw content) takes priority over `captionOptions.prompt_name` (so unsaved template edits show up immediately)
- Neither set → API uses the dataset's configured default (existing behavior of `POST /preview-prompt` with empty body)

A "Rendering…" indicator shows next to the collapse toggle while the API call is in flight. Race protection via the standard `cancelled` flag in the effect cleanup + `clearTimeout` for the debounce — the previous response is discarded if the image/template changes mid-fetch.

**Rationale:** The "preview" should reflect what would actually be sent to the model. Anything else is misleading. Now there's one source of truth for the template (`captionOptions`), the preview updates in real time as the user edits settings in the caption tab, and the UI loses a control that was just maintaining a parallel copy of state.

**Tradeoffs:**

- 200 ms debounce is a small but visible delay between stopping typing and the preview updating. 200 ms was chosen as a good middle ground — short enough to feel snappy, long enough to avoid hammering the API on every keystroke.
- If the user has never opened the caption tab, `captionOptions` is `{}` and the preview falls back to the dataset's default template (same as the old "default" option in the dropdown).

**Files touched:**

- `yadc/webui/src/lib/components/ui/PromptPreview.svelte` — full rewrite of the data flow; template `<select>`, "Render" button, and the `previewTemplateName` state removed; new subscription to `captionOptions`; image-change-only clear via `lastRenderedImageId` tracker; 200 ms debounce.
