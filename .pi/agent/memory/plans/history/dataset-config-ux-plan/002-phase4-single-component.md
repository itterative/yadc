---
date: 2025-05-30
---
# Phase 4 Implementation: Single Shared Component

**Context:** Phase 4 (shared field components) was implemented as a single `CaptionOptionsFields.svelte` component that renders both the Options and Reasoning sections together, rather than splitting into per-section components.

**Decision:** Use one monolithic component instead of per-section components (`<OptionsFields>`, `<ReasoningFields>`). The display config is passed as a single `display` prop object rather than via Svelte context.

**Rationale:** Both consumers use Options + Reasoning together in the same order with the same fields. A single component avoids the overhead of coordinating multiple section components while still eliminating the duplication. The `display` prop groups read-only config (nullable, helpText, idPrefix, diffDefaults, visibility flags) cleanly without needing context since there's only one level of nesting.

**Trade-off:** If sections are later added independently (e.g. DatasetConfig gets Reasoning but not Options, or new sections are introduced), splitting into per-section components with a context would be cleaner. The current approach is pragmatic for the current two consumers.

**Files touched:**
- NEW: `yadc/webui/src/lib/components/settings/CaptionOptionsFields.svelte` (335 lines)
- `yadc/webui/src/routes/datasets/[name]/CaptionSettings.svelte` (772→653, -119 lines)
- `yadc/webui/src/routes/datasets/[name]/DatasetConfig.svelte` (579→431, -148 lines)
