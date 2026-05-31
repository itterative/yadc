---
date: 2026-05-30
---
# Typed Extras, History Restore Fix, and View Sync Preservation

**Context:** User reported two edge cases while testing the config editor: (1) restoring a history snapshot didn't refresh the form fields, requiring a manual refresh; (2) switching Advanced → Form → Advanced discarded unsaved advanced edits.

**Decision:**

1. **Typed KeyValueEditor**: Implemented type detection and per-row type selection (`string`/`number`/`boolean`/`object`) for dataset extras. Auto-detects types from parsed config values. Objects render as read-only JSON previews.
2. **History restore auto-refresh**: `handleConfigSaved()` now calls `loadConfig()` after incrementing `configVersion`, so the form repopulates from server state immediately after a history restore.
3. **View sync preservation**: Removed the `rawDirty → loadConfig()` path when switching Advanced → Form. The `previousView` effect now only syncs `previewContent` into the advanced editor when `rawDirty` is false. This prevents data loss but introduces a known divergence issue.

**Rationale:**
- Typed extras were already in the plan (Phase 1) and the user explicitly requested them with a key-adjacent type dropdown.
- History restore without refresh was a plain bug — the `onsaved` callback only bumped a counter.
- The view sync issue is a tradeoff: preserving drafts prevents data loss but means Form and Advanced can diverge silently. The old behavior (always reloading) caused data loss, which is worse UX.

**Known issue (deferred):** Form/Advanced view sync divergence. When both tabs have unsaved edits, they don't see each other's changes. Added as Phase 7.6 TODO in the plan. Possible future fixes: sync notices with "Apply/Kee" buttons, re-parse raw TOML into structured fields on Advanced→Form switch, or unify drafts into a single source of truth.

**Files touched:**
- `yadc/webui/src/lib/components/ui/KeyValueEditor.svelte` — rewritten with typed value support
- `yadc/webui/src/routes/datasets/[name]/DatasetConfig.svelte` — `detectType()`, typed `extrasToEntries()`, `entriesEqual()` uses JSON.stringify + type compare, `buildPatch()` passes typed values through, `handleConfigSaved()` reloads config, view-switch effect preserves drafts
