---
date: 2025-05-30
---
# Phase 3 Refactor: CompactPillTabs Component + Icon Support

**Context:** The view mode toggle was initially built inline in DatasetConfig. Extracted into a reusable `CompactPillTabs` component built on the existing Tabs/Tab infrastructure, with icon support added to the tab system.

**Changes to tab infrastructure:**
- `TabsContext.svelte.ts` — `TabItem` now has optional `icon?: Component<{ class?: string }>`. `registerTab()` accepts icon parameter.
- `Tab.svelte` — new `icon` prop, passed to `registerTab()`.
- `Tabs.svelte` — unchanged (already passes full `TabItem` to snippet).
- `PillTabs.svelte` — unchanged (doesn't render icons, but could be updated later).

**New component: `CompactPillTabs.svelte`**
- Segmented control style: `bg-gray-800/50` container, `rounded-md` buttons, small text
- Uses `TabsContext` directly (not wrapping `Tabs`) — avoids the border-bottom header bar
- Renders icons via `{@const Icon = t.icon}; <Icon class="h-3.5 w-3.5" />`
- `mt-2` spacing between header and content area
- Same `bind:value` and `Tab` child API as `PillTabs`

**DatasetConfig.svelte changes:**
- Removed inline toggle buttons, `ViewMode` type, `switchToAdvanced/Simplified/handleViewModeChange` functions
- Uses `CompactPillTabs` with `Tab` children (`id="simplified"` / `id="advanced"`) with icons
- Mode switching via `$effect` watching `activeView` vs `previousView`
- `viewMode` renamed to `activeView` (bound to CompactPillTabs value)

**Files:**
- `yadc/webui/src/lib/components/ui/tabs/TabsContext.svelte.ts` — icon support
- `yadc/webui/src/lib/components/ui/tabs/Tab.svelte` — icon prop
- NEW: `yadc/webui/src/lib/components/ui/tabs/CompactPillTabs.svelte`
- `yadc/webui/src/routes/datasets/[name]/DatasetConfig.svelte` — uses CompactPillTabs

**Checks:** eslint, prettier, svelte-check, build, pytest (168 passed) — all clean.
