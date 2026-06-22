---
name: frontend/components-ui
description: Frontend lib/components/ui/ — atomic, reusable primitives with no domain logic.
category: architecture
---

# Frontend: `lib/components/ui/`

Atomic, reusable UI primitives. No domain concepts, no API calls. The "scope rule" for component placement is in `frontend-architecture`.

## See also (in `.pi/agent/memory/docs/`)

- `frontend-patterns` — Tab system (context, registration, variants), Topbar pattern, Z-index layers (toasts above dialogs)
- `codemirror-quirks` — CodeMirror sizing pitfalls

```
yadc/webui/src/lib/components/ui/
  Dialog.svelte                # Modal dialog (HTML <dialog>)
  Alert.svelte                 # Inline alert (info/warning/error/success, dismissable, optional actions snippet)
  FileDropZone.svelte          # Drag-and-drop + file/folder picker with client-side validation, recursive directory traversal, and `allowedExtensions` filtering
  Checkbox.svelte              # Checkbox component
  CodeMirror.svelte            # CodeMirror 6 wrapper (Svelte 5 runes, doc/ext sync)
  JinjaEditor.svelte           # Jinja2 template editor (CM6 + @codemirror/lang-jinja). Bindable `value` + `variables` (extracted template vars)
  TomlEditor.svelte            # TOML editor (CM6 + @codemirror/legacy-modes, optional readonly mode). Bindable `value`
  KeyValueEditor.svelte        # Key-value pair editor (e.g. for `extras` TOML fields)
  EmptyState.svelte            # Reusable empty-state placeholder (icon + message + optional action)
  IntersectionObserverElement.svelte  # Infinite scroll sentinel
  tabs/                        # Tab system (Svelte 5 context + snippets)
    Tabs.svelte                # Generic container — tab bar layout, registration, bindable value
    PillTabs.svelte            # Pre-styled pill variant (wraps Tabs with rounded-full buttons)
    CompactPillTabs.svelte     # Compact segmented control variant (bg-gray-800/50 container, rounded-md buttons, icon support)
    Tab.svelte                 # Child that auto-registers via context, shows/hides content. Supports optional icon prop.
    TabsContext.svelte.ts      # Symbol key + state factory + typed helpers. TabItem has optional icon (Component).
  ConfirmDialog.svelte         # Global confirmation dialog (Promise-based, mounted in layout, supports string + snippet body, variant: danger/warning/info)
  SpinnerBlock.svelte          # Centered spinner with optional label and size
  PromptPreview.svelte         # Self-contained prompt preview (template selector + system/user prompt display)
  Tooltip.svelte               # Pure-CSS hover tooltip (wraps a trigger, shows label to the right on hover)
  Topbar.svelte                # Sets the layout topbar snippet from a page component (lifecycle-managed via $effect)
  SidePanel.svelte             # Generic slide-over side panel (mobile drawer + desktop static, FAB, backdrop, bindable `open`, configurable `class` for width). Hosts any `children` snippet (typically a `PillTabs` host). Outside-click-to-close via a click handler on the backdrop itself (NOT a `<svelte:window>` listener) — the backdrop is `fixed inset-0` covering the entire mobile viewport, so any click not landing on the panel hits it; the Svelte 5 synchronous-flush footgun doesn't apply because the click target is the always-present backdrop. The FAB is a `FabButton` rendered inside a positioning wrapper; the default is the panel toggle, overridable via the optional `fab` snippet (e.g. PromptSidePanel swaps in a streaming cancel). The X close button is intentionally NOT in the generic — callers put their own inside their content and mutate `open` via `bind:open`.
  ToastContainer.svelte        # Fixed-position toast stack (mounted in +layout.svelte)
  ToastItem.svelte             # Single toast (message, variant, progress bar, dismiss, optional action button)
  Card.svelte                  # Generic card wrapper (`rounded-lg bg-gray-800`). Body and footer (typically `ActionBar`) go in `children`.
  ActionBar.svelte             # Footer flex container (`flex gap-2 bg-black/15`) for card action buttons. Items distribute evenly via `flex-1`.
  ActionBarItem.svelte         # Standardized action button inside ActionBar — accepts icon, variant (`primary`/`secondary`/`danger`).
  FabButton.svelte             # Floating action button — the FAB shape used by SidePanel (`h-16 w-16` rounded-full, shadow, accent/error variant, `h-6 w-6` icon). Icon passed as a `Component`, like ActionBarItem.
```
