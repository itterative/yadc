---
name: frontend/components-ui
description: Frontend lib/components/ui/ — atomic, reusable primitives with no domain logic.
category: architecture
---

# Frontend: `lib/components/ui/`

Atomic, reusable UI primitives. No domain concepts, no API calls. The "scope rule" for component placement is in `frontend-architecture`.

```
yadc/webui/src/lib/components/ui/
  Dialog.svelte                # Modal dialog (HTML <dialog>)
  Alert.svelte                 # Inline alert (info/warning/error/success, dismissable, optional actions snippet)
  FileDropZone.svelte          # Drag-and-drop + file/folder picker with client-side validation, recursive directory traversal, and `allowedExtensions` filtering
  Checkbox.svelte              # Checkbox component
  CodeMirror.svelte            # CodeMirror 6 wrapper (Svelte 5 runes, doc/ext sync)
  JinjaEditor.svelte           # Jinja2 template editor (CM6 + @codemirror/lang-jinja). Bindable `value` + `variables` (extracted template vars)
  TomlEditor.svelte            # TOML editor (CM6 + @codemirror/legacy-modes, optional readonly mode). Bindable `value`
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
  SetTopbar.svelte             # Sets the layout topbar snippet from a page component (lifecycle-managed via $effect)
  ToastContainer.svelte        # Fixed-position toast stack (mounted in +layout.svelte)
  ToastItem.svelte             # Single toast (message, variant, progress bar, dismiss, optional action button)
  Card.svelte                  # Generic card wrapper (`rounded-lg bg-gray-800`). Body and footer (typically `ActionBar`) go in `children`.
  ActionBar.svelte             # Footer flex container (`flex gap-2 bg-black/15`) for card action buttons. Items distribute evenly via `flex-1`.
  ActionBarItem.svelte         # Standardized action button inside ActionBar — accepts icon, variant (`primary`/`secondary`/`danger`).
```

**Cross-references:**
- Tab system (context, registration, variants): `frontend-patterns` → Tabs System
- CodeMirror sizing pitfalls: `codemirror-quirks`
- Topbar pattern: `frontend-patterns`
- Z-index layers (toasts above dialogs): `frontend-patterns` → Z-Index Layers
