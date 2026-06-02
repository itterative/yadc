---
name: codemirror-quirks
description: CodeMirror 6 editor sizing quirks in the yadc webui — the CSS percentage-height trap, the flex circular dependency, the `minmax(0, 1fr)` grid pattern that breaks it, and the absolute-positioning fallback kept in CodeMirror.svelte.
category: architecture
---

# CodeMirror 6 Editor Quirks

Reference for the gotchas around sizing the CodeMirror-based editors (`TomlEditor`, `JinjaEditor`) in `yadc/webui/src/lib/components/ui/CodeMirror.svelte`. These are non-obvious CSS/flex/grid interactions that took several iterations to get right.

## TL;DR

The editor is a **passive filler** — the wrapper gives it a box, the editor fills it. There are three working patterns:

1. **Grid with `minmax(0, 1fr)`** — the recommended pattern for dialogs and side panels. `minmax(0, 1fr)` is the grid equivalent of `flex: 1 1 0%` and breaks the circular dependency that plain `1fr` (or flex `flex-1` on the editor) would cause.
2. **Definite heights** — for simple cases where you know the exact height (e.g. `h-64`, `h-[50vh]`).
3. **Absolute positioning in `CodeMirror.svelte`** — kept as a safety net. The editor is `position: absolute; inset: 0` and fills its wrapper regardless of the wrapper's own size.

The editor cannot determine its own size in non-`autoHeight` mode. The one exception is `autoHeight: true`, where the editor grows with content.

## CodeMirror 6 DOM structure

```
.cm-editor                  ← flex container (display: flex, flex-direction: column)
├── .cm-gutters            ← line number gutter (left)
├── .cm-scroller           ← scrollable area (flex: 1 1 0 by default; overflow: auto)
│   └── .cm-content        ← text container (where lines live; where clicks are handled)
│       └── .cm-line       ← individual lines
```

Clicks are handled by a listener attached **only to `.cm-content`**. If a click lands on `.cm-scroller` outside `.cm-content` (i.e. in the empty padding at the bottom of a short document), the cursor does not appear. This is why `.cm-content` needs `min-height: 100%` of the scroller — so the empty area is part of `.cm-content` and the click handler fires.

## The CSS percentage-height trap

`height: 100%` (and `min-height: 100%`) on a child only resolves when the **parent has a *definite* `height`**. A parent with only `min-height: X` does *not* count as definite — `min-height` is a floor, not a height, so child percentage heights resolve to `auto` / 0.

This is a CSS spec issue, not a browser bug. Concretely, given:

```html
<div class="min-h-64">  <!-- parent: min-height: 16rem, height: auto -->
    <div class="h-full"> <!-- child: height: 100% — does NOT resolve -->
        <div class="h-full"> <!-- grandchild: same problem -->
        </div>
    </div>
</div>
```

…the inner divs are 0 tall, even though the parent is 16rem.

`h-full` (i.e. `height: 100%`) only works when **every ancestor up the chain** resolves its `h-full` to a definite height. If any link in the chain is `auto` (even with a `min-height` floor), the chain breaks and all descendants with `h-full` collapse.

## The flex circular dependency

The natural-looking fix for the above is flex:

```html
<div class="flex h-full flex-col min-h-64">     <!-- wrapper -->
    <div class="cm-editor flex-1">              <!-- .cm-editor with flex: 1 1 0% -->
    </div>
</div>
```

This works for some cases. It breaks for others. The failure mode:

1. `.cm-editor` has `flex: 1 1 0%` → `flex-basis: 0%` → intrinsic size 0
2. Wrapper's content size is 0 (driven by `.cm-editor`)
3. If the wrapper is itself a regular block child (not a flex item in a flex parent), the wrapper is `auto` and collapses
4. Even when the wrapper is a flex item in a flex parent, if the parent's `h-full` chain breaks anywhere, the available space is 0 and the wrapper grows to 0

The circular dependency: wrapper's size depends on `.cm-editor`'s content, and `.cm-editor`'s size depends on the wrapper.

This bit the **template editor dialog** (`routes/templates/EditTemplateDialog.svelte`) and the **Advanced tab in the dataset config** (`lib/components/datasets/DatasetConfigAdvanced.svelte`). In both, the editor's parent is a flex item (the dialog / the Tab content), and the wrapper is a regular block child of that parent — `h-full` on the wrapper couldn't resolve and the flex circular dependency collapsed the editor to 0.

## Why plain `1fr` has the same problem as flex

The natural-looking grid fix is `grid-template-rows: 1fr`. But `1fr` is shorthand for `minmax(auto, 1fr)` — the minimum is `auto` (the content's intrinsic minimum size), not 0. So a `1fr` track won't shrink below the content's intrinsic size, and the circular dependency forms just like with flex.

The fix is **`minmax(0, 1fr)`**, which sets the minimum to 0 — the grid equivalent of `flex: 1 1 0%`. With `minmax(0, 1fr)`, the track can shrink to 0, breaking the circular dependency.

## Working solutions

### 1. Grid with `minmax(0, 1fr)` — recommended for dialogs and side panels

The pattern that works for both the dataset edit dialog and the image detail side panel:

```html
<!-- Dialog: definite height on the dialog itself -->
<Dialog class="dialog-panel h-[55vh] max-w-2xl">
    <div class="grid grid-rows-[min-content_minmax(0,1fr)] h-full p-5">
        <div class="shrink-0 dialog-header">...</div>
        <div>
            <TomlEditor ... />
        </div>
    </div>
</Dialog>
```

```html
<!-- Side panel: min-h-full + h-fit on the root, grid for the rows -->
<div class="grid grid-cols-1 grid-rows-[min-content_minmax(0,1fr)] min-h-full h-fit p-4">
    <div>header (min-content)</div>
    <div>tab area (minmax(0,1fr))</div>
</div>
```

The recipe:
- The grid container has a definite or near-definite height (`h-[55vh]`, `min-h-full h-fit`).
- The first row is `min-content` (the header, sized to its content).
- The second row is `minmax(0, 1fr)` (grows to fill, but can shrink to 0).
- The editor is a child of the second row, typically with `flex-1` to grow within the row.

This pattern was adopted because the side panel and the dialog needed to work with the same height-handling approach — flex was giving issues on the dialog (the dialog's `h-full` chain wasn't resolving reliably), but grid with `minmax(0, 1fr)` worked in both cases.

### 2. Definite heights — simple cases

For simpler cases where the editor area has a known height, just use a definite height on the wrapper:

```html
<div class="relative h-64">
    <JinjaEditor class="rounded-md border border-border text-sm h-full" ... />
</div>
```

```html
<div class="h-[50vh] flex-1 overflow-y-auto">
    <JinjaEditor class="rounded-md border border-border" ... />
</div>
```

This works because the wrapper has a definite height, so the editor's `h-full` resolves correctly.

### 3. Absolute positioning in `CodeMirror.svelte` — safety net

The base theme in `CodeMirror.svelte` sets `.cm-editor` to `position: absolute; inset: 0`. This is a safety net — the editor fills its wrapper regardless of the wrapper's own size. The wrapper is:

```html
<div class="relative h-full min-h-32 flex-1 overflow-hidden {klazz}"></div>
```

This pattern is no longer the primary solution (consumers now provide their own height context), but it's kept because:
- It handles the case where the consumer doesn't provide any height context (e.g. a regular block parent with no height, like `PromptPreview`'s `<div class="mt-1">` inside a `<details>`).
- It makes the editor a passive filler, which is conceptually simpler than tracking the height chain in every consumer.

The `min-h-32` floor ensures the editor is always at least 8rem tall. Consumers that want a different floor can override via the `class` prop (e.g. `class="min-h-96"`).

**Background is intentionally not set on `.cm-editor` / `.cm-gutters` in the base theme.** Consumers set the background on the wrapper (e.g. `bg-surface` for the standard dark editors, `bg-gray-800` for the image-detail Extras tab to match the caption box), and the editor paints transparently over it. The gutter's `borderRight` is the only visual separator between line numbers and text.

## autoHeight mode

In `autoHeight: true` mode the editor grows with content. The absolute positioning is wrong for that case (would make the editor 0×0), so the autoHeight theme overrides:

```css
&.cm-editor {
    position: static !important;
    height: auto !important;
}
```

And the wrapper drops all the layout classes:

```html
<div class="overflow-hidden">  <!-- autoHeight wrapper -->
```

## File touchpoints

- `yadc/webui/src/lib/components/ui/CodeMirror.svelte` — the wrapper, the base theme (with absolute positioning), and the autoHeight theme
- `yadc/webui/src/lib/components/ui/TomlEditor.svelte` — TOML-specific wrapper (adds TOML language, merge-view support)
- `yadc/webui/src/lib/components/ui/JinjaEditor.svelte` — Jinja-specific wrapper (adds `jinja()` language, variable extraction)

## All current usages (verified working)

| File | Parent | Editor wrapper / class | How it's sized |
|------|--------|------------------------|----------------|
| `ImageDetail/Extras.svelte` | `ImageDetail.svelte` grid track (`minmax(0, 1fr)`) | wrapper `relative flex-1 p-2 box-content`; class `min-h-32 rounded-md border-0 bg-gray-800 text-sm` | wrapper grows in the grid track via `flex-1`; `bg-gray-800` matches the caption box |
| `ConfigHistory.svelte` | `autoHeight` | n/a | `position: static` override |
| `DatasetConfigAdvanced.svelte` | Tab content (`h-full overflow-y-auto`) | `rounded-md border border-border bg-surface text-sm` | wrapper falls back to `min-h-32` floor (absolute positioning kicks in) |
| `DatasetConfigForm.svelte` | `div.max-h-[40vh].min-h-30.overflow-y-auto` | `rounded-md border border-border bg-surface text-sm` | wrapper has `h-full` → 100% of div (≥7.5rem) |
| `settings/TemplateManager.svelte` | `div.min-h-0.flex-1.overflow-hidden` | `rounded-md border border-border bg-surface` | wrapper has `h-full` → 100% of div |
| `ui/PromptPreview.svelte` | `div.mt-1` (inside `<details>`) | `rounded-md border border-border bg-surface text-sm` | wrapper falls back to `min-h-32` (8rem) — absolute positioning |
| `routes/datasets/[name]/CaptionSettings.svelte` | `div.relative h-64` (definite height 16rem) | `rounded-md border border-border bg-surface text-sm h-full` | wrapper has `h-full` → 16rem |
| `routes/templates/EditTemplateDialog.svelte` | `div.h-[50vh].flex-1.overflow-y-auto` | `rounded-md border border-border bg-surface` | wrapper has `h-full` → 50vh |

The EditDatasetDialog (`routes/EditDatasetDialog.svelte`) doesn't directly use the editors but hosts the `DatasetConfig` component inside a grid with `minmax(0, 1fr)` — so editors inside the dialog benefit from the same grid pattern.

## How to extend the wrapper class from a consumer

A consumer can append Tailwind classes to the `TomlEditor` / `JinjaEditor` via the `class` prop. These are appended to the wrapper:

```svelte
<TomlEditor class="min-h-64 rounded-md border border-border text-sm" ... />
```

The consumer's classes are appended *after* the base wrapper classes in the `class` attribute, so a consumer's `min-h-64` wins over the base `min-h-32` (CSS cascade — both have the same specificity, last one wins).

## Choosing a pattern

| Situation | Use |
|-----------|-----|
| Dialog or side panel with a header + flexible content area | Grid with `minmax(0, 1fr)` on the content row |
| Editor area with a known fixed height | Definite height on the wrapper (`h-64`, `h-[50vh]`, etc.) |
| Editor inside a scrollable container with a max height | Definite height on the container, let the editor fill it |
| No height context at all (regular block, no min-h) | Absolute positioning kicks in via the `min-h-32` floor — editor is 8rem |
| Editor should grow with content | `autoHeight: true` |
