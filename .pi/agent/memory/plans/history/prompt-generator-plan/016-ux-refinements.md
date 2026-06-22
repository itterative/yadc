---
date: 2026-06-22
---
# Footer: drop status text, move cancel in from preview

UX refinement.

**`PromptSidePanel.svelte`** — footer: dropped the contextual
status `<p>` on the left (info was redundant with the tabs
themselves and crowded the footer on narrow widths). The
Generate/Refine primary CTA now swaps to a `btn-danger` Cancel
mid-stream, so the cancel is reachable from inside the panel on
both layouts — closing the gap from 015 where, on mobile with the
panel open during streaming, neither the FAB nor the preview
footer cancel was visible.

**`GenerationPreview.svelte`** — cancel button removed (now in the
panel footer). `cancelGenerationState` import dropped. Header's
`SvgClose` "Clear preview" stays (different action — `reset`).

The FAB cancel and the footer cancel both call
`cancelGenerationState`, so behaviour is the same regardless of
which the user reaches for.

**Files touched (so far):**
- `yadc/webui/src/lib/components/prompts/PromptSidePanel.svelte`
- `yadc/webui/src/lib/components/prompts/GenerationPreview.svelte`
- `docs/frontend/components-domain.md` (`PromptSidePanel` +
  `GenerationPreview` entries)

## History: per-entry restore icon (no more click-anywhere)

The whole entry was a `role="button"` that triggered restore on
click. On mobile that's both undiscoverable (no hover to reveal
the "Restore" label) and easy to fire by accident. Replaced with
an always-visible `SvgHistory` icon button (the section header
already uses `SvgHistory` — semantic match for "restore from
history"). The card is no longer a button: `role`/`tabindex`/
`onclick`/`onkeydown` removed, `cursor-pointer` and `focus:*`
dropped, no `disabled:*` on the entry itself. Restore shows a
spinner in place of the icon when this entry is restoring.
Delete moved out of its hover-only wrapper to always-visible too
— same discoverability reasoning (mobile has no hover, so the
delete was unreachable there).

**Files touched:**
- `yadc/webui/src/lib/components/prompts/PromptHistoryPanel.svelte`
- `docs/frontend/components-domain.md` (`PromptHistoryPanel`
  entry)

## Generate/Refine closes the panel

PromptSidePanel's Generate/Refine primary CTA now closes the
panel before calling `ongenerate`. Initially scoped to refine,
expanded to both modes (no good reason to leave the panel open
when the preview starts streaming). Done by wrapping the
callback in a local `handleGenerateClick` that sets
`open = false` first.

**Files touched:**
- `yadc/webui/src/lib/components/prompts/PromptSidePanel.svelte`

## GenerationPreview: inline status below the stream

The "Generating…" footer row moved inline, right under the `<pre>`,
where it actually describes what's being watched. Other terminal
states got the same treatment.

- `<pre>` wrapped in `{#if generation.body}` with a comment
  explaining why ("a bit ugly, but necessary to keep the
  whitespace of the template") — without the wrapper, an empty
  `<pre>` rendered awkwardly in some status transitions. Earlier
  tried `<p>` (no wrapper needed) but reverted to `<pre>` for
  semantic / a11y reasons (screen readers announce `<pre>` as
  preformatted text and respect whitespace).
- Caret uses `class:hidden={generation.status !== 'streaming'}`
  instead of `{#if}` — same result, more idiomatic.
- Inline status row (`mt-2 flex justify-center gap-1.5 text-sm`):
  - `streaming && !generation.body` → spinner + "Generating…"
    (only before the first token — once text streams in, the
    caret is the in-progress signal and a redundant label would
    compete with it).
  - `cancelled` → close + "Cancelled" (`text-warning` —
    user-initiated, so warning instead of error red).
  - **No "Done" indicator** — body being complete + Copy/Save
    buttons appearing in the header is enough.
- Error state unchanged — its full empty-state UI already
  displays text.
- Footer now variables-only; hidden entirely when
  `variables.length === 0`.

**Files touched:**
- `yadc/webui/src/lib/components/prompts/GenerationPreview.svelte`

## Streaming auto-scroll action

The "follow the stream + jump to bottom on open" logic that
lived in `ReasoningCard.svelte` (two `$effect` blocks + a
`scrollEl` ref + a `tick()` defer) is now a reusable Svelte
action in `lib/actions/autoscroll.ts` — same pattern as the
existing `autosize.ts`. Applied to both the reasoning card
`<pre>` and `GenerationPreview`'s body `<pre>` so the streamed
template follows the same auto-follow behaviour.

API:
```ts
use:autoscroll                              // defaults: scrollOnMount true
use:autoscroll={{ scrollOnMount: false }}   // don't jump on mount
```

Internals: a `MutationObserver` with `childList` + `characterData`
+ `subtree` options, so it catches both new child nodes and
text-content updates (which is what Svelte does when re-rendering
`{text}` bindings). On mount, a `requestAnimationFrame` callback
sets `scrollTop = scrollHeight` if `scrollOnMount` is true (defer
to next frame so the freshly-rendered DOM has its layout
computed).

Pause/resume: the original design used a position threshold
(`distFromBottom < 50` → auto-scroll), which meant small scroll
gestures within the threshold didn't disable it — users had to
scroll "aggressively" up to escape auto-follow. Replaced with
gesture-based detection: a `scroll` event listener compares
`scrollTop` against the previous value. Since our own
programmatic scroll only ever sets `scrollTop = scrollHeight`
(max position), any *decrease* in `scrollTop` between events is
unambiguously user-initiated (mouse wheel, touch drag, or
keyboard) and pauses auto-scroll. Resume happens only when the
user scrolls back to within 5px of the bottom.

Where to attach it: an earlier attempt put `use:autoscroll` on
`GenerationPreview`'s `<pre>`, but that didn't work because the
`<pre>` itself wasn't the scroll container (the body `<div>`
was, via `overflow-auto`) — `scrollTop` on the `<pre>` was a
no-op. Moved the action to the parent `<div>`.

The MutationObserver is disconnected in `destroy()`, so the
action is safe to attach to conditionally-rendered elements
(ReasoningCard's `<pre>` only exists when the card is expanded).

**Files touched:**
- `yadc/webui/src/lib/actions/autoscroll.ts` (new)
- `yadc/webui/src/lib/components/prompts/ReasoningCard.svelte`
  (dropped two `$effect`s + `scrollEl` ref + `tick` import;
  added `use:autoscroll`)
- `yadc/webui/src/lib/components/prompts/GenerationPreview.svelte`
  (added `use:autoscroll` on the body scroll container, not
  the `<pre>`)
- `.pi/agent/memory/frontend-architecture.md` (`lib/actions/`
  folder added to the current-layout one-liner, alongside
  `autosize.ts`)
- `.pi/agent/memory/docs/frontend/components-domain.md`
  (`ReasoningCard` + `GenerationPreview` entries mention
  `use:autoscroll`)

## Footer moved onto the Generate tab

The footer action bar (Save-to-history + Generate/Refine, or
Cancel while streaming) was a sibling of `PillTabs` at the panel
level, so it showed on all three tabs. Moved it inside the
Generate tab: that tab is now a flex column (`flex h-full
flex-col overflow-hidden`) with the form in a scrollable area
(`min-h-0 flex-1 overflow-y-auto p-4`) above a pinned footer.
The panel-level wrapper `<div class="flex h-full min-h-0
flex-col">` around `PillTabs` is now redundant (PillTabs is the
only child) and was dropped. The Settings / History tabs are
unchanged (self-scrolling `h-full overflow-y-auto`).

The footer is a Generate-tab concern — Save / Generate / Refine
all operate on the form state, which lives on that tab — so the
other tabs no longer carry an unrelated action bar.

Behavioural note: the in-panel Cancel now only renders on the
Generate tab. Since Generate is only clickable from that tab, a
stream always starts with `activeTab === 'generate'`, so
reopening the panel mid-stream still shows the Cancel; only a
manual switch to Settings / History mid-stream hides it (the FAB
Cancel is still reachable by closing the panel). Narrow enough
not to special-case.

**Files touched:**
- `yadc/webui/src/lib/components/prompts/PromptSidePanel.svelte`
- `docs/frontend/components-domain.md` (`PromptSidePanel` +
  `PromptGenerator` entries)

## Refine template: card with read-only editor + Edit dialog

The refine-mode Template section in `PromptForm.svelte` was a
bare `<select>` + a full-height editable `JinjaEditor` + a
variables strip. Wrapped the editor in a `Card` (mirrors
`caption/TemplateSection`) so the template reads like the rest
of the app's template surfaces.

The card is **never inline-editable** — all edits go through
`EditTemplateDialog`. The picker's empty option is `(custom)`;
states:

- **Existing template selected** → read-only editor + `ActionBar`
  "Edit" → `EditTemplateDialog` (edit mode) → saves back to the
  backend; `handleEditSaved` toggles `selectedTemplateName`
  ('' → name) to force the content-load effect to refetch.
- **`(custom)` with ephemeral content set** (from the dialog, or
  from a history restore) → read-only editor + "Edit" →
  `EditTemplateDialog` in **ephemeral mode** (seeded via
  `initialContent`); the content comes back via `onapply`, **not**
  persisted.
- **`(custom)` with nothing set** → a placeholder ("No template
  set. Refining will generate a new template instead of refining
  an existing one.") + "Edit" → ephemeral dialog (empty).

The single ActionBar button always says "Edit" (no "Create" /
"Generate-new" wording); its `onclick` opens the backend-edit
dialog when an existing template is picked, the ephemeral dialog
otherwise.

`EditTemplateDialog` gained an **additive** `ephemeral` mode
(`ephemeral?: boolean` + `onapply?: (content: string) => void`):
hides the name field, skips `saveTemplate`, rebrands the header
→ "Template Content" and Save → "Apply". None of the 5 existing
call sites are affected.

Picker → `(custom)` clears `templateContent` (the content-load
effect's `!name` branch now sets it to `''`) so stale content
from a previously-picked template can't masquerade as ephemeral.
Ephemeral edits set `templateContent` without touching
`selectedTemplateName`, so they survive (the effect doesn't
re-run).

### Backend + validation follow-ons

The backend already does the right thing: `_build_messages`
decides generate-vs-refine purely on `template_content is not
None` (no `mode` flag), and the request validator rejects
empty-string `template_content` (must be `None` or non-empty).
So refine mode with no template → the generate branch runs → a
new template is emitted. That made the placeholder wording
accurate, and meant two frontend changes:

- **`PromptSidePanel`**: `canGenerate` / `canSave` no longer
  require a template in refine mode — gated only by env + intent.
  `handleSaveToHistory` coalesces empty `templateContent` →
  `null`.
- **`stores/prompts/actions.ts`**: `template_content:
  args.templateContent ?? null` → `|| null` — `??` left an empty
  string, which the backend validator would 422 on.

**Files touched:**
- `yadc/webui/src/lib/components/prompts/PromptForm.svelte`
  (Card + ActionBar + placeholder; two `EditTemplateDialog`
  instances; `ephemeralDialogOpen`; picker clear-on-`!name`;
  variables-clear-on-empty effect — the `JinjaEditor` only mounts
  when there's content so nothing else cleared `templateVariables`
  when the template became empty)
- `yadc/webui/src/lib/components/templates/EditTemplateDialog.svelte`
  (`ephemeral` mode + `onapply`)
- `yadc/webui/src/lib/components/prompts/PromptSidePanel.svelte`
  (`canGenerate` / `canSave` relaxed; history template_content
  coalesced)
- `yadc/webui/src/lib/stores/prompts/actions.ts` (empty → null)
- `docs/frontend/components-domain.md` (`PromptForm`,
  `PromptSidePanel`, `EditTemplateDialog` entries)

## Custom template content preserved across picker switches

Refine's `(custom)` picker option had no backend identity, so
its content lived in the single bindable `templateContent`.
Switching to an existing template overwrote `templateContent`
with the loaded body; switching back to `(custom)` cleared it
(the `!name` branch set it to `''`). Result: any custom edits
were lost the moment you browsed another template.

A confirmation dialog before the overwrite was the alternative,
but that's friction for something the user shouldn't have to
justify keeping. Instead, added an internal
`customTemplateContent` state to `PromptForm` that holds the
custom draft independently of the picker:

- The load effect's `!name` branch now sets
  `templateContent = customTemplateContent` (restore) instead of
  `''` (clear). Empty custom draft → placeholder shows, as
  before.
- The ephemeral dialog's `onapply` writes both
  `customTemplateContent` and `templateContent`.
- `templateContent` (bound to host, snapshotted for generation /
  history) remains the *effective* content — what would be sent
  right now. `customTemplateContent` is purely a "remember this
  across picker switches" concern, so it stays internal (the
  host has no reason to own it — custom content isn't
  persisted).

Switching to an existing template doesn't touch
`customTemplateContent`, so round-tripping back to `(custom)`
recovers the draft.

**Files touched:**
- `yadc/webui/src/lib/components/prompts/PromptForm.svelte`
- `docs/frontend/components-domain.md` (`PromptForm` entry)

## History restore lands on (custom) + seeds the draft

`handleRestore` (in `PromptSidePanel`) set the bindable
`templateContent` from the entry but couldn't reach
`selectedTemplateName` or `customTemplateContent` (both internal
to `PromptForm`). Since history entries carry content, not a
template name, this left the picker selection stale relative to
the restored content: an existing template still picked meant
the card showed the restored body but Edit opened the backend
editor for the picked (wrong) template, and switching the picker
away-and-back replaced the restored content with the old
`customTemplateContent` draft (which was never seeded).

Fix: exposed `PromptForm.restoreTemplate(content)` (exported
method, reached via `bind:this` — same pattern as
`historyPanel?.refresh()` in the same file). It resets the
picker to `(custom)` and seeds both `customTemplateContent` and
`templateContent`, so restore always lands as a custom template
with the content preserved. `handleRestore` calls it instead of
setting `templateContent` directly. Generate-mode entries pass
`''` → harmless reset.

Restore overwrites the current custom draft (and the rest of
the form) by design — that's what "restore" means. Added a
small muted note above the history list ("Restoring replaces
your current form values.") so the overwrite isn't surprising,
rather than a per-restore confirmation (friction).

**Files touched:**
- `yadc/webui/src/lib/components/prompts/PromptForm.svelte`
  (`restoreTemplate` exported method)
- `yadc/webui/src/lib/components/prompts/PromptSidePanel.svelte`
  (`promptForm` ref + `bind:this`; `handleRestore` calls
  `restoreTemplate`)
- `yadc/webui/src/lib/components/prompts/PromptHistoryPanel.svelte`
  (restore-overwrites note above the list)
- `docs/frontend/components-domain.md` (`PromptHistoryPanel` entry)

## GenerationPreview: actions → bottom ActionBar, then chromeless panel

The terminal-state actions (Copy / Save as template / Clear) were
three small buttons crammed into the preview header next to the
title — poor tap targets on mobile, and Save (the primary
action) was demoted to `text-xs`. Clear used `SvgClose` (an ✕),
which read as "close panel" rather than "clear preview".

First pass moved them to a bottom `ActionBar` reusing the
`Card` + `ActionBar` / `ActionBarItem` primitives (same pattern
as `caption/TemplateSection` and the `Caption` drafts). But the
`pb-24 lg:pb-0` that lifts the bar above the mobile FAB was put
on the panel's own `card` wrapper — so the reserved space read
as a large framed gap inside the bordered panel rather than
open space below it.

Refined to a chromeless panel: dropped the title **and** the
outer `card` (bg + border + rounded + `overflow-hidden`) and
the body's `bg-gray-900/40` tint — the streamed text is now
full-bleed, and the **footer is the only framed element**. The
footer is a `Card` (shown on terminal state) that stacks the
variables chip strip (hidden when none) above the `ActionBar`:

- **Save as template** (`SvgSave`, `primary`, disabled unless
  `status === 'done'`)
- **Copy** (`SvgCopy`, `secondary`; label flips to "Copied" for
  1.5s)
- **Clear** (`SvgDelete`, `danger` — replaces `SvgClose`)

The `pb-24 lg:pb-0` moved onto the now-chromeless outer flex
container, so on mobile the footer card (and the streaming
body's tail) sits above the side-panel FAB's `bottom-6` + `h-16`
band with open page bg below it (not a framed gap); no padding
on desktop (`lg:pb-0`, FAB is `lg:hidden`). During streaming
there's no footer, so the body fills down to the padding edge —
the caret stays visible above the FAB.

**Files touched:**
- `yadc/webui/src/lib/components/prompts/GenerationPreview.svelte`
- `docs/frontend/components-domain.md` (`GenerationPreview` entry)

## Mobile: area-level scroll (no inner "box" scroll)

With the panel chromeless, the inner `overflow-auto` on the body
felt like a disconnected scroll box on mobile. Switched to a
responsive scroll model — on mobile the OUTER container scrolls
(block flow, body + footer together as one content area); on
desktop (`lg:`) the body is the inner scroll container and the
footer is pinned below (flex column), as before.

Classes: outer `h-full overflow-y-auto pb-24 lg:flex lg:min-h-0
lg:flex-col lg:overflow-visible lg:pb-0`; body `p-4 lg:min-h-0
lg:flex-1 lg:overflow-auto`; footer `Card class="mx-4 lg:mx-0"`
(aligns with the body's `p-4` on mobile, full-width pinned on
desktop). `pb-24 lg:pb-0` unchanged (FAB clearance on mobile).
The change is localized to `GenerationPreview` — the
`h-[calc(100dvh-7.5rem)]` page box and `PromptGenerator` layout
are untouched (the scroll stays inside that box, so the topbar
stays fixed).

### `autoscroll` resolves its scroll target

For this to work, `autoscroll` had to scroll whatever element
actually scrolls `node`'s content, not always `node` itself.
The action now resolves its target: `node` if it's a scroll
container, else the nearest scrollable ancestor, else the page
(`document.scrollingElement`). It reads/writes `target.scrollTop`
and attaches the `scroll` listener to the target (or `window`
when the target is the page, since `scroll` doesn't bubble).
Re-resolves on viewport `resize`, because a responsive layout
can move which element scrolls (inner container desktop ↔
content area mobile); the listener follows if the target
changed.

`ReasoningCard`'s `<pre>` is itself a scroll container (`overflow-
y-auto`) → the action resolves to self there → behavior
unchanged. The only other user is the preview body, which now
drives the outer on mobile and itself on desktop.

Small fixup: the idle/error empty-state wrappers changed from
`h-full` to `min-h-[50vh] lg:h-full` — their vertical centering
relied on a definite parent height, which the mobile flow (no
`flex-1`) doesn't provide.

**Files touched:**
- `yadc/webui/src/lib/actions/autoscroll.ts` (scroll-target
  resolution + resize re-resolve)
- `yadc/webui/src/lib/components/prompts/GenerationPreview.svelte`
  (responsive scroll model + empty-state `min-h` + footer `mx-4`)
- `.pi/agent/memory/frontend-architecture.md` (`autoscroll`
  description)
- `docs/frontend/components-domain.md` (`GenerationPreview` entry)
