---
date: 2026-05-30
---
# History Tab Layout Alternatives

**Context:** The History tab was implemented as a two-pane layout (`w-40` sidebar + TOML preview) inside the ~480px side panel. The user felt it was cramped and asked for layout suggestions.

The panel is ~480px max on desktop and 80vw on mobile (often 300–400px). The current two-pane split gives the preview ~300px, which is just enough for TOML but the list feels squeezed.

---

## Alternative 1: Dense Sidebar (evolution of current)

Keep the two-pane split but compress the sidebar aggressively.

```
┌─ Revision History ───────────────────────────┬─┐
│                                              │↻│
├──────────────────────────────────────────────┴─┤
│ ┌────────┐  ┌────────────────────────────────┐ │
│ │Current │  │  api.url = "http://..."        │ │
│ │▓ Active│  │  model_name = "gemma3"         │ │
│ │        │  │  [settings]                    │ │
│ │2h ago  │  │  max_tokens = 1024             │ │
│ │5h ago  │  │                                │ │
│ │1d ago  │  │                                │ │
│ └────────┘  │              [Restore version] │ │
│             └────────────────────────────────┘ │
```

- **Sidebar:** `w-28`–`w-32` (~110–130px). Single-line relative time (`2h ago`), absolute timestamp on hover via `Tooltip`. Remove the second line of text per entry.
- **List density:** `py-1.5` padding, `text-xs`. Group by day with a sticky divider if needed.
- **Preview:** Gets ~330–350px. Still tight, but workable.

| Pros | Cons |
|------|------|
| Smallest code change | Still cramped for wide TOML lines |
| List + preview visible at once | Sidebar steals space on mobile |
| No new interaction patterns | |

---

## Alternative 2: Full-Width Accordion

Drop the sidebar entirely. Entries stack as full-width rows; click to expand and show the TOML preview inline.

```
┌─ Revision History ─────────────────────────────┐
│                                              ↻ │
├────────────────────────────────────────────────┤
│ ▓ Current (active)                          ▼  │
│ ┌────────────────────────────────────────────┐ │
│ │  api.url = "http://..."                    │ │
│ │  model_name = "gemma3"                     │ │
│ └────────────────────────────────────────────┘ │
│ ─────────────────────────────────────────────  │
│ ▶ 2h ago                                   …  │
│ ▶ 5h ago                                   …  │
│ ▶ 1d ago                                   …  │
│ ▶ Initial                                  …  │
└────────────────────────────────────────────────┘
```

- **Collapsed row:** relative time left, absolute time or "Restore" action right.
- **Expanded row:** read-only `TomlEditor` with `max-h-64` + Restore button at the bottom.
- **Behavior:** only one expanded at a time (auto-collapse others).

| Pros | Cons |
|------|------|
| Full ~440px for TOML preview | Scroll position jumps when expanding |
| Clean, no width split | Can't skim list while reading preview |
| Matches `ImageDetail` history pattern | |
| Works great on mobile | |

---

## Alternative 3: Compact List + Inline Drawer

A thin strip on the left acts as a scrubber; clicking an entry slides a preview drawer over it from the right.

```
┌─ Revision History ─────────────────────────────┐
│                                              ↻ │
├────────────────────────────────────────────────┤
│ ┌────┐ ┌─────────────────────────────────────┐ │
│ │Curr│ │ Snapshot from 2h ago                │ │
│ │▓   │ │                                     │ │
│ │2h  │ │ api.url = "http://..."              │ │
│ │5h  │ │ model_name = "gemma3"               │ │
│ │1d  │ │                                     │ │
│ │Init│ │                          [Restore]  │ │
│ └────┘ └─────────────────────────────────────┘ │
└────────────────────────────────────────────────┘
           ▲───────────────────────▲
           │   preview drawer      │
           │   (slides over list)  │
```

- **Strip:** `w-16`–`w-20`. Just relative time, vertical. Could be dots with tooltips.
- **Drawer:** absolute-positioned, `inset-y-0 right-0`, `w-[calc(100%-5rem)]`, slides in with a transition. Backdrop click or a "←" button closes it.

| Pros | Cons |
|------|------|
| Preview gets almost full width | Extra click to go back to list |
| List stays permanently visible | More complex positioning inside tab |
| Feels like a native drill-down | Drawer inside a drawer on mobile is weird |

---

## Alternative 4: Modal Preview

Full-width compact list. Clicking an entry opens the existing `Dialog` component with the preview and restore action.

```
┌─ Revision History ─────────────────────────────┐
│                                              ↻ │
├────────────────────────────────────────────────┤
│ Current (active)                    [Preview]  │
│ ─────────────────────────────────────────────  │
│ 2h ago                              [Preview]  │
│ 5h ago                              [Preview]  │
│ 1d ago                              [Preview]  │
│ Initial                             [Preview]  │
└────────────────────────────────────────────────┘

        ┌─────────────────────────────┐
        │  Revision from 2h ago    X  │
        │                             │
        │  api.url = "http://..."     │
        │  model_name = "gemma3"      │
        │                             │
        │            [Restore]        │
        └─────────────────────────────┘
```

- **List rows:** relative time + absolute time on one line, "Preview" link on the right.
- **Dialog:** `TomlEditor` (readonly) + Restore button. The app's `Dialog` is already built for this.

| Pros | Cons |
|------|------|
| Maximum space for TOML (dialog is centered, ~560px) | Modal stacking on mobile (side panel is already a drawer) |
| Very simple list implementation | One more click to dismiss |
| Restore is isolated — harder to misclick | |
| Easy to swap in a diff view later | |

---

## Alternative 5: Incremental Diff Sections (selected)

Each entry is a section in a vertical stack. Sections are **always expanded** — no collapse/expand interaction. By default each section shows **only what changed in that save** (diff vs. the previous version). A toggle at the top of the section flips to the raw full snapshot.

```
┌─ Revision History ─────────────────────────────┐
│                                              ↻ │
├────────────────────────────────────────────────┤
│ ┌─ 2h ago — model_name changed ──────────────┐ │
│ │ May 30, 2026, 10:42 AM           [Restore] │ │
│ │                                [Show full] │ │
│ ├────────────────────────────────────────────┤ │
│ │ - model_name = "old-model"                 │ │
│ │ + model_name = "gemma3"                    │ │
│ └────────────────────────────────────────────┘ │
│ ─────────────────────────────────────────────  │
│ ┌─ 5h ago — added dataset entry ─────────────┐ │
│ │ May 30, 2026, 07:42 AM           [Restore] │ │
│ │                                [Show diff] │ │
│ ├────────────────────────────────────────────┤ │
│ │ + [[dataset]]                              │ │
│ │ + path = "/tmp/new_images"                 │ │
│ └────────────────────────────────────────────┘ │
│ ─────────────────────────────────────────────  │
│ ┌─ 1d ago — Initial config ──────────────────┐ │
│ │ May 29, 2026, 08:00 AM           [Restore] │ │
│ │                                [Show full] │ │
│ ├────────────────────────────────────────────┤ │
│ │ api.url = "http://localhost:11434"         │ │
│ │ model_name = "gemma3"                      │ │
│ │ ...                                        │ │
│ └────────────────────────────────────────────┘ │
│                                                │
│         [Show 5 older revisions]               │
└────────────────────────────────────────────────┘
```

### How diffs map to entries

History is saved pre-write, and the API returns newest-first. Each entry's default diff shows the changes introduced **by that save** (this version minus the one below it):

| Entry | Represents | Diff shows |
|-------|-----------|------------|
| Top | State after latest save | Changes in latest save (vs next entry) |
| Middle | State after previous save | Changes in that save (vs next entry) |
| Bottom | Initial config | Raw snapshot only (no previous) |

The current live config on disk is the same as the topmost history entry. No separate "Current" row is needed.

**Pagination:**

- Fetch 5 entries by default
- "Show N older" fetches the next batch via `before_id` cursor, appends to the list
- No page numbers — simple infinite-style loading

**Implementation notes:**

- Uses `@codemirror/merge` (`UnifiedMergeView`) for read-only diff rendering with green/red highlights and `+`/`-` gutter markers
- Each diff is a lightweight read-only CodeMirror instance — no accept/reject UI
- Restore button at the top of each section, styled as `.btn-secondary`, with `confirm()` dialog
- "Show full" / "Show diff" toggle at the top of each section — keeping buttons at the top ensures scroll position stays consistent when switching views
- Entry header shows relative time (primary) and absolute timestamp (secondary, muted)

| Pros | Cons |
|------|------|
| Full ~440px for each section's content | More CodeMirror instances to manage |
| "What changed?" answered immediately | |
| No sidebar split, works on mobile | |
| Raw snapshot one click away | |
| Buttons at top = stable scroll position | |

---

## Decision

Adopt **alternative 5** with non-collapsible sections.

**Rationale:**
- The user's mental model is "what did I change at this point?" not "what did the full file look like?"
- Diff sections are naturally scannable — green/red highlights make changes obvious
- Raw snapshot is one click away when needed
- No sidebar means the full ~440px is available for each section
- Buttons at the top keep scroll position stable when toggling views

**Files touched:** `DatasetConfig.svelte`, `ConfigHistory.svelte` (to be rewritten), `configs.ts` (no changes — existing pagination params already support this)
