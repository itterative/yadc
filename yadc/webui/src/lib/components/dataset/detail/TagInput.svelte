<script lang="ts">
    import { getAbortContext, linkedController } from '$lib/abort';
    import { fetchTagSuggestions } from '$lib/stores/tagging/api';
    import type { TagSuggestion } from '$lib/stores/tagging/types';
    import { recentTags, recordRecentTag, removeRecentTag, tagSettings } from '$lib/stores/tagging';
    import SvgPlus from '$lib/icons/SvgPlus.svelte';
    import SvgClose from '$lib/icons/SvgClose.svelte';

    interface Props {
        /** Category this input belongs to — shown as context on the add button. */
        category: string;
        /** Tags already present in the model output. Used for dedupe (re-adding a
         *  pruned model tag is a separate flow handled by the parent). */
        modelTags: readonly string[];
        /** Currently-added custom tags (name → category). Used for dedupe. */
        existingCustomTags: ReadonlyMap<string, string>;
        /** Called with a normalized, validated tag name. The parent handles
         *  registering it and adding it to its enabled set. */
        onadd: (tag: string) => void;
    }

    let { category, modelTags, existingCustomTags, onadd }: Props = $props();

    /** Parent abort scope (set by ``Tags.svelte``'s ``createAbortContext``).
     *  Linking each fetch's controller to it cancels a pending autocomplete
     *  request when the image/dataset changes or the view tears down, not
     *  only when a newer keystroke supersedes it locally. */
    const parentSignal = getAbortContext();

    /** How many rows the dropdown shows at most — matches the backend's
     *  default ``limit`` and the suggestion LRU. */
    const DROPDOWN_LIMIT = 20;

    /** Native HTML ``popover`` API feature-detect.
     *
     *  The autocomplete dropdown lives on top of an inline input that may sit
     *  inside an ``overflow: hidden/auto`` ancestor, so it uses the browser's
     *  top layer (``popover="auto"``) to escape clipping. When the API is
     *  absent we hide the dropdown entirely and skip the fetch — there is no
     *  absolute-positioning fallback (it loses the offscreen-clamping /
     *  ancestor-scroll handling the top layer gives for free), so the user
     *  just gets the manual text input + commit/cancel flow.
     *
     *  Supported in Chrome 114+, Safari 17+, Firefox 125+ at time of writing. */
    const popoverSupported: boolean = (() => {
        if (typeof HTMLElement === 'undefined') {
            return false; // SSR — keep the check safe.
        }
        return 'popover' in HTMLElement.prototype;
    })();

    /** Whether the inline input is open. Local state — only one input across
     *  the Tags tab is open at a time, but the parent doesn't need to know. */
    let isOpen = $state(false);

    let value = $state('');
    let inputEl = $state<HTMLInputElement | null>(null);

    // The dropdown has two modes, unified through ``dropdownItems``:
    //   1. **Recent** (query empty): recently-added tags from
    //      ``recentTags`` (localStorage), filtered to tags not already on
    //      this image. Picking one keeps the dropdown open for several at
    //      once; each row has an inline × to delete it from history.
    //   2. **Fetched** (query non-empty): autocomplete hits from
    //      ``GET /api/tagging/suggest``, debounced + cached. Picking one
    //      closes the dropdown (the query is stale after a pick).
    //
    // ``forceHidden`` suppresses the dropdown when its fixed-position anchor
    // (the input) has scrolled out from under it — popovers don't track
    // their anchor, so we hide rather than show a misaligned list.

    /** Fetched suggestions. ``null`` while loading / not in fetched mode;
     *  ``[]`` when a fetch resolved with no matches. */
    let suggestions = $state<TagSuggestion[] | null>(null);
    /** Set by scroll/resize handlers to drop the dropdown until the next
     *  value/open change resets it. */
    let forceHidden = $state(false);
    /** Index of the currently highlighted row. ``-1`` = no highlight. */
    let highlighted = $state(-1);

    /** Recent tags not already present on this image, capped to the dropdown
     *  limit. Re-derives whenever the store, the model output, or the user's
     *  custom additions change. */
    let recentItems = $derived.by<TagSuggestion[]>(() => {
        const present = new Set<string>([...modelTags, ...existingCustomTags.keys()]);
        const out: TagSuggestion[] = [];
        for (const t of $recentTags.tags) {
            if (present.has(t.name)) {
                continue;
            }
            out.push({ name: t.name, category: t.category });
            if (out.length >= DROPDOWN_LIMIT) {
                break;
            }
        }
        return out;
    });

    /** ``true`` when the dropdown is showing recent tags (empty-query mode).
     *  Drives the per-row delete button and Esc semantics. */
    let showingRecent = $derived(
        isOpen && popoverSupported && !forceHidden && !value.trim() && recentItems.length > 0
    );

    /** The items currently rendered in the dropdown, regardless of source.
     *  ``null`` means "no dropdown" (closed, loading, or empty). */
    let dropdownItems = $derived.by<TagSuggestion[] | null>(() => {
        if (!isOpen || !popoverSupported || forceHidden) {
            return null;
        }
        if (value.trim()) {
            return suggestions;
        }
        return recentItems.length > 0 ? recentItems : null;
    });

    /** Debounce window before firing the suggestion fetch. */
    const SUGGESTION_DEBOUNCE_MS = 250;
    let debounceTimer: number | null = null;
    /** Abort controller for the in-flight fetch — replaced and aborted when a
     *  newer query supersedes it. */
    let inflightController: AbortController | null = null;

    $effect(() => {
        // Focus the input whenever it opens.
        if (isOpen) {
            inputEl?.focus();
        }
    });

    $effect(() => {
        // Drive the fetched-suggestions half. On any value/open change: reset
        // the anchor-loss flag, then either clear (empty query → recent takes
        // over) or schedule a debounced fetch.
        if (!isOpen || !popoverSupported) {
            suggestions = null;
            highlighted = -1;
            return;
        }
        forceHidden = false;
        const query = value.trim();
        if (!query) {
            suggestions = null;
            highlighted = -1;
            return;
        }

        if (debounceTimer !== null) {
            window.clearTimeout(debounceTimer);
        }
        inflightController?.abort();
        const controller = linkedController(parentSignal);
        inflightController = controller;
        suggestions = null;
        highlighted = -1;

        debounceTimer = window.setTimeout(async () => {
            debounceTimer = null;
            try {
                const result = await fetchTagSuggestions(
                    query,
                    controller.signal,
                    DROPDOWN_LIMIT,
                    $tagSettings.replaceUnderscores
                );
                if (!controller.signal.aborted) {
                    // Late results aren't useful — the user has moved on.
                    suggestions = result;
                    highlighted = -1;
                }
            } catch (e) {
                if ((e as Error).name !== 'AbortError') {
                    suggestions = [];
                }
            }
        }, SUGGESTION_DEBOUNCE_MS);

        return () => {
            if (debounceTimer !== null) {
                window.clearTimeout(debounceTimer);
                debounceTimer = null;
            }
            controller.abort();
        };
    });

    /** Outcome of :func:`tryAdd`, so callers can decide whether to record
     *  the tag in recent history: freshly-added tags (custom or picked from
     *  the catalog) are recorded; re-enabling a model tag isn't, and
     *  duplicates are a no-op. */
    type AddOutcome = 'added' | 'model' | 'duplicate';

    /** Dedupe + commit a tag. Used by both the manual-commit and dropdown-pick
     *  paths so the rules don't drift. Returns the outcome so the caller
     *  knows whether to record the tag in recent history. */
    function tryAdd(rawTag: string): AddOutcome {
        if (modelTags.includes(rawTag)) {
            // Re-enable the pruned model tag rather than double-registering it.
            onadd(rawTag);
            return 'model';
        }

        if (existingCustomTags.has(rawTag)) {
            return 'duplicate';
        }

        onadd(rawTag);
        return 'added';
    }

    function commit() {
        const raw = value.trim();
        if (!raw) {
            isOpen = false;
            value = '';
            return;
        }
        const outcome = tryAdd(raw);
        if (outcome === 'added') {
            // Manually-typed tags have no catalog category — record them as
            // 'custom' so the recent dropdown can badge them distinctly.
            recordRecentTag(raw, 'custom');
        }
        if (outcome !== 'duplicate') {
            value = '';
        }
    }

    /** Commit a tag selected from the dropdown. The category is recorded into
     *  recent history so the badge persists across sessions. In recent mode
     *  the dropdown stays open (grabbing several reuse tags); in fetched mode
     *  it closes (the query is stale once a tag is chosen). */
    function commitFromDropdown(name: string, category: string) {
        const wasRecentPick = showingRecent;
        const outcome = tryAdd(name);
        if (outcome === 'added') {
            recordRecentTag(name, category);
        }
        highlighted = -1;
        if (wasRecentPick) {
            return; // ``recentItems`` re-derives and drops the picked tag.
        }
        value = '';
        suggestions = null;
    }

    /** Drop a recent tag from history (the inline × button). Stopping the
     *  event keeps the input focused so the dropdown doesn't blur-commit. */
    function onRemoveRecent(e: MouseEvent, name: string) {
        e.preventDefault();
        e.stopPropagation();
        removeRecentTag(name);
    }

    /** Category badge styling for each dropdown row. The danbooru catalog
     *  carries ``general`` / ``artist`` / ``copyright`` / ``character`` /
     *  ``meta``; ``custom`` is a manually-typed tag; ``rating`` is kept as a
     *  defensive fallback (the catalog has no rating category today). */
    function suggestionCategoryClass(category: string): string {
        switch (category) {
            case 'rating':
                return 'bg-warning/20 text-warning';
            case 'character':
            case 'custom':
                return 'bg-purple-500/20 text-purple-300';
            case 'general':
            default:
                return 'bg-gray-700/50 text-gray-400';
        }
    }

    function cancel() {
        isOpen = false;
        value = '';
    }

    /** Dismiss fetched suggestions without committing — keeps the input open
     *  so the user can refine the query. Used by Esc in fetched mode and the
     *  popover's outside-click close. */
    function dismissSuggestions() {
        suggestions = null;
        highlighted = -1;
    }

    /** Hide the dropdown because its anchor (the input) has scrolled fully
     *  out of the viewport. Resets on the next value/open change (see the
     *  fetch effect). */
    function hideForAnchorLoss() {
        forceHidden = true;
        highlighted = -1;
    }

    let dropdownEl = $state<HTMLUListElement | null>(null);

    /** Recompute the dropdown's fixed position from the input's current rect
     *  and clamp it into the viewport. Called on first show and on every
     *  scroll / resize / content-reflow so the popover *follows* its anchor.
     *
     *  Returns ``false`` when the input has scrolled entirely out of the
     *  viewport — the caller then hides via :func:`hideForAnchorLoss`. */
    function positionDropdown(el: HTMLElement, input: HTMLElement): boolean {
        const rect = input.getBoundingClientRect();
        const padding = 4;
        if (rect.bottom < 0 || rect.top > window.innerHeight) {
            return false;
        }
        const minWidth = Math.max(Math.round(rect.width), 160);

        el.style.minWidth = `${minWidth}px`;
        el.style.top = `${Math.round(rect.bottom + padding)}px`;
        el.style.left = `${Math.round(rect.left)}px`;
        if (!el.matches(':popover-open')) {
            el.showPopover();
        }

        // Clamp into the viewport: flip above the input if there's no room
        // below, and shift the leading edge left if it would run off the right.
        const ddRect = el.getBoundingClientRect();
        let top = rect.bottom + padding;
        if (top + ddRect.height > window.innerHeight - padding) {
            const flipped = rect.top - ddRect.height - padding;
            top = Math.max(flipped, padding);
        }
        let left = rect.left;
        if (left + ddRect.width > window.innerWidth - padding) {
            left = Math.max(window.innerWidth - ddRect.width - padding, padding);
        }
        el.style.top = `${Math.round(top)}px`;
        el.style.left = `${Math.round(left)}px`;
        return true;
    }

    $effect(() => {
        // Sync popover visibility with the dropdown state. ``auto`` popovers
        // also auto-close on outside click and Esc.
        const el = dropdownEl;
        const input = inputEl;
        if (!el) {
            return;
        }
        const hasItems = dropdownItems !== null && dropdownItems.length > 0;

        if (hasItems && input) {
            if (!positionDropdown(el, input)) {
                hideForAnchorLoss();
                return;
            }

            // Popovers are top-layer + fixed-positioned, so they don't track
            // their anchor — re-position on everything that can move the input.
            // The ``ResizeObserver`` on the nearest scrollable ancestor is the
            // important one: it catches *content* reflows (a chip wrapping the
            // row, the save-preview height changing) that aren't scroll/resize
            // events. Selective ancestor walk mirrors ``ContextMenu``.
            const cleanups: (() => void)[] = [];
            let scrollableAncestor: HTMLElement | null = null;
            let current: Element | null = input.parentElement;
            while (current) {
                const cs = getComputedStyle(current);
                if (
                    cs.overflow === 'auto' ||
                    cs.overflow === 'scroll' ||
                    cs.overflowY === 'auto' ||
                    cs.overflowY === 'scroll'
                ) {
                    const node = current;
                    const handler = () => {
                        if (!positionDropdown(el, input)) {
                            hideForAnchorLoss();
                        }
                    };
                    node.addEventListener('scroll', handler, { passive: true });
                    cleanups.push(() => node.removeEventListener('scroll', handler));
                    if (!scrollableAncestor) {
                        scrollableAncestor = node as HTMLElement;
                    }
                }
                current = current.parentElement;
            }
            const onResize = () => {
                if (!positionDropdown(el, input)) {
                    hideForAnchorLoss();
                }
            };
            window.addEventListener('resize', onResize);
            cleanups.push(() => window.removeEventListener('resize', onResize));

            const ro = new ResizeObserver(() => {
                if (!positionDropdown(el, input)) {
                    hideForAnchorLoss();
                }
            });
            if (scrollableAncestor) {
                ro.observe(scrollableAncestor);
            }
            cleanups.push(() => ro.disconnect());

            return () => {
                for (const cleanup of cleanups) {
                    cleanup();
                }
            };
        } else if (el.matches(':popover-open')) {
            el.hidePopover();
        }
    });

    function onKeyDown(e: KeyboardEvent) {
        const items = dropdownItems;
        if (items && items.length > 0) {
            if (e.key === 'ArrowDown') {
                e.preventDefault();
                highlighted = (highlighted + 1) % items.length;
                return;
            }
            if (e.key === 'ArrowUp') {
                e.preventDefault();
                highlighted = highlighted <= 0 ? items.length - 1 : highlighted - 1;
                return;
            }
            if (e.key === 'Enter') {
                e.preventDefault();
                if (highlighted >= 0 && items[highlighted] !== undefined) {
                    commitFromDropdown(items[highlighted].name, items[highlighted].category);
                } else {
                    commit();
                }
                return;
            }
            if (e.key === 'Escape') {
                e.preventDefault();
                if (showingRecent) {
                    cancel(); // Recent is the home state — Esc closes the input.
                } else {
                    dismissSuggestions();
                }
                return;
            }
        }
        if (e.key === 'Enter') {
            e.preventDefault();
            commit();
        } else if (e.key === 'Escape') {
            e.preventDefault();
            cancel();
        }
    }

    /** Keep the highlighted row visible inside the dropdown's scroll
     *  viewport. ``block: 'nearest'`` scrolls only when the row is actually
     *  off-screen, so a highlight change that stays in view is a no-op. */
    $effect(() => {
        if (highlighted < 0 || !dropdownEl) {
            return;
        }
        // Use ``data-index`` rather than ``children[highlighted]`` because the
        // dropdown may include a non-item header row (the "Recent" label).
        const row = dropdownEl.querySelector(
            `li[data-index="${highlighted}"]`
        ) as HTMLElement | null;
        row?.scrollIntoView({ block: 'nearest' });
    });

    /** Click on a dropdown row: preventDefault on mousedown keeps focus on the
     *  input (so its blur doesn't close the input), then the click commits. */
    function onRowMouseDown(e: MouseEvent, item: TagSuggestion) {
        e.preventDefault();
        commitFromDropdown(item.name, item.category);
    }
</script>

<!--
    Inline "add a custom tag" affordance for the Tags tab. Renders as a
    dashed-outline chip in the same flex-wrap row as the model's tags; click
    → swaps for a small text input. Submit via Enter (keeps the input open
    for repeat entry); Esc or blur-with-empty-input closes it.

    While the input is open, a dropdown shows recent tags (localStorage) when
    the query is empty, or autocomplete hits from ``GET /api/tagging/suggest``
    when the user is typing (debounced 250ms, in-flight requests cancelled on
    each keystroke). ArrowDown / Up navigates, Enter commits the highlighted
    row, Esc dismisses fetched suggestions (or closes the input from recent).
-->
{#if isOpen}
    <input
        bind:this={inputEl}
        bind:value
        type="text"
        placeholder="new tag…"
        aria-label="Add a custom {category} tag"
        aria-expanded={dropdownItems !== null && dropdownItems.length > 0}
        aria-controls="tag-suggestions-{category}"
        aria-autocomplete="list"
        role="combobox"
        class="w-28 rounded-md border border-purple-500/50 bg-purple-500/10 px-2 py-1 text-xs text-purple-200 placeholder:text-purple-300/50 focus:border-purple-400 focus:outline-none"
        onkeydown={onKeyDown}
        onblur={() => {
            // Discard the draft on blur — only explicit Enter (keyboard) or a
            // chip-click (pointer) commits a tag. Auto-committing on blur was
            // surprising: a stray click elsewhere would silently write
            // whatever text was sitting in the box (often half-typed).
            // Dropdown interactions keep focus on the input via
            // ``mousedown + preventDefault`` so they don't trip this blur.
            isOpen = false;
        }}
    />
    <!-- Always rendered (and only conditionally visible) so ``bind:this``
         fires on every input mount. The dropdown is a top-layer popover;
         its position is fixed on every open based on the input's bounding
         rect, escaping every overflow ancestor (absolute positioning was
         clipped by scrollers when the input sat near the bottom of the page).
         Skipped entirely on browsers without the popover API. -->
    {#if popoverSupported}
        <ul
            bind:this={dropdownEl}
            popover="auto"
            id="tag-suggestions-{category}"
            role="listbox"
            class="m-0 max-h-48 w-56 overflow-y-auto rounded-md border border-gray-700 bg-gray-800 text-xs shadow-lg"
            onbeforetoggle={(e) => {
                // Popover closing — could be our ``hidePopover``, the
                // browser's auto-close on outside click, or Esc. Sync state
                // so a future ``showPopover`` doesn't fire on a stale list.
                if (e.newState === 'closed') {
                    dismissSuggestions();
                }
            }}
        >
            {#if showingRecent}
                <li
                    class="px-2 py-1 text-[10px] tracking-wide text-gray-500 uppercase"
                    role="presentation"
                >
                    Recent
                </li>
            {/if}
            {#each dropdownItems ?? [] as item, i (item.name)}
                <li
                    role="option"
                    data-index={i}
                    aria-selected={i === highlighted}
                    class="flex cursor-pointer items-center justify-between gap-2 px-2 py-1.5 {i ===
                    highlighted
                        ? 'bg-accent/20 text-accent'
                        : 'text-gray-300 hover:bg-gray-700'}"
                    onmousedown={(e) => onRowMouseDown(e, item)}
                >
                    <span class="truncate">{item.name}</span>
                    <span class="flex shrink-0 items-center gap-1">
                        {#if showingRecent}
                            <button
                                type="button"
                                class="cursor-pointer rounded p-0.5 text-gray-500 hover:bg-gray-600 hover:text-gray-200"
                                title="Remove from recent"
                                aria-label="Remove {item.name} from recent"
                                onmousedown={(e) => onRemoveRecent(e, item.name)}
                                onclick={(e) => {
                                    e.preventDefault();
                                    e.stopPropagation();
                                }}
                            >
                                <SvgClose class="h-3 w-3" />
                            </button>
                        {/if}
                        <span
                            class="rounded px-1.5 py-0.5 text-[10px] tracking-wide uppercase {suggestionCategoryClass(
                                item.category
                            )}"
                            aria-label="category: {item.category}"
                        >
                            {item.category}
                        </span>
                    </span>
                </li>
            {/each}
        </ul>
    {/if}
{:else}
    <button
        type="button"
        class="cursor-pointer rounded-md border border-dashed border-gray-600 bg-transparent px-2 py-1 text-xs text-gray-500 transition-colors hover:border-gray-400 hover:text-gray-300"
        title="Add a custom {category} tag"
        onclick={() => (isOpen = true)}
    >
        <SvgPlus class="-mt-0.5 mr-1 inline h-3 w-3 align-middle" />
        custom
    </button>
{/if}
