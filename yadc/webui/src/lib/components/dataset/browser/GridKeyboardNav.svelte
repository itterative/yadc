<script lang="ts">
    /* eslint-disable svelte/prefer-svelte-reactivity -- The nav graph and
       its build-time grouping maps are an imperative cache, never read by
       the template, so SvelteMap's reactivity overhead is unwanted here. */
    // Grid keyboard navigation for the dataset image gallery.
    //
    // Arrow keys move the focused image — the host keeps ``focusedId`` and
    // is notified via ``onfocus`` so the side panel / selection outline can
    // follow. The component nests inside the grid and discovers its container
    // via the ``data-grid-container`` marker.
    //
    // The gallery is a masonry layout, so the two axes use different
    // strategies. Left/right is **spatial** and **wraps**: a horizontal ray
    // is cast from the focused tile's top edge, and the target is the first
    // tile that ray crosses when scanning right — so on a dataset with
    // varying heights it moves to the adjacent column at a similar position
    // rather than jumping by array index (which lands in arbitrary columns).
    // At the right edge it wraps to the start of the next reading row, so
    // repeated right-nav walks the whole grid and cycles back to the first.
    // This is modelled as a single cyclic chain built greedily (see
    // ``buildChain``). Up/down is **within-column** via DOM order: each
    // column in ``DatasetBrowser`` is a ``flex flex-col`` div of tile
    // buttons, so the previous/next tile in the same column is the one
    // directly above/below. This keeps every image reachable vertically
    // (spatial up/down could skip awkwardly-placed tiles) and does not wrap.
    //
    // Neighbors are **precomputed into a graph** (``Map<id, Neighbors>``)
    // rather than recalculated per keypress. Neighbor relationships are
    // scroll-invariant (scrolling shifts every tile equally, so relative
    // positions don't change), so the graph only needs rebuilding when the
    // DOM mutates (tiles added/removed) or the layout reflows (resize). A
    // ``MutationObserver`` + ``ResizeObserver`` schedule rebuilds via a
    // coalescing ``requestAnimationFrame`` so a burst of changes (e.g. a
    // load-more batch) rebuilds once, after layout settles.
    //
    // Inert when ``disabled`` (lightbox open) or when nothing is focused,
    // and ignores key events that originate in editable fields so caption
    // editing isn't disrupted.
    import type { ImageInfo } from '$lib/stores/dataset';

    interface Props {
        /** ID of the currently focused image, or null when nothing is
         *  focused. Navigation is inert until the user clicks a tile. */
        focusedId: number | null;
        /** Loaded images — used to resolve a navigated tile id back to its
         *  ``ImageInfo`` for the ``onfocus`` callback. */
        images: ImageInfo[];
        /** When true, all key handling is skipped. Set while the lightbox is
         *  open so its own arrows take over. */
        disabled?: boolean;
        /** More images can be paginated. Right-nav at the edge fetches the
         *  next page so the next press can reach newly loaded tiles. */
        hasMore?: boolean;
        isLoadingMore?: boolean;
        onfocus: (item: ImageInfo) => void;
        onloadmore?: () => void;
    }

    let {
        focusedId,
        images,
        disabled = false,
        hasMore = false,
        isLoadingMore = false,
        onfocus,
        onloadmore
    }: Props = $props();

    // Placeholder so the component has a DOM mount point to discover its grid
    // container from. ``hidden`` (display: none) keeps it out of the grid
    // layout — a bare element would occupy a cell.
    let host: HTMLElement | undefined = $state();
    let element = $derived(host?.closest('[data-grid-container]') as HTMLElement | undefined);

    interface Neighbors {
        left: number | null;
        right: number | null;
        up: number | null;
        down: number | null;
    }

    // Plain (non-reactive) variable: only the imperative keydown handler
    // reads it, never the template. Reassignment is visible to closures in
    // the same script because they share the binding.
    let navGraph: Map<number, Neighbors> = new Map();
    // The chain in traversal order. The tail (last id) is the right-wrap
    // seam: pressing right there either loads more (if paginated) or wraps
    // to the head (first id).
    let chainOrder: number[] = [];

    interface MeasuredTile {
        id: number;
        /** Position in the host ``images`` prop. Used for the chain start and
         *  wrap target so traversal follows backend order rather than DOM
         *  order (the grid renders column-major, so ``querySelectorAll``
         *  yields column-by-column, not the source order). */
        index: number;
        left: number;
        right: number;
        top: number;
        bottom: number;
        centerX: number;
        parent: ParentNode | null;
    }

    // Cap how many candidates we scan per step. The masonry distributes
    // consecutive indices across columns round-robin-ish, so the right
    // neighbour of the current tile lands within the next few indices — and
    // ``candidates`` iterates index-ascending (Set preserves insertion
    // order, and the chain deletes visited tiles as it walks), so its first
    // few elements are exactly those likely-rightward tiles. Tunable: raise
    // for very wide grids.
    const RIGHT_NEIGHBOR_SCAN_LIMIT = 10;
    // A candidate whose top starts below this fraction of the current tile's
    // height (measured from the current tile's top) is a next-row tile, not a
    // right neighbour, so it's skipped. Lower = stricter vertical band.
    const RIGHT_NEIGHBOR_MAX_TOP_FRACTION = 0.5;

    // Rightward spatial neighbor for one tile, mirroring the original
    // ray-cast: among the first few candidates with a real rightward offset,
    // prefer one whose vertical span contains the focused top (a ray "hit");
    // break ties by closest left edge (the nearest tile the ray crosses),
    // falling back to smallest vertical gap to the ray. A candidate whose
    // top starts below the top half of the current tile
    // (``RIGHT_NEIGHBOR_MAX_TOP_FRACTION``) is a next-row tile, not a right
    // neighbour, so it's skipped. Candidates starting at or above ``cur.top``
    // are kept — a tall neighbour that began higher but overlaps is still
    // valid, and the ray-cast handles it. Direction is always right: left is
    // the chain's mirror, not computed here.
    function rightNeighbor(candidates: Set<MeasuredTile>, cur: MeasuredTile): number | null {
        const cx = cur.centerX;
        const rayY = cur.top;
        const tooLowY = cur.top + (cur.bottom - cur.top) * RIGHT_NEIGHBOR_MAX_TOP_FRACTION;
        let best: number | null = null;
        let bestHit = false;
        let bestEdge = Infinity;
        let bestGap = Infinity;
        let scanned = 0;

        for (const t of candidates) {
            if (++scanned > RIGHT_NEIGHBOR_SCAN_LIMIT) {
                break;
            }
            const dx = t.centerX - cx;
            // Same-column tiles share a center x (dx ≈ 0); require a real
            // rightward offset so they're never considered.
            if (dx <= 1) {
                continue;
            }
            // Skip next-row tiles: a right neighbour should start near the
            // current tile's top, within its top half.
            if (t.top > tooLowY) {
                continue;
            }
            const hit = t.top <= rayY && rayY <= t.bottom;
            const gap = hit ? 0 : Math.min(Math.abs(rayY - t.top), Math.abs(rayY - t.bottom));
            // A ray hit always beats a miss; among hits the closest left edge
            // wins; among misses the smallest vertical gap wins, ties broken
            // by edge.
            const better = hit && !bestHit;
            const betterEdge = hit === bestHit && t.left < bestEdge && (bestHit || gap <= bestGap);
            const betterGap = hit === bestHit && !bestHit && gap < bestGap;
            if (better || betterEdge || betterGap) {
                best = t.id;
                bestHit = hit;
                bestEdge = t.left;
                bestGap = gap;
            }
        }
        return best;
    }

    // Greedily build a single cyclic traversal order over all tiles. Each
    // step advances to the directional (right) spatial neighbor among
    // **unvisited** tiles — removing each visited tile from the candidate
    // set so the walk never revisits. At the right edge every remaining tile
    // is to the left, so the directional search returns null; the wrap then
    // jumps to the earliest unvisited tile in backend order (next
    // reading-row start). Returns tile ids in traversal order; callers derive
    // left/right via modular indexing so the chain closes into a cycle
    // (last → first). ``tiles`` must be sorted by ``index`` ascending.
    function buildChain(tiles: MeasuredTile[]): number[] {
        const n = tiles.length;
        if (n === 0) {
            return [];
        }
        // ``candidates`` is the set of unvisited tiles. It's populated in
        // index order so iteration yields the lowest-index tile first
        // (Set preserves insertion order): the chain start and each wrap
        // target is just ``candidates``'s first remaining element.
        const candidates = new Set<MeasuredTile>(tiles);
        const byId = new Map<number, MeasuredTile>();
        for (const t of tiles) {
            byId.set(t.id, t);
        }
        const order: number[] = [];
        let current = candidates.values().next().value;
        for (let i = 0; i < n && current; i++) {
            order.push(current.id);
            candidates.delete(current);
            // Advance rightward among the remaining tiles, wrapping to the
            // next reading-row start when no forward neighbor remains.
            let nextId = rightNeighbor(candidates, current);
            if (nextId === null) {
                nextId = candidates.values().next().value?.id ?? null;
            }
            current = nextId !== null ? byId.get(nextId) : undefined;
        }
        return order;
    }

    function rebuildGraph() {
        const el = element;
        if (!el) {
            navGraph = new Map();
            chainOrder = [];
            return;
        }
        // Read every rect in one synchronous pass (no interleaved writes) so
        // the browser only flushes layout once — the layout-thrash-safe
        // pattern. ``querySelectorAll`` is document order (column-major);
        // the chain start/wrap use each tile's backend index from
        // ``data-image-idx`` (set by ``DatasetImage``), not DOM order.
        const tiles: MeasuredTile[] = [];
        for (const node of el.querySelectorAll('[data-image-id]')) {
            const r = (node as HTMLElement).getBoundingClientRect();
            tiles.push({
                id: Number((node as HTMLElement).dataset.imageId),
                index: Number((node as HTMLElement).dataset.imageIdx ?? -1),
                left: r.left,
                right: r.right,
                top: r.top,
                bottom: r.bottom,
                centerX: r.left + r.width / 2,
                parent: node.parentElement
            });
        }
        // Sort once by backend index so the chain start/wrap can be resolved
        // by a simple first-unvisited scan, and so traversal follows source
        // order rather than the column-major DOM order above. The order of
        // ``tiles`` doesn't matter to ``rightNeighbor`` (a full comparison
        // scan) or the column grouping below.
        tiles.sort((a, b) => a.index - b.index);

        // Group by column div so up/down can use within-column document order.
        const columns = new Map<ParentNode | null, MeasuredTile[]>();
        for (const t of tiles) {
            let arr = columns.get(t.parent);
            if (!arr) {
                arr = [];
                columns.set(t.parent, arr);
            }
            arr.push(t);
        }

        const graph = new Map<number, Neighbors>();
        // Left/right come from a single cyclic chain (greedy spatial walk
        // with wrap) so the two are exact mirrors and right-nav wraps at the
        // edges. O(N · S) arithmetic overall (N chain steps × at most
        // ``RIGHT_NEIGHBOR_SCAN_LIMIT`` candidates each), so effectively
        // O(N); fine for large grids. No layout reads happen here — rects
        // were captured in the measurement pass above.
        const chain = buildChain(tiles);
        chainOrder = chain;
        const m = chain.length;
        for (let i = 0; i < m; i++) {
            graph.set(chain[i], {
                left: chain[(i - 1 + m) % m],
                right: chain[(i + 1) % m],
                up: null,
                down: null
            });
        }
        for (const arr of columns.values()) {
            for (let i = 0; i < arr.length; i++) {
                const n = graph.get(arr[i].id);
                if (n) {
                    n.up = i > 0 ? arr[i - 1].id : null;
                    n.down = i < arr.length - 1 ? arr[i + 1].id : null;
                }
            }
        }
        navGraph = graph;
    }

    // Rebuild on DOM mutation or resize. Coalesced through a single rAF so a
    // batch of changes (e.g. load-more appending many tiles) triggers one
    // rebuild, after layout has settled.
    $effect(() => {
        const el = element;
        if (!el) {
            return;
        }
        let raf = 0;
        const schedule = () => {
            if (raf) {
                return;
            }
            raf = requestAnimationFrame(() => {
                raf = 0;
                rebuildGraph();
            });
        };
        const mo = new MutationObserver(schedule);
        mo.observe(el, { childList: true, subtree: true });
        const ro = new ResizeObserver(schedule);
        ro.observe(el);
        schedule();
        return () => {
            if (raf) {
                cancelAnimationFrame(raf);
            }
            mo.disconnect();
            ro.disconnect();
        };
    });

    function isEditableTarget(target: EventTarget | null): boolean {
        if (!(target instanceof HTMLElement)) {
            return false;
        }
        if (target.isContentEditable) {
            return true;
        }
        const tag = target.tagName;
        if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') {
            return true;
        }
        // CodeMirror renders its editor inside a ``.cm-editor`` container.
        if (target.closest('.cm-editor')) {
            return true;
        }
        return false;
    }

    function focusById(id: number) {
        const item = images.find((img) => img.id === id);
        if (!item) {
            return;
        }
        onfocus(item);
        element
            ?.querySelector(`[data-image-id="${id}"]`)
            ?.scrollIntoView({ block: 'nearest', inline: 'nearest' });
    }

    function moveHorizontal(direction: 'left' | 'right') {
        if (focusedId === null) {
            return;
        }
        // At the right-wrap seam with more images to load, grow the grid
        // instead of wrapping so keyboard-only users can reach every page.
        // Once fully loaded, the same press wraps to the first tile.
        const tail = chainOrder.length > 0 ? chainOrder[chainOrder.length - 1] : null;
        if (direction === 'right' && focusedId === tail && hasMore && !isLoadingMore) {
            onloadmore?.();
            return;
        }
        const next = navGraph.get(focusedId)?.[direction] ?? null;
        if (next !== null) {
            focusById(next);
        }
    }

    function moveWithinColumn(direction: 'up' | 'down') {
        if (focusedId === null) {
            return;
        }
        const next = navGraph.get(focusedId)?.[direction] ?? null;
        if (next !== null) {
            focusById(next);
        }
    }

    function handleKeydown(ev: KeyboardEvent) {
        if (disabled || focusedId === null || isEditableTarget(ev.target)) {
            return;
        }
        if (ev.key === 'ArrowLeft') {
            ev.preventDefault();
            moveHorizontal('left');
        } else if (ev.key === 'ArrowRight') {
            ev.preventDefault();
            moveHorizontal('right');
        } else if (ev.key === 'ArrowUp') {
            ev.preventDefault();
            moveWithinColumn('up');
        } else if (ev.key === 'ArrowDown') {
            ev.preventDefault();
            moveWithinColumn('down');
        }
    }
</script>

<svelte:window onkeydown={handleKeydown} />

<div hidden bind:this={host} aria-hidden="true"></div>
