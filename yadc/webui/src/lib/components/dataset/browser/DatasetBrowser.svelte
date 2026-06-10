<script lang="ts">
    import IntersectionObserverElement from '$lib/components/ui/IntersectionObserverElement.svelte';
    import DatasetImage from '$lib/components/dataset/browser/DatasetImage.svelte';
    import SpinnerBlock from '$lib/components/ui/SpinnerBlock.svelte';
    import { random } from '$lib/random';
    import type { ImageInfo } from '$lib/stores/dataset';

    interface Props {
        class?: string;
        datasetName: string;
        items: ImageInfo[];
        isLoading: boolean;
        isLoadingMore: boolean;
        selectedId?: number | null;
        /** IDs of images currently being captioned (in this dataset).
         *  Under ``max_concurrent > 1`` this can hold multiple IDs. */
        captioningIds?: ReadonlySet<number>;
        onclick: (item: ImageInfo) => void;
        ondblclick?: (item: ImageInfo) => void;
        onendreached: () => void;
    }

    let {
        class: klazz = '',
        datasetName,
        items = [],
        isLoading,
        isLoadingMore,
        selectedId = null,
        captioningIds = new Set<number>(),
        onclick,
        ondblclick,
        onendreached
    }: Props = $props();

    const DEFAULT_GRID_COLUMNS = 4;

    let width = $state(-1);
    let container: HTMLElement;

    let gridColumns = $derived.by(() => {
        if (width < 0) {
            return DEFAULT_GRID_COLUMNS;
        }

        try {
            const cssVar = getComputedStyle(container).getPropertyValue('--x-grid-cols');
            const parsed = parseInt(cssVar);
            return isNaN(parsed) ? DEFAULT_GRID_COLUMNS : parsed;
        } catch {
            return DEFAULT_GRID_COLUMNS;
        }
    });

    const COLUMN_TOLERANCE = 50;

    function argmin(array: number[]) {
        if (!array.length) {
            return 0;
        }

        let minValue = array[0];

        for (let i = 1; i < array.length; i++) {
            if (array[i] < minValue) {
                minValue = array[i];
            }
        }

        // Pick leftmost column within tolerance of the minimum height
        const threshold = minValue + COLUMN_TOLERANCE;
        for (let i = 0; i < array.length; i++) {
            if (array[i] <= threshold) {
                return i;
            }
        }

        // Fallback (shouldn't happen since minValue <= threshold)
        return 0;
    }

    const [columns, columnHeights] = $derived.by(() => {
        if (width < 0) {
            return [[], []];
        }

        const normalWidth = 100;
        const columnHeights: number[] = new Array(gridColumns).fill(0);
        const splitItems: ImageInfo[][] = new Array(gridColumns).fill(1).map(() => []);

        for (const item of items) {
            const index = argmin(columnHeights);
            splitItems[index].push(item);

            if (!item.width || !item.height) {
                columnHeights[index] += 1;
                continue;
            }

            columnHeights[index] += item.height * (normalWidth / item.width) + 1;
        }

        return [splitItems, columnHeights];
    });

    interface Stub {
        width: number;
        height: number;
    }

    const LOADING_STUB_IMAGES = 25;
    const stubRandomGenerator = random(42);

    const columnsStubs = $derived.by(() => {
        if (width < 0) {
            return [];
        }

        const normalWidth = 100;
        const splitStubs: Stub[][] = new Array(gridColumns).fill(1).map(() => []);

        for (let i = 0; i < LOADING_STUB_IMAGES; i++) {
            const stub: Stub = {
                width: normalWidth * (stubRandomGenerator() + 1),
                height: normalWidth * (stubRandomGenerator() + 1)
            };

            const index = argmin(columnHeights);
            splitStubs[index].push(stub);

            columnHeights[index] += stub.height * (normalWidth / stub.width) + 1;
        }

        return splitStubs;
    });
</script>

<div class="{klazz} relative" style="overflow: hidden;">
    <div
        class="grid-cols-auto grid w-full gap-4 [--x-grid-cols:2] lg:[--x-grid-cols:3] xl:[--x-grid-cols:4] 2xl:[--x-grid-cols:5]"
        bind:this={container}
        bind:offsetWidth={width}
    >
        {#each columns as column, column_index (column_index)}
            <div class="flex flex-col gap-4">
                {#each column as item (`${column_index}-${item.id}`)}
                    <DatasetImage
                        class="relative row-end-[auto_span_20px] transform cursor-pointer overflow-hidden rounded-xl bg-gray-800 shadow-md transition-all hover:scale-105 hover:shadow-xl"
                        {datasetName}
                        {item}
                        selected={item.id === selectedId}
                        captioning={captioningIds.has(item.id)}
                        onclick={() => onclick(item)}
                        ondblclick={ondblclick ? () => ondblclick(item) : undefined}
                    />
                {/each}

                {#if isLoading}
                    {#each columnsStubs[column_index] || [] as stub, stub_index (stub_index)}
                        <div
                            class="row-end-[auto_span_20px] transform overflow-hidden rounded-xl bg-gray-800 shadow-md transition-all hover:scale-105 hover:shadow-xl"
                        >
                            <div
                                class="w-full animate-pulse bg-gray-500"
                                style:aspect-ratio={`${stub.width} / ${stub.height}`}
                            ></div>
                        </div>
                    {/each}
                {/if}
            </div>
        {/each}
    </div>

    {#if isLoadingMore}
        <SpinnerBlock class="py-6" size="h-8 w-8" />
    {/if}

    <!-- Sentinel for infinite scroll. Positioned absolutely at the bottom of
         the scrollable parent so it always sits at the visual bottom of the
         gallery, regardless of column heights. (Inside the grid, the sentinel
         would be auto-placed in the second row, first column — which can sit
         higher than the bottom of a taller neighbouring column, causing the
         observer to never fire as the user scrolls.) -->
    <IntersectionObserverElement
        class="absolute right-0 bottom-4"
        top={200}
        onintersect={onendreached}
    />
</div>

<style>
    .grid-cols-auto {
        display: grid;
        --grid-cols: var(--x-grid-cols, 1);
        grid-template-columns: repeat(var(--grid-cols), minmax(0, 1fr));
    }
</style>
