<script lang="ts">
  import IntersectionObserverElement from "$lib/components/ui/IntersectionObserverElement.svelte";
  import DatasetImage from "$lib/components/dataset/DatasetImage.svelte";
  import SpinnerBlock from "$lib/components/ui/SpinnerBlock.svelte";
  import { random } from "$lib/random";
  import type { ImageInfo } from "$lib/stores/datasetImages";

  interface Props {
    class?: string;
    datasetName: string;
    items: ImageInfo[];
    isLoading: boolean;
    isLoadingMore: boolean;
    selectedId?: number | null;
    onclick: (item: ImageInfo) => void;
    onendreached: () => void;
  }

  let {
    class: klazz = "",
    datasetName,
    items = [],
    isLoading,
    isLoadingMore,
    selectedId = null,
    onclick,
    onendreached,
  }: Props = $props();

  const DEFAULT_GRID_COLUMNS = 4;

  let width = $state(-1);
  let container: HTMLElement;

  let gridColumns = $derived.by(() => {
    if (width < 0) {
      return DEFAULT_GRID_COLUMNS;
    }

    try {
      const cssVar = getComputedStyle(container).getPropertyValue("--x-grid-cols");
      const parsed = parseInt(cssVar);
      return isNaN(parsed) ? DEFAULT_GRID_COLUMNS : parsed;
    } catch {
      return DEFAULT_GRID_COLUMNS;
    }
  });

  function argmin(array: number[]) {
    if (!array.length) {
      return 0;
    }

    let argminIndex = 0;
    let argminValue = array[0];

    for (let i = 1; i < array.length; i++) {
      if (array[i] >= argminValue) {
        continue;
      }

      argminIndex = i;
      argminValue = array[i];
    }

    return argminIndex;
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
        height: normalWidth * (stubRandomGenerator() + 1),
      };

      const index = argmin(columnHeights);
      splitStubs[index].push(stub);

      columnHeights[index] += stub.height * (normalWidth / stub.width) + 1;
    }

    return splitStubs;
  });
</script>

<div class={klazz} style="overflow: hidden;">
  <div
    class="grid-cols-auto grid w-full [--x-grid-cols:2] lg:[--x-grid-cols:3] xl:[--x-grid-cols:4] 2xl:[--x-grid-cols:5] gap-4"
    bind:this={container}
    bind:offsetWidth={width}
  >
    {#each columns as column, column_index (column_index)}
      <div class="flex flex-col gap-4">
        {#each column as item (`${column_index}-${item.id}`)}
          <DatasetImage
            class="row-end-[auto_span_20px] rounded-xl overflow-hidden bg-gray-800 shadow-md transform transition-all hover:scale-105 hover:shadow-xl cursor-pointer relative"
            {datasetName}
            {item}
            selected={item.id === selectedId}
            onclick={() => onclick(item)}
          />
        {/each}

        {#if isLoading}
          {#each columnsStubs[column_index] || [] as stub, stub_index (stub_index)}
            <div class="row-end-[auto_span_20px] rounded-xl overflow-hidden bg-gray-800 shadow-md transform transition-all hover:scale-105 hover:shadow-xl">
              <div class="w-full bg-gray-500 animate-pulse" style:aspect-ratio={`${stub.width} / ${stub.height}`}></div>
            </div>
          {/each}
        {/if}
      </div>
    {/each}

    <IntersectionObserverElement top={200} onintersect={onendreached} />
  </div>

  {#if isLoadingMore}
    <SpinnerBlock class="py-6" size="h-8 w-8" />
  {/if}
</div>

<style>
  .grid-cols-auto {
    display: grid;
    --grid-cols: var(--x-grid-cols, 1);
    grid-template-columns: repeat(var(--grid-cols), minmax(0, 1fr));
  }
</style>
