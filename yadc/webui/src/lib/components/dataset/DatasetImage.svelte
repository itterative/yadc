<script lang="ts">
  import type { ImageInfo } from "$lib/stores/datasetImages";

  interface Props {
    class?: string;
    datasetName: string;
    item: ImageInfo;
    selected?: boolean;
    onclick: (item: ImageInfo) => void;
  }

  let { class: klazz = "", datasetName, item, selected = false, onclick }: Props = $props();

  let media: HTMLImageElement | null = $state(null);
  let loading: boolean = $state(true);

  let thumbnailSrc = $derived(
    `/api/datasets/${encodeURIComponent(datasetName)}/images/${item.id}/thumbnail?size=512`,
  );

  let aspectRatio = $derived.by(() => {
    if (!item.width || !item.height) return undefined;
    return `${item.width} / ${item.height}`;
  });

  $effect(() => {
    const setLoadingToFalse = () => (loading = false);
    media?.addEventListener("load", setLoadingToFalse);
    return () => media?.removeEventListener("load", setLoadingToFalse);
  });
</script>

<button
  class={klazz}
  onclick={() => onclick(item)}
  class:has-caption={item.has_caption}
  class:has-toml={item.has_toml}
  class:selected={selected}
>
  <img
    src={thumbnailSrc}
    alt={item.file_name}
    loading="lazy"
    class="block w-full bg-gray-500"
    class:animate-pulse={loading}
    width={item.width || undefined}
    height={item.height || undefined}
    style:aspect-ratio={aspectRatio}
    bind:this={media}
  />

  {#if item.has_caption}
    <span class="caption-badge" title="Has caption">✓</span>
  {/if}

  {#if item.has_toml}
    <span class="toml-badge" title="Has TOML extras">T</span>
  {/if}

  {#if item.draft_names.length > 0}
    <span class="draft-badge" title="{item.draft_names.length} draft(s)">D</span>
  {/if}
</button>

<style>
  .has-caption {
    /* subtle indicator in the border */
  }

  .selected {
    outline: 2px solid var(--color-accent);
    outline-offset: -2px;
  }

  .caption-badge,
  .toml-badge,
  .draft-badge {
    position: absolute;
    top: 0.25rem;
    font-size: 0.625rem;
    font-weight: 700;
    padding: 0.125rem 0.25rem;
    border-radius: 0.25rem;
    line-height: 1;
    pointer-events: none;
  }

  .caption-badge {
    right: 0.25rem;
    background: var(--color-success);
    color: #000;
  }

  .toml-badge {
    right: 1.5rem;
    background: var(--color-accent);
    color: #000;
  }

  .draft-badge {
    left: 0.25rem;
    background: var(--color-error);
    color: #fff;
  }
</style>
