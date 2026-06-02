<script lang="ts">
  import SvgSpinner from "$lib/icons/SvgSpinner.svelte";

  interface Props {
    class?: string;
    label?: string;
    size?: string;
  }

  let { class: klazz = "", label, size = "h-6 w-6" }: Props = $props();
</script>

<div class="flex items-center justify-center {klazz}">
  <SvgSpinner class="{size} animate-spin text-gray-400" />
  {#if label}
    <span class="text-sm text-gray-400 ml-2">{label}</span>
  {/if}
</div>
