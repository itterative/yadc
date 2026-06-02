<script lang="ts">
  import { extractVariables } from "$lib/stores/templates";

  interface Props {
    value: string;
    onchange: (value: string) => void;
    readonly?: boolean;
    class?: string;
  }

  let { value = $bindable(), onchange, readonly = false, class: klazz = "" }: Props = $props();

  let textareaEl: HTMLTextAreaElement | null = $state(null);

  // Extract variables reactively
  let variables = $derived(extractVariables(value));

  // Sync value changes upward
  function handleInput(e: Event) {
    const target = e.target as HTMLTextAreaElement;
    value = target.value;
    onchange(target.value);
  }

  // Tab key inserts 2 spaces
  function handleKeydown(e: KeyboardEvent) {
    if (e.key === "Tab") {
      e.preventDefault();
      const ta = e.target as HTMLTextAreaElement;
      const start = ta.selectionStart;
      const end = ta.selectionEnd;

      value = value.substring(0, start) + "  " + value.substring(end);
      onchange(value);

      // Restore cursor position after Svelte re-renders
      requestAnimationFrame(() => {
        ta.selectionStart = ta.selectionEnd = start + 2;
      });
    }
  }
</script>

<div class="flex flex-col {klazz}">
  <!-- Editor area -->
  <div class="relative flex-1 min-h-0">
    <textarea
      bind:this={textareaEl}
      {readonly}
      {value}
      oninput={handleInput}
      onkeydown={handleKeydown}
      spellcheck="false"
      class="w-full h-full min-h-[200px] rounded-lg bg-bg border border-border px-3 py-2 text-sm text-gray-200 font-mono resize-y focus:ring-2 focus:ring-accent focus:outline-none"
    ></textarea>
  </div>

  <!-- Variables bar -->
  {#if variables.length > 0}
    <div class="mt-2 flex flex-wrap gap-1.5 items-center">
      <span class="text-xs text-gray-500">Variables:</span>
      {#each variables as v (v)}
        <code class="text-xs px-1.5 py-0.5 rounded bg-accent/10 text-accent font-mono">{v}</code>
      {/each}
    </div>
  {/if}
</div>
