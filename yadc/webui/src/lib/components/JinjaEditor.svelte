<script lang="ts">
  import CodeMirror from "./CodeMirror.svelte";
  import { minimalSetup } from "codemirror";
  import { jinja } from "@codemirror/lang-jinja";
  import { EditorView } from "@codemirror/view";
  import { extractVariables } from "$lib/stores/templates";

  interface Props {
    value: string;
    onchange: (value: string) => void;
    readonly?: boolean;
    class?: string;
  }

  let { value = $bindable(), onchange, readonly = false, class: klazz = "" }: Props = $props();

  // Extract variables reactively
  let variables = $derived(extractVariables(value));

  // CodeMirror extensions
  let extensions = $derived([
    minimalSetup,
    jinja(),
    EditorView.lineWrapping,
    // Custom theme for the dark background
    EditorView.theme({
      "&": {
        fontSize: "0.875rem",
        borderRadius: "0.5rem",
        border: "1px solid var(--color-border)",
      },
      ".cm-content": {
        fontFamily: "'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace",
        padding: "0.5rem 0",
      },
      ".cm-focused": {
        outline: "2px solid var(--color-accent)",
        outlineOffset: "-1px",
      },
      "&.cm-editor": {
        background: "var(--color-bg)",
        color: "var(--color-text)",
        minHeight: "200px",
      },
      ".cm-gutters": {
        background: "var(--color-surface)",
        borderRight: "1px solid var(--color-border)",
        color: "var(--color-text-dim)",
      },
    }),
  ]);

  function handleChange(text: string) {
    value = text;
    onchange(text);
  }
</script>

<div class="flex flex-col {klazz}">
  <!-- Editor -->
  <CodeMirror doc={value} {extensions} {readonly} onchange={handleChange} />

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
