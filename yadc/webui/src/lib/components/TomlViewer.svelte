<script lang="ts">
  import CodeMirror from "./CodeMirror.svelte";
  import { minimalSetup } from "codemirror";
  import { EditorView } from "@codemirror/view";
  import { StreamLanguage } from "@codemirror/language";
  import { toml } from "@codemirror/legacy-modes/mode/toml";

  interface Props {
    /** Raw TOML text to display. */
    value: string;
    class?: string;
  }

  let { value, class: klazz = "" }: Props = $props();

  let extensions = $derived([
    minimalSetup,
    StreamLanguage.define(toml),
    EditorView.lineWrapping,
    EditorView.theme({
      "&": {
        fontSize: "0.75rem",
        borderRadius: "0.5rem",
        border: "1px solid var(--color-border)",
      },
      ".cm-content": {
        fontFamily: "'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace",
        padding: "0.5rem 0",
      },
      "&.cm-editor": {
        background: "var(--color-surface)",
        color: "var(--color-text)",
      },
      ".cm-gutters": {
        display: "none",
      },
      ".cm-cursor": {
        display: "none",
      },
    }),
  ]);
</script>

<CodeMirror doc={value} {extensions} readonly={true} class={klazz} />
