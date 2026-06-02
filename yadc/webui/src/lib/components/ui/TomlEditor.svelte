<script lang="ts">
  import CodeMirror from "./CodeMirror.svelte";
  import { minimalSetup } from "codemirror";
  import { EditorView } from "@codemirror/view";
  import { StreamLanguage } from "@codemirror/language";
  import { toml } from "@codemirror/legacy-modes/mode/toml";

  interface Props {
    /** Raw TOML text to display. */
    value: string;
    /** Whether the editor is editable. Defaults to true. */
    editable?: boolean;
    /** Called when the document text changes (only when editable). */
    onchange?: (value: string) => void;
    class?: string;
  }

  let { value, editable = true, onchange, class: klazz = "" }: Props = $props();

  let extensions = $derived([
    minimalSetup,
    StreamLanguage.define(toml),
    EditorView.lineWrapping,
  ]);
</script>

<CodeMirror doc={value} {extensions} {editable} {onchange} class={klazz} />
