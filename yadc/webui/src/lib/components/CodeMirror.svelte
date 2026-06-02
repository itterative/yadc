<!--
  Minimal CodeMirror 6 wrapper for Svelte 5.
  Based on the public-domain shim, converted to Svelte 5 runes API.
-->

<script lang="ts">
  import { EditorView, minimalSetup } from "codemirror";
  import { StateEffect } from "@codemirror/state";
  import type { Extension } from "@codemirror/state";

  interface Props {
    /** Document text. Setting this after mount overwrites the editor content. */
    doc?: string;
    /** CodeMirror extensions (e.g. language support, theme). */
    extensions?: Extension[];
    /** Called when the document text changes. */
    onchange?: (value: string) => void;
    /** Whether the editor is read-only. */
    readonly?: boolean;
    /** CSS class applied to the wrapper div. */
    class?: string;
  }

  let {
    doc = "",
    extensions = [minimalSetup],
    onchange,
    readonly = false,
    class: klazz = "",
  }: Props = $props();

  let dom: HTMLDivElement | undefined = $state();
  let view: EditorView | null = $state(null);

  // --- Create editor once when DOM is ready ---

  $effect(() => {
    if (dom && !view) {
      view = new EditorView({
        doc: "",
        extensions: [],
        parent: dom,
        dispatchTransactions: handleTransactions,
      });
    }
  });

  // --- Destroy on unmount ---

  $effect(() => {
    return () => {
      view?.destroy();
    };
  });

  // --- Sync doc from outside ---

  let prevDoc: string | undefined = $state();
  $effect(() => {
    const currentDoc = doc;
    if (!view) return;
    if (currentDoc !== prevDoc && currentDoc !== view.state.doc.toString()) {
      view.dispatch({
        changes: { from: 0, to: view.state.doc.length, insert: currentDoc },
      });
    }
    prevDoc = currentDoc;
  });

  // --- Sync extensions + readonly ---

  $effect(() => {
    const exts = [...extensions, EditorView.editable.of(!readonly)];
    if (view) {
      view.dispatch({
        effects: StateEffect.reconfigure.of(exts),
      });
    }
  });

  // --- Transaction handler ---

  function handleTransactions(trs: readonly import("@codemirror/state").Transaction[]) {
    if (!view) return;
    view.update(trs);
    const lastChange = trs.findLast((tr) => tr.docChanged);
    if (lastChange) {
      const text = lastChange.newDoc.toString();
      prevDoc = text;
      onchange?.(text);
    }
  }
</script>

<div class="codemirror-wrapper {klazz}" bind:this={dom}></div>

<style>
  .codemirror-wrapper {
    display: contents;
  }
</style>
