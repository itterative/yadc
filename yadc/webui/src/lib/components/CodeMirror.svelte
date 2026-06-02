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
    /** Whether the editor is editable. Defaults to true. */
    editable?: boolean;
    /** CSS class applied to the wrapper div. */
    class?: string;
  }

  let {
    doc = "",
    extensions = [minimalSetup],
    onchange,
    editable = true,
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

  // --- Sync extensions + editable ---

  // --- Base theme (dark background, uses Tailwind CSS variables) ---

  const baseTheme = EditorView.theme({
    "&": {
      borderRadius: "0.5rem",
      border: "1px solid var(--color-border)",
    },
    ".cm-content": {
      fontFamily: "var(--font-mono)",
      padding: "0.5rem 0",
    },
    "&.cm-editor": {
      background: "var(--color-surface)",
      color: "var(--color-text)",
      height: "100%",
    },
    ".cm-focused": {
      outline: "2px solid var(--color-accent)",
      outlineOffset: "-1px",
    },
    ".cm-cursor": {
      borderLeftColor: "var(--color-text)",
    },
    ".cm-gutters": {
      background: "var(--color-surface)",
      borderRight: "1px solid var(--color-border)",
      color: "var(--color-text-dim)",
    },
  });

  $effect(() => {
    const exts = [baseTheme, ...extensions, EditorView.editable.of(editable)];
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
    height: 100%;
    overflow: hidden;
  }
</style>
