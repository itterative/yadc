<!--
  Minimal CodeMirror 6 wrapper for Svelte 5.
  Based on the public-domain shim, converted to Svelte 5 runes API.
-->

<script lang="ts">
  import { EditorView, minimalSetup } from "codemirror";
  import { StateEffect, type Extension } from "@codemirror/state";
  import { HighlightStyle, syntaxHighlighting } from "@codemirror/language";
  import { tags } from "@lezer/highlight";

  // Prevent Vite's SSR tree-shaking from flagging StateEffect as unused.
  // It's used inside $effect bodies that get stripped during SSR compilation.
  void StateEffect.reconfigure;

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
      color: "var(--color-fg)",
      height: "100%",
    },
    ".cm-focused": {
      outline: "2px solid var(--color-accent)",
      outlineOffset: "-1px",
    },
    ".cm-cursor": {
      borderLeftColor: "var(--color-fg)",
    },
    ".cm-gutters": {
      background: "var(--color-surface)",
      borderRight: "1px solid var(--color-border)",
      color: "var(--color-muted)",
    },
    ".cm-selectionBackground": {
      background: "color-mix(in oklch, var(--color-accent) 40%, transparent) !important",
    },
    "&.cm-focused .cm-selectionBackground": {
      background: "color-mix(in oklch, var(--color-accent) 50%, transparent) !important",
    },
  });

  // --- Dark syntax highlighting (Tokyo Night–inspired) ---

  const darkHighlightStyle = HighlightStyle.define([
    { tag: tags.keyword, color: "var(--color-syn-keyword)" },
    { tag: tags.controlKeyword, color: "var(--color-syn-keyword)", fontStyle: "italic" },
    { tag: tags.definitionKeyword, color: "var(--color-accent)" },
    { tag: tags.moduleKeyword, color: "var(--color-accent)" },
    { tag: tags.operatorKeyword, color: "var(--color-syn-keyword)" },
    { tag: tags.comment, color: "var(--color-syn-comment)", fontStyle: "italic" },
    { tag: tags.string, color: "var(--color-syn-string)" },
    { tag: tags.special(tags.string), color: "var(--color-syn-property)" },
    { tag: tags.character, color: "var(--color-syn-string)" },
    { tag: tags.number, color: "var(--color-syn-number)" },
    { tag: tags.integer, color: "var(--color-syn-number)" },
    { tag: tags.float, color: "var(--color-syn-number)" },
    { tag: tags.bool, color: "var(--color-syn-number)" },
    { tag: tags.null, color: "var(--color-syn-number)" },
    { tag: tags.typeName, color: "var(--color-syn-type)" },
    { tag: tags.variableName, color: "var(--color-fg)" },
    { tag: tags.definition(tags.variableName), color: "var(--color-accent)" },
    { tag: tags.propertyName, color: "var(--color-syn-property)" },
    { tag: tags.function(tags.variableName), color: "var(--color-accent)" },
    { tag: tags.labelName, color: "var(--color-syn-label)" },
    { tag: tags.literal, color: "var(--color-syn-number)" },
    { tag: tags.escape, color: "var(--color-syn-escape)" },
    { tag: tags.meta, color: "var(--color-syn-comment)" },
    { tag: tags.invalid, color: "var(--color-syn-invalid)" },
    { tag: tags.special(tags.variableName), color: "var(--color-syn-type)" },
    { tag: tags.separator, color: "var(--color-syn-punctuation)" },
    { tag: tags.punctuation, color: "var(--color-syn-punctuation)" },
    { tag: tags.bracket, color: "var(--color-syn-punctuation)" },
    { tag: tags.angleBracket, color: "var(--color-syn-punctuation)" },
    { tag: tags.regexp, color: "var(--color-syn-regexp)" },
    { tag: tags.color, color: "var(--color-syn-number)" },
    { tag: tags.content, color: "var(--color-fg)" },
    { tag: tags.contentSeparator, color: "var(--color-syn-comment)" },
    { tag: tags.heading, color: "var(--color-accent)", fontWeight: "bold" },
    { tag: tags.link, color: "var(--color-accent)", textDecoration: "underline" },
    { tag: tags.emphasis, fontStyle: "italic" },
    { tag: tags.strong, fontWeight: "bold" },
    { tag: tags.strikethrough, textDecoration: "line-through" },
    { tag: tags.inserted, color: "var(--color-success)" },
    { tag: tags.deleted, color: "var(--color-error)" },
    { tag: tags.changed, color: "var(--color-syn-label)" },
  ]);

  const darkHighlightExt = syntaxHighlighting(darkHighlightStyle);

  $effect(() => {
    const exts = [baseTheme, darkHighlightExt, ...extensions, EditorView.editable.of(editable)];
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
