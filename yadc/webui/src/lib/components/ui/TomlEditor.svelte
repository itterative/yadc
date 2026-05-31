<script lang="ts">
    import CodeMirror from './CodeMirror.svelte';
    import { minimalSetup } from 'codemirror';
    import { EditorView } from '@codemirror/view';
    import { StreamLanguage } from '@codemirror/language';
    import { toml } from '@codemirror/legacy-modes/mode/toml';
    import { unifiedMergeView } from '@codemirror/merge';

    interface Props {
        /** Raw TOML text to display. */
        value: string;
        /** Whether the editor is editable. Defaults to true. */
        editable?: boolean;
        /** Called when the document text changes (only when editable). */
        onchange?: (value: string) => void;
        class?: string;
        /** When set, renders a unified diff between original and value. Overrides editable to false. */
        original?: string;
        /** When true, editor height auto-sizes to content instead of filling container. */
        autoHeight?: boolean;
        /** When true (diff mode only), collapse unchanged regions to show only changed lines with small margin. */
        compactDiff?: boolean;
    }

    let {
        value,
        editable = true,
        onchange,
        class: klazz = '',
        original,
        autoHeight = false,
        compactDiff = false
    }: Props = $props();

    let extensions = $derived.by(() => {
        const extensions = [minimalSetup, StreamLanguage.define(toml), EditorView.lineWrapping];

        if (original !== undefined) {
            extensions.push([
                unifiedMergeView({
                    original: original!,
                    highlightChanges: false,
                    gutter: false,
                    mergeControls: false,
                    ...(compactDiff
                        ? {
                              collapseUnchanged: {
                                  margin: 2,
                                  minSize: 1
                              }
                          }
                        : {})
                })
            ]);
        }

        return extensions;
    });
</script>

<CodeMirror
    doc={value}
    {extensions}
    editable={editable && original === undefined}
    {onchange}
    class={klazz}
    {autoHeight}
/>
