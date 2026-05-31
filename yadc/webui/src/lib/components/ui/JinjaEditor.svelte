<script lang="ts">
    import CodeMirror from './CodeMirror.svelte';
    import { minimalSetup } from 'codemirror';
    import { jinja } from '@codemirror/lang-jinja';
    import { EditorView } from '@codemirror/view';
    import { extractVariables } from '$lib/stores/templates';
    import { unifiedMergeView } from '@codemirror/merge';

    interface Props {
        /** Raw Jinja text. Bindable — use `bind:value` for two-way sync. */
        value?: string;
        /** Whether the editor is editable. Defaults to true. */
        editable?: boolean;
        /** Called when the document text changes (only when editable). Useful for side effects like dirty flags. */
        onchange?: (value: string) => void;
        class?: string;
        /** When set, renders a unified diff between original and value. Overrides editable to false. */
        original?: string;
        /** When true, editor height auto-sizes to content instead of filling container. */
        autoHeight?: boolean;
        /** When true (diff mode only), collapse unchanged regions to show only changed lines with small margin. */
        compactDiff?: boolean;
        /** Bindable — extracted template variables. Updated when value changes. */
        variables?: string[];
    }

    let {
        value = $bindable(''),
        editable = true,
        onchange,
        class: klazz = '',
        original,
        autoHeight = false,
        compactDiff = false,
        variables = $bindable<string[]>([])
    }: Props = $props();

    // Sync variables whenever value changes
    $effect(() => {
        variables = extractVariables(value);
    });

    let extensions = $derived.by(() => {
        const extensions = [minimalSetup, jinja(), EditorView.lineWrapping];

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

    function handleChange(text: string) {
        value = text;
        onchange?.(text);
    }
</script>

<!-- Editor -->
<CodeMirror
    class={klazz}
    doc={value}
    {extensions}
    editable={editable && original === undefined}
    onchange={handleChange}
    {autoHeight}
/>
