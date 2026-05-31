<script lang="ts">
    import CodeMirror from './CodeMirror.svelte';
    import { minimalSetup } from 'codemirror';
    import { jinja } from '@codemirror/lang-jinja';
    import { EditorView } from '@codemirror/view';
    import { extractVariables } from '$lib/stores/templates';
    import { unifiedMergeView } from '@codemirror/merge';

    interface Props {
        /** Raw Jinja text to display. */
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

    // Extract variables reactively
    let variables = $derived(extractVariables(value));

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

<!-- Variables bar -->
<!-- TODO: move this to a bindable (need to make the parent display them) -->
{#if variables.length > 0}
    <div class="mt-2 flex flex-wrap items-center gap-1.5">
        <span class="text-xs text-gray-500">Variables:</span>
        {#each variables as v (v)}
            <code class="rounded bg-accent/10 px-1.5 py-0.5 font-mono text-xs text-accent">{v}</code
            >
        {/each}
    </div>
{/if}
