<script lang="ts">
    import CodeMirror from './CodeMirror.svelte';
    import { minimalSetup } from 'codemirror';
    import { jinja } from '@codemirror/lang-jinja';
    import { EditorView } from '@codemirror/view';
    import { extractVariables } from '$lib/stores/templates';

    interface Props {
        value: string;
        onchange: (value: string) => void;
        editable?: boolean;
        class?: string;
    }

    let { value = $bindable(), onchange, editable = true, class: klazz = '' }: Props = $props();

    // Extract variables reactively
    let variables = $derived(extractVariables(value));

    let extensions = $derived([minimalSetup, jinja(), EditorView.lineWrapping]);

    function handleChange(text: string) {
        value = text;
        onchange(text);
    }
</script>

<div class="flex h-full min-h-0 flex-col {klazz}">
    <!-- Editor -->
    <CodeMirror
        doc={value}
        {extensions}
        {editable}
        onchange={handleChange}
        class="min-h-0 flex-1"
    />

    <!-- Variables bar -->
    {#if variables.length > 0}
        <div class="mt-2 flex flex-wrap items-center gap-1.5">
            <span class="text-xs text-gray-500">Variables:</span>
            {#each variables as v (v)}
                <code class="rounded bg-accent/10 px-1.5 py-0.5 font-mono text-xs text-accent"
                    >{v}</code
                >
            {/each}
        </div>
    {/if}
</div>
