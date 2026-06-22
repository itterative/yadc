<script lang="ts">
    import SvgClose from '$lib/icons/SvgClose.svelte';
    import SvgCopy from '$lib/icons/SvgCopy.svelte';
    import SvgSave from '$lib/icons/SvgSave.svelte';
    import SvgSpinner from '$lib/icons/SvgSpinner.svelte';
    import { extractVariables } from '$lib/stores/templates';
    import { generation, reset } from '$lib/stores/prompts';
    import { toast } from '$lib/stores/toasts';
    import ReasoningCard from './ReasoningCard.svelte';

    interface Props {
        /** Called when the user wants to save the generated body as a new template. */
        onsavetemplate: (body: string) => void;
    }

    let { onsavetemplate }: Props = $props();

    let variables = $derived(extractVariables(generation.body));
    let canSave = $derived(generation.status === 'done' && generation.body.trim().length > 0);
    let copied = $state(false);

    async function copyBody() {
        if (!generation.body) {
            return;
        }
        try {
            await navigator.clipboard.writeText(generation.body);
            copied = true;
            setTimeout(() => (copied = false), 1500);
        } catch {
            toast.error('Failed to copy to clipboard');
        }
    }
</script>

<div class="card flex h-full min-h-0 flex-col overflow-hidden">
    <!-- Header -->
    <div class="flex items-center justify-between border-b border-border px-4 py-2">
        <h3 class="text-sm font-medium text-gray-300">Generated template</h3>
        {#if generation.body && generation.status !== 'streaming'}
            <div class="flex items-center gap-1">
                <button
                    class="flex cursor-pointer items-center gap-1.5 rounded-md p-1.5 text-gray-400 transition-colors hover:bg-bg hover:text-white"
                    onclick={copyBody}
                    title={copied ? 'Copied!' : 'Copy to clipboard'}
                >
                    <SvgCopy class="h-4 w-4" />
                </button>
                <button
                    class="btn-primary flex cursor-pointer items-center gap-1.5 px-2.5 py-1 text-xs"
                    onclick={() => onsavetemplate(generation.body)}
                    disabled={!canSave}
                    title="Save as template"
                >
                    <SvgSave class="h-3.5 w-3.5" />
                    Save as template
                </button>
                <button
                    class="flex cursor-pointer items-center gap-1.5 rounded-md p-1.5 text-gray-400 transition-colors hover:bg-bg hover:text-white"
                    onclick={reset}
                    title="Clear preview"
                >
                    <SvgClose class="h-4 w-4" />
                </button>
            </div>
        {/if}
    </div>

    <!-- Body -->
    <div class="min-h-0 flex-1 overflow-auto bg-gray-900/40 p-4">
        <ReasoningCard />
        {#if generation.status === 'idle' && !generation.body}
            <div class="flex h-full items-center justify-center text-sm text-gray-500">
                <p>The generated template will stream here.</p>
            </div>
        {:else if generation.status === 'error'}
            <div class="flex h-full flex-col items-center justify-center gap-2 text-sm text-error">
                <p class="font-medium">Generation failed</p>
                <p class="text-center text-xs text-gray-400">
                    {generation.error ?? 'Unknown error'}
                </p>
            </div>
        {:else}
            <pre
                class="w-full font-mono text-sm leading-relaxed wrap-break-word whitespace-pre-wrap text-gray-200">{generation.body}{#if generation.status === 'streaming'}<span
                        class="animate-pulse text-accent">▍</span
                    >{/if}</pre>
        {/if}
    </div>

    <!-- Footer: status + cancel + variables -->
    {#if generation.status !== 'idle' || variables.length > 0}
        <div class="space-y-2 border-t border-border bg-surface/50 px-4 py-2 text-xs">
            {#if variables.length > 0}
                <div class="flex flex-wrap items-center gap-1.5">
                    <span class="text-gray-500">Variables:</span>
                    {#each variables as v (v)}
                        <code class="rounded bg-accent/10 px-1.5 py-0.5 font-mono text-accent"
                            >{v}</code
                        >
                    {/each}
                </div>
            {/if}
            {#if generation.status === 'streaming' || generation.status === 'error'}
                <div class="flex items-center gap-1.5 text-gray-400">
                    {#if generation.status === 'streaming'}
                        <SvgSpinner class="h-3.5 w-3.5 animate-spin text-accent" />
                        <span class="text-accent">Generating…</span>
                    {:else}
                        <span class="text-error">Error</span>
                    {/if}
                </div>
            {/if}
        </div>
    {/if}
</div>
