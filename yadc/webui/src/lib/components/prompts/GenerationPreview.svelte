<script lang="ts">
    import SvgClose from '$lib/icons/SvgClose.svelte';
    import SvgCopy from '$lib/icons/SvgCopy.svelte';
    import SvgDelete from '$lib/icons/SvgDelete.svelte';
    import SvgSave from '$lib/icons/SvgSave.svelte';
    import SvgSpinner from '$lib/icons/SvgSpinner.svelte';
    import ActionBar from '$lib/components/ui/ActionBar.svelte';
    import ActionBarItem from '$lib/components/ui/ActionBarItem.svelte';
    import Card from '$lib/components/ui/Card.svelte';
    import { autoscroll } from '$lib/actions/autoscroll';
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

<!-- No card chrome on the panel itself — the streamed text is full-bleed
     and the footer (variables + actions) is the only framed element.

     Responsive scroll model: on mobile the OUTER container is the scroll
     area (block flow, body + footer scroll together as one content area —
     no inner "box" scroll now that the panel has no frame); on desktop the
     body is the inner scroll container and the footer is pinned below it
     (flex column). The ``use:autoscroll`` action resolves its target
     (self if scrollable, else nearest scrollable ancestor) so it follows
     the stream in both layouts. ``pb-24 lg:pb-0`` keeps the footer above
     the side-panel FAB on mobile (it floats ``bottom-6 right-6``); no
     padding on desktop (the FAB is ``lg:hidden``). -->
<div
    class="h-full overflow-y-auto pb-24 lg:flex lg:min-h-0 lg:flex-col lg:overflow-visible lg:pb-0"
>
    <!-- Body -->
    <div class="mb-4 lg:min-h-0 lg:flex-1 lg:overflow-auto lg:p-4" use:autoscroll>
        <ReasoningCard />
        {#if generation.status === 'idle' && !generation.body}
            <div
                class="flex min-h-[50vh] items-center justify-center text-sm text-gray-500 lg:h-full"
            >
                <p>The generated template will stream here.</p>
            </div>
        {:else if generation.status === 'error'}
            <div
                class="flex min-h-[50vh] flex-col items-center justify-center gap-2 text-sm text-error lg:h-full"
            >
                <p class="font-medium">Generation failed</p>
                <p class="text-center text-xs text-gray-400">
                    {generation.error ?? 'Unknown error'}
                </p>
            </div>
        {:else}
            {#if generation.body}
                <!-- NOTE: a bit ugly, but necessary to keep the whitespace of the template -->
                <pre
                    class="w-full font-mono text-sm leading-relaxed wrap-break-word whitespace-pre-wrap text-gray-200 lg:p-2">{generation.body}<span
                        class="animate-pulse text-accent"
                        class:hidden={generation.status !== 'streaming'}>▍</span
                    ></pre>
            {/if}

            <!-- Inline status, right under the stream. Same shape for
                 all terminal / in-progress states; error has its own
                 full-state UI above. -->
            <div class="mt-2 flex items-center justify-center gap-1.5 text-sm">
                {#if generation.status === 'streaming' && !generation.body}
                    <SvgSpinner class="h-3.5 w-3.5 animate-spin text-accent" />
                    <span class="text-accent">Generating…</span>
                {:else if generation.status === 'cancelled'}
                    <SvgClose class="h-3.5 w-3.5 text-warning" />
                    <span class="text-warning">Cancelled</span>
                {/if}
            </div>
        {/if}
    </div>

    <!-- Footer card (terminal state only — body present and stream not
         live): variables chip strip + actions. The only framed element
         in the panel; full-width labeled tap targets. ``SvgDelete`` +
         "Clear" replaces the old ``SvgClose`` (which read as "close
         panel", not "clear"). -->
    {#if generation.body && generation.status !== 'streaming'}
        <Card class="mx-4 lg:mx-auto lg:w-full lg:max-w-xl">
            {#if variables.length > 0}
                <div class="px-3 pt-2.5 pb-1.5 text-xs">
                    <div class="flex flex-wrap items-center gap-1.5">
                        <span class="text-gray-500">Variables:</span>
                        {#each variables as v (v)}
                            <code class="rounded bg-accent/10 px-1.5 py-0.5 font-mono text-accent"
                                >{v}</code
                            >
                        {/each}
                    </div>
                </div>
            {/if}
            <ActionBar>
                <ActionBarItem
                    onclick={() => onsavetemplate(generation.body)}
                    disabled={!canSave}
                    icon={SvgSave}
                    variant="primary">Save</ActionBarItem
                >
                <ActionBarItem onclick={copyBody} icon={SvgCopy} variant="secondary">
                    {copied ? 'Copied' : 'Copy'}
                </ActionBarItem>
                <ActionBarItem onclick={reset} icon={SvgDelete} variant="danger"
                    >Clear</ActionBarItem
                >
            </ActionBar>
        </Card>
    {/if}
</div>
