<script lang="ts">
    import PillTabs from '$lib/components/ui/tabs/PillTabs.svelte';
    import Tab from '$lib/components/ui/tabs/Tab.svelte';
    import SidePanel from '$lib/components/ui/SidePanel.svelte';
    import FabButton from '$lib/components/ui/FabButton.svelte';
    import SvgClose from '$lib/icons/SvgClose.svelte';
    import SvgMenuLeft from '$lib/icons/SvgMenuLeft.svelte';
    import SvgSave from '$lib/icons/SvgSave.svelte';
    import SvgSparkle from '$lib/icons/SvgSparkle.svelte';
    import PromptForm from './PromptForm.svelte';
    import PromptHistoryPanel from './PromptHistoryPanel.svelte';
    import PromptSettings from './PromptSettings.svelte';
    import { generation, saveHistoryEntry, cancelGenerationState } from '$lib/stores/prompts';
    import type {
        ExamplePair,
        PromptGenFocus,
        PromptGenMode,
        PromptHistoryEntry,
        PromptImageQuality
    } from '$lib/stores/prompts';
    import { friendlyErrorMessage } from '$lib/api';
    import { toast } from '$lib/stores/toasts';

    type ActiveTab = 'generate' | 'settings' | 'history';

    interface Props {
        // Form state — bindable so the host owns the actual `$state`
        // and the persistence `$effect` (mirrors into `promptSettings`)
        // stays co-located with `EditTemplateDialog`. Bindings pass
        // through this panel to `PromptForm` / `PromptSettings`.
        mode: PromptGenMode;
        intent: string;
        focus: PromptGenFocus;
        examples: ExamplePair[];
        templateContent: string;
        env: string;
        apiUrl: string;
        apiToken: string;
        apiModelName: string;
        maxTokens: number;
        imageQuality: PromptImageQuality;

        // Panel UI state — bindable. `open` lets the mobile X close
        // button (inside the PillTabs `end` snippet) mutate it. The
        // host doesn't need to read or write it — desktop visibility
        // is unconditional (handled by the generic SidePanel via
        // `lg:translate-x-0`).
        open?: boolean;
        activeTab?: ActiveTab;

        // Hooks
        onchange?: (field: string, value: unknown) => void;
        /** Called when the user clicks the Generate / Refine button. */
        ongenerate?: () => void;
    }

    let {
        mode = $bindable('generate'),
        intent = $bindable(''),
        focus = $bindable('both'),
        examples = $bindable([]),
        templateContent = $bindable(''),
        env = $bindable('default'),
        apiUrl = $bindable(''),
        apiToken = $bindable(''),
        apiModelName = $bindable(''),
        maxTokens = $bindable(16384),
        imageQuality = $bindable('auto'),
        open = $bindable(false),
        activeTab = $bindable<ActiveTab>('generate'),
        onchange = () => {},
        ongenerate
    }: Props = $props();

    // History panel ref + save/restore state live here because the
    // Save-to-history button and the history list are both inside
    // this panel.
    let historyPanel: PromptHistoryPanel | undefined = $state();
    let isSaving = $state(false);

    // Derive streaming / validity state directly from the store so
    // the button disable logic doesn't need a round-trip through
    // the host.
    let isStreaming = $derived(generation.status === 'streaming');
    let canGenerate = $derived.by(() => {
        if (isStreaming || env.trim().length === 0 || intent.trim().length === 0) {
            return false;
        }
        if (mode === 'refine' && templateContent.trim().length === 0) {
            return false;
        }
        return true;
    });
    let canSave = $derived(
        intent.trim().length > 0 && (mode === 'generate' || templateContent.trim().length > 0)
    );

    function handleCloseClick() {
        // Mobile-only X inside the tab bar. Mutating ``open`` propagates
        // through ``bind:open`` to the generic SidePanel.
        open = false;
    }

    async function handleSaveToHistory() {
        if (!canSave || isSaving) {
            return;
        }
        isSaving = true;
        try {
            await saveHistoryEntry({
                mode,
                intent,
                focus,
                examples: examples.map((e) => ({ ...e })),
                template_content: mode === 'refine' ? templateContent : null
            });
            toast.success('Saved to history');
            // Refresh the history panel so the new entry shows up
            // when the user switches to the History tab.
            await historyPanel?.refresh();
        } catch (e) {
            toast.error(friendlyErrorMessage(e, 'Failed to save to history'));
        } finally {
            isSaving = false;
        }
    }

    function handleRestore(entry: PromptHistoryEntry) {
        // Jump to the Generate tab so the user immediately sees the
        // restored form values rather than the history list.
        activeTab = 'generate';
        mode = entry.mode;
        intent = entry.intent;
        focus = entry.focus;
        examples = entry.examples.map((e) => ({ ...e }));
        templateContent = entry.template_content ?? '';
        toast.success('Restored from history');
    }
</script>

<SidePanel bind:open>
    {#snippet fab()}
        <!-- Context-sensitive FAB. While streaming, the only useful
             action is cancel (the form is snapshotted into the
             in-flight call, so opening the panel mid-stream is
             pointless); otherwise it's the default panel toggle.
             Shape, positioning, and click-containment come from
             ``<FabButton>`` + the ``ui/SidePanel`` wrapper. -->
        {#if isStreaming}
            <FabButton
                icon={SvgClose}
                variant="error"
                label="Cancel generation"
                onclick={cancelGenerationState}
            />
        {:else}
            <FabButton icon={SvgMenuLeft} label="Toggle panel" onclick={() => (open = !open)} />
        {/if}
    {/snippet}
    <div class="flex h-full min-h-0 flex-col">
        <PillTabs bind:value={activeTab} class="min-h-0 flex-1">
            {#snippet end()}
                <button
                    class="cursor-pointer p-1 text-gray-400 transition-colors hover:text-white lg:hidden"
                    onclick={handleCloseClick}
                    title="Close panel"
                    aria-label="Close panel"
                >
                    <SvgClose class="h-5 w-5" />
                </button>
            {/snippet}
            <Tab id="generate" label="Generate" class="h-full overflow-y-auto">
                <div class="p-4">
                    <PromptForm
                        bind:mode
                        bind:intent
                        bind:focus
                        bind:examples
                        bind:templateContent
                        {onchange}
                    />
                </div>
            </Tab>
            <Tab id="settings" label="Settings" class="h-full overflow-y-auto">
                <div class="p-4">
                    <PromptSettings
                        bind:env
                        bind:apiUrl
                        bind:apiToken
                        bind:apiModelName
                        bind:maxTokens
                        bind:imageQuality
                        {onchange}
                    />
                </div>
            </Tab>
            <Tab id="history" label="History" class="h-full overflow-y-auto">
                <PromptHistoryPanel bind:this={historyPanel} onrestore={handleRestore} />
            </Tab>
        </PillTabs>

        <!-- Footer action bar. Generate / Refine is the primary CTA;
             Save-to-history is always available so the user can stash
             the form even mid-stream or after a partial response.
             Contextual status text on the left summarises the current
             tab. -->
        <div
            class="flex items-center justify-between border-t border-border bg-surface/50 px-4 py-2"
        >
            <p class="text-xs text-gray-500">
                {#if activeTab === 'history'}
                    {examples.length} example{examples.length === 1 ? '' : 's'} ready to save
                {:else if activeTab === 'settings'}
                    Environment &amp; generation limits
                {:else if examples.length > 0}
                    {examples.length} example{examples.length === 1 ? '' : 's'}
                {:else}
                    No examples — model will infer style from intent alone
                {/if}
            </p>
            <div class="flex items-center gap-2">
                <button
                    class="btn-secondary flex cursor-pointer items-center gap-1.5 text-sm disabled:cursor-not-allowed disabled:opacity-50"
                    onclick={handleSaveToHistory}
                    disabled={!canSave || isSaving}
                    title="Save current prompt to history"
                >
                    <SvgSave class="h-3.5 w-3.5" />
                    Save to history
                </button>
                <button
                    class="btn-primary flex cursor-pointer items-center gap-1.5 disabled:cursor-not-allowed disabled:opacity-50"
                    onclick={ongenerate}
                    disabled={!canGenerate}
                >
                    <SvgSparkle class="h-4 w-4" />
                    {isStreaming ? 'Streaming…' : mode === 'refine' ? 'Refine' : 'Generate'}
                </button>
            </div>
        </div>
    </div>
</SidePanel>
