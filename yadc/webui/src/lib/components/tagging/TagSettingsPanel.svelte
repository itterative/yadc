<script lang="ts">
    import { startBatchTagging, stopTagging, taggingStatuses } from '$lib/stores/tagging';
    import CompactPillTabs from '$lib/components/ui/tabs/CompactPillTabs.svelte';
    import Tab from '$lib/components/ui/tabs/Tab.svelte';
    import SvgSettings from '$lib/icons/SvgSettings.svelte';
    import SvgStar from '$lib/icons/SvgStar.svelte';
    import TagSettings from './TagSettings.svelte';
    import TagCustomize from './TagCustomize.svelte';
    import { toast } from '$lib/stores/toasts';
    import { friendlyErrorMessage } from '$lib/api';

    interface Props {
        /** Which dataset this is for. */
        datasetName: string;
        /** Called when the panel wants to close (e.g. after starting/stopping). */
        onclose?: () => void;
    }

    let { datasetName, onclose }: Props = $props();

    // Derive batch tagging state from the per-dataset status map so this
    // panel reflects only this dataset's job.
    let isBatchTagging = $derived.by(() => {
        const s = $taggingStatuses.get(datasetName);
        return s?.status === 'running' || s?.status === 'stopping';
    });

    // Inner sub-tab (Settings / Customize). The footer Start/Stop button
    // stays visible on both — it's the panel's primary, dataset-level action.
    let activeTab = $state('settings');

    let isCancelling = $state(false);

    async function handleStart() {
        try {
            await startBatchTagging(datasetName);
        } catch (e) {
            toast.error(friendlyErrorMessage(e, 'Failed to start tagging'));
            return;
        }
        onclose?.();
    }

    async function handleStop() {
        isCancelling = true;
        try {
            await stopTagging(datasetName);
        } finally {
            isCancelling = false;
        }
    }
</script>

<div class="flex h-full flex-col">
    <div class="flex-1 space-y-5 overflow-y-auto p-4">
        <CompactPillTabs
            storageId="dataset/sidePanel/tagSettings"
            bind:value={activeTab}
            class="h-full"
        >
            <Tab id="settings" label="Settings" icon={SvgSettings} class="h-full">
                <TagSettings />
            </Tab>
            <Tab id="customize" label="Customize" icon={SvgStar} class="h-full">
                <TagCustomize />
            </Tab>
        </CompactPillTabs>
    </div>

    <!-- Footer: sticky start/stop button -->
    <div class="shrink-0 border-t border-border p-4">
        {#if isBatchTagging || isCancelling}
            <button
                class="btn w-full border-error/40 bg-error/20 px-4 py-2 text-sm text-error hover:bg-error/30"
                onclick={handleStop}
                disabled={isCancelling}
            >
                {isCancelling ? 'Cancelling...' : 'Cancel'}
            </button>
        {:else}
            <button class="btn-primary w-full" onclick={handleStart}>Start</button>
        {/if}
    </div>
</div>
