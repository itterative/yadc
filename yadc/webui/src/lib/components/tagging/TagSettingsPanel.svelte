<script lang="ts">
    import {
        ensureHighlightsLoaded,
        loadTagPolicy,
        startBatchTagging,
        stopTagging,
        taggingStatuses
    } from '$lib/stores/tagging';
    import CompactPillTabs from '$lib/components/ui/tabs/CompactPillTabs.svelte';
    import ActionBar from '$lib/components/ui/ActionBar.svelte';
    import ActionBarItem from '$lib/components/ui/ActionBarItem.svelte';
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

    // Hydrate the global highlights (one-shot cached promise — repeated
    // calls are no-ops) and the per-dataset policy whenever the
    // dataset changes. Both stores are mirror caches of backend state;
    // without a load here the PolicyList / TierList would render empty
    // until a manual mutation.
    $effect(() => {
        void datasetName;
        ensureHighlightsLoaded().catch((e) => {
            toast.error('Failed to load tag highlights', {
                details: [friendlyErrorMessage(e, 'request failed')]
            });
        });
        loadTagPolicy(datasetName).catch((e) => {
            toast.error('Failed to load tag policy for this dataset', {
                details: [friendlyErrorMessage(e, 'request failed')]
            });
        });
    });

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
    <div class="shrink-0 border-t border-border">
        <ActionBar>
            {#if isBatchTagging || isCancelling}
                <ActionBarItem variant="danger" onclick={handleStop} disabled={isCancelling}>
                    {isCancelling ? 'Cancelling...' : 'Cancel'}
                </ActionBarItem>
            {:else}
                <ActionBarItem variant="primary" onclick={handleStart}>Start</ActionBarItem>
            {/if}
        </ActionBar>
    </div>
</div>
