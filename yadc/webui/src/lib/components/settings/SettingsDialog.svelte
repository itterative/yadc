<script lang="ts">
    import Dialog from '$lib/components/ui/Dialog.svelte';
    import PillTabs from '$lib/components/ui/tabs/PillTabs.svelte';
    import Tab from '$lib/components/ui/tabs/Tab.svelte';
    import SvgClose from '$lib/icons/SvgClose.svelte';
    import { settingsDialog } from '$lib/stores/settings';
    import GeneralSettings from '$lib/components/settings/GeneralSettings.svelte';
    import EnvironmentSettings from '$lib/components/env/EnvironmentSettings.svelte';
    import SecuritySettings from '$lib/components/settings/SecuritySettings.svelte';

    interface Props {
        open: boolean;
        onclose: () => void;
    }

    let { open, onclose }: Props = $props();

    // eslint-disable-next-line svelte/prefer-writable-derived
    let activeTab: 'general' | 'environments' | 'security' = $state('general');

    $effect(() => {
        activeTab = $settingsDialog.tab;
    });

    $effect(() => {
        if (activeTab !== $settingsDialog.tab) {
            settingsDialog.update((d) => ({ ...d, tab: activeTab }));
        }
    });
</script>

<Dialog class="dialog-panel flex max-h-[85vh] max-w-3xl flex-col overflow-hidden" {open} {onclose}>
    <!-- Header -->
    <div class="flex items-center justify-between p-5 pb-0">
        <h2 class="dialog-title">Settings</h2>
        <button class="btn-close" onclick={onclose}>
            <SvgClose class="h-5 w-5" />
        </button>
    </div>

    <PillTabs bind:value={activeTab} class="min-h-0 flex-1">
        <Tab id="general" label="General" class="overflow-y-auto">
            <GeneralSettings />
        </Tab>
        <Tab id="environments" label="Environments" class="overflow-y-auto">
            <EnvironmentSettings />
        </Tab>
        <Tab id="security" label="Security" class="overflow-y-auto">
            <SecuritySettings />
        </Tab>
    </PillTabs>
</Dialog>
