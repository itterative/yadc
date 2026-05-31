<script lang="ts">
    import type { Snippet } from 'svelte';
    import { page } from '$app/stores';
    import './layout.css';
    import SettingsDialog from '$lib/components/dialogs/SettingsDialog.svelte';
    import ExportDialog from '$lib/components/dialogs/ExportDialog.svelte';
    import PasswordPromptDialog from '$lib/components/dialogs/PasswordPromptDialog.svelte';
    import SvgMenu from '$lib/icons/SvgMenu.svelte';
    import SvgImage from '$lib/icons/SvgImage.svelte';
    import SvgFile from '$lib/icons/SvgFile.svelte';
    import SvgUpload from '$lib/icons/SvgUpload.svelte';
    import SvgSettings from '$lib/icons/SvgSettings.svelte';
    import Tooltip from '$lib/components/ui/Tooltip.svelte';
    import type { ExportResult } from '$lib/stores/configs';
    import { getTopbarContent } from '$lib/stores/topbar.svelte';
    import ToastContainer from '$lib/components/ui/ToastContainer.svelte';
    import ConfirmDialog from '$lib/components/ui/ConfirmDialog.svelte';
    import { toast } from '$lib/stores/toasts';
    import { captioningStatus, resumptionFailed } from '$lib/stores/events';
    import { sendNotification } from '$lib/notifications';
    import { settingsDialog } from '$lib/stores/settings';
    import { passwordPromptOpen, submitPassword, cancelPassword } from '$lib/stores/passwordPrompt';

    interface Props {
        children: Snippet;
    }

    let { children }: Props = $props();

    let currentHash = $derived($page.url.hash);
    let isDatasetPage = $derived(currentHash === '#/' || currentHash.startsWith('#/datasets/'));
    let isTemplatesPage = $derived(currentHash.startsWith('#/templates'));

    let showExport = $state(false);
    let sidebarOpen = $state(false);

    $effect(() => {
        document.getElementById('yadc-loading-screen')?.remove();
    });

    function handleExported(result: ExportResult) {
        toast.success(`Exported ${result.count} images → ${result.output}`);
    }

    // Fire a persistent warning toast when SSE resumption fails.
    // The inline banner on the dataset page handles the route-specific case;
    // this toast provides visibility on other pages.
    let prevResumptionFailed = $state(false);
    $effect(() => {
        if ($resumptionFailed && !prevResumptionFailed) {
            toast.warning('Some events may have been missed due to a disconnected event stream.', {
                action: { label: 'Dismiss', handler: () => {} }
            });
        }
        prevResumptionFailed = $resumptionFailed;
    });

    // Global browser notifications for captioning completion.
    // Tracked here (layout level) so notifications fire even if the user
    // navigated away from the dataset page.
    let prevCaptioningActive = $state(false);
    $effect(() => {
        const s = $captioningStatus;
        const isActive = s.status === 'running' || s.status === 'stopping';

        // Detect transition from active → terminal
        if (prevCaptioningActive && !isActive && s.dataset_name) {
            const hadErrors = s.errors > 0;
            if (s.status === 'error') {
                sendNotification({
                    title: `Captioning failed: ${s.dataset_name}`,
                    body: s.error ?? 'Unknown error',
                    tag: `caption-${s.dataset_name}`
                });
            } else if (hadErrors) {
                sendNotification({
                    title: `Captioning complete with errors: ${s.dataset_name}`,
                    body: `${s.processed}/${s.total} done, ${s.errors} errors`,
                    tag: `caption-${s.dataset_name}`
                });
            } else {
                sendNotification({
                    title: `Captioning complete: ${s.dataset_name}`,
                    body: `${s.processed}/${s.total} images captioned`,
                    tag: `caption-${s.dataset_name}`
                });
            }
        }
        prevCaptioningActive = isActive;
    });
</script>

<!-- Mobile backdrop scrim -->
{#if sidebarOpen}
    <!-- svelte-ignore a11y_click_events_have_key_events -->
    <!-- svelte-ignore a11y_no_static_element_interactions -->
    <div
        class="fixed inset-0 z-30 bg-black/50 md:hidden"
        onclick={() => (sidebarOpen = false)}
    ></div>
{/if}

<div class="flex h-screen overflow-hidden">
    <nav
        class="fixed top-0 bottom-0 left-0 z-40 flex w-56 -translate-x-full flex-col items-center border-r border-border bg-surface p-3 transition-transform duration-200 ease-in-out md:relative md:w-14 md:shrink-0 md:translate-x-0 md:items-stretch md:p-2 md:transition-none"
        class:translate-x-0={sidebarOpen}
    >
        <!-- Brand -->
        <a
            href="#/"
            class="mb-3 flex items-center gap-2 p-1 no-underline"
            onclick={() => (sidebarOpen = false)}
        >
            <img src="/android-chrome-192x192.png" alt="yadc" class="h-8 w-8 shrink-0 rounded-md" />
            <span class="text-[1.1rem] font-bold text-accent md:hidden">yadc</span>
        </a>

        <!-- Nav links -->
        <div class="flex w-full flex-col gap-1">
            <a
                href="#/"
                class="flex items-center gap-3 rounded-md p-2 no-underline transition-colors duration-200 {isDatasetPage
                    ? 'bg-bg text-fg'
                    : 'text-muted hover:bg-bg hover:text-fg'}"
                onclick={() => (sidebarOpen = false)}
            >
                <Tooltip label="Datasets" class="z-50" direction="right">
                    <SvgImage class="h-6 w-6 shrink-0" />
                </Tooltip>
                <span class="text-lg md:hidden">Datasets</span>
            </a>
            <a
                href="#/templates"
                class="flex items-center gap-3 rounded-md p-2 no-underline transition-colors duration-200 {isTemplatesPage
                    ? 'bg-bg text-fg'
                    : 'text-muted hover:bg-bg hover:text-fg'}"
                onclick={() => (sidebarOpen = false)}
            >
                <Tooltip label="Templates" class="z-50" direction="right">
                    <SvgFile class="h-6 w-6 shrink-0" />
                </Tooltip>
                <span class="text-lg md:hidden">Templates</span>
            </a>
        </div>

        <div class="flex-1"></div>

        <!-- Actions -->
        <div class="flex w-full flex-col gap-1">
            <Tooltip label="Export" class="z-50" direction="right">
                <button
                    class="flex w-full cursor-pointer items-center gap-2 rounded-md border-none bg-transparent p-2 text-base text-muted transition-colors duration-200 hover:bg-bg hover:text-fg"
                    onclick={() => (showExport = true)}
                >
                    <SvgUpload class="h-6 w-6 shrink-0" />
                    <span class="text-lg md:hidden">Export</span>
                </button>
            </Tooltip>
            <Tooltip label="Settings" class="z-50" direction="right">
                <button
                    class="flex w-full cursor-pointer items-center gap-2 rounded-md border-none bg-transparent p-2 text-base text-muted transition-colors duration-200 hover:bg-bg hover:text-fg"
                    onclick={() => settingsDialog.update((d) => ({ ...d, open: true }))}
                >
                    <SvgSettings class="h-6 w-6 shrink-0" />
                    <span class="text-lg md:hidden">Settings</span>
                </button>
            </Tooltip>
        </div>
    </nav>

    <div class="flex min-w-0 flex-1 flex-col">
        <header class="flex min-h-12 items-center gap-4 px-6 pt-4 md:empty:hidden">
            <button
                class="flex h-9 w-9 shrink-0 cursor-pointer items-center justify-center rounded-md border-none bg-transparent text-muted transition-colors duration-200 hover:bg-bg hover:text-fg md:hidden"
                onclick={() => (sidebarOpen = !sidebarOpen)}
                title="Menu"
            >
                <SvgMenu class="h-6 w-6" />
            </button>
            {#if getTopbarContent()}
                <div class="flex min-w-0 flex-1 items-center gap-3">
                    {@render getTopbarContent()!()}
                </div>
            {/if}
        </header>

        <main class="min-h-0 flex-1 overflow-y-auto p-6">
            {@render children()}
        </main>
    </div>
</div>

<SettingsDialog
    open={$settingsDialog.open}
    onclose={() => settingsDialog.update((d) => ({ ...d, open: false }))}
/>
<ExportDialog open={showExport} onclose={() => (showExport = false)} onexported={handleExported} />
<PasswordPromptDialog
    open={$passwordPromptOpen}
    onsubmit={submitPassword}
    oncancel={cancelPassword}
/>
<ConfirmDialog />
<ToastContainer />
