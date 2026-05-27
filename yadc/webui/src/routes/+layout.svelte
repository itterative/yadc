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
    <div class="sidebar-scrim" onclick={() => (sidebarOpen = false)}></div>
{/if}

<div class="app-shell">
    <nav class="app-sidebar" class:open={sidebarOpen}>
        <!-- Brand -->
        <div class="sidebar-brand">
            <a href="#/" onclick={() => (sidebarOpen = false)}>
                <img src="/android-chrome-192x192.png" alt="yadc" class="brand-icon" />
                <span class="brand-text">yadc</span>
            </a>
        </div>

        <!-- Nav links -->
        <div class="sidebar-links">
            <a href="#/" class:active={isDatasetPage} onclick={() => (sidebarOpen = false)}>
                <Tooltip label="Datasets">
                    <SvgImage class="nav-icon" />
                </Tooltip>
                <span class="nav-label">Datasets</span>
            </a>
            <a
                href="#/templates"
                class:active={isTemplatesPage}
                onclick={() => (sidebarOpen = false)}
            >
                <Tooltip label="Templates">
                    <SvgFile class="nav-icon" />
                </Tooltip>
                <span class="nav-label">Templates</span>
            </a>
        </div>

        <div class="sidebar-spacer"></div>

        <!-- Actions -->
        <div class="sidebar-actions">
            <Tooltip label="Export">
                <button class="nav-btn" onclick={() => (showExport = true)}>
                    <SvgUpload class="nav-icon" />
                    <span class="nav-label">Export</span>
                </button>
            </Tooltip>
            <Tooltip label="Settings">
                <button
                    class="nav-btn"
                    onclick={() => settingsDialog.update((d) => ({ ...d, open: true }))}
                >
                    <SvgSettings class="nav-icon" />
                    <span class="nav-label">Settings</span>
                </button>
            </Tooltip>
        </div>
    </nav>

    <div class="app-content">
        <header class="app-topbar">
            <button class="burger-btn" onclick={() => (sidebarOpen = !sidebarOpen)} title="Menu">
                <SvgMenu class="h-6 w-6" />
            </button>
            {#if getTopbarContent()}
                <div class="topbar-content">
                    {@render getTopbarContent()!()}
                </div>
            {/if}
        </header>

        <main class="app-main">
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
<ToastContainer />

<style>
    .app-shell {
        display: flex;
        height: 100vh;
        overflow: hidden;
    }

    /* --- Content area (topbar + main) --- */
    .app-content {
        display: flex;
        flex-direction: column;
        flex: 1;
        min-width: 0;
    }

    /* --- Topbar --- */
    .app-topbar {
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 1rem 1.5rem 0;
        min-height: 3rem;
    }

    /* Hide topbar on desktop when there's no content (burger is hidden too) */
    @media (min-width: 768px) {
        .app-topbar:empty {
            display: none;
        }
    }

    /* --- Burger button (mobile only, inside topbar) --- */
    .burger-btn {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 2.25rem;
        height: 2.25rem;
        border-radius: 0.375rem;
        color: var(--color-muted);
        background: transparent;
        border: none;
        cursor: pointer;
        flex-shrink: 0;
        transition:
            color 0.2s,
            background-color 0.2s;
    }

    .burger-btn:hover {
        color: var(--color-fg);
        background: var(--color-bg);
    }

    @media (min-width: 768px) {
        .burger-btn {
            display: none;
        }
    }

    .topbar-content {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        min-width: 0;
        flex: 1;
    }

    /* --- Mobile backdrop scrim --- */
    .sidebar-scrim {
        position: fixed;
        inset: 0;
        z-index: 30;
        background: rgba(0, 0, 0, 0.5);
    }

    @media (min-width: 768px) {
        .sidebar-scrim {
            display: none;
        }
    }

    /* --- Sidebar --- */
    .app-sidebar {
        display: flex;
        flex-direction: column;
        align-items: center;
        background: var(--color-surface);
        border-right: 1px solid var(--color-border);
        /* Mobile: fixed overlay */
        position: fixed;
        top: 0;
        left: 0;
        bottom: 0;
        width: 14rem;
        z-index: 40;
        transform: translateX(-100%);
        transition: transform 0.2s ease-in-out;
        padding: 1rem 0.75rem;
    }

    .app-sidebar.open {
        transform: translateX(0);
    }

    @media (min-width: 768px) {
        .app-sidebar {
            position: relative;
            transform: none;
            transition: none;
            z-index: auto;
            width: 3.5rem;
            flex-shrink: 0;
            padding: 0.75rem 0.5rem;
            align-items: stretch;
        }
    }

    /* --- Brand --- */
    .sidebar-brand a {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        text-decoration: none;
        padding: 0.25rem;
        margin-bottom: 0.75rem;
    }

    .brand-icon {
        width: 2rem;
        height: 2rem;
        border-radius: 0.375rem;
        flex-shrink: 0;
    }

    .brand-text {
        font-weight: 700;
        font-size: 1.1rem;
        color: var(--color-accent);
    }

    @media (min-width: 768px) {
        .brand-text {
            display: none;
        }
    }

    /* --- Nav links --- */
    .sidebar-links {
        display: flex;
        flex-direction: column;
        gap: 0.25rem;
        width: 100%;
    }

    .sidebar-links a {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        color: var(--color-muted);
        text-decoration: none;
        padding: 0.5rem;
        border-radius: 0.375rem;
        transition:
            color 0.2s,
            background-color 0.2s;
    }

    .sidebar-links a:hover {
        color: var(--color-fg);
        background: var(--color-bg);
    }

    .sidebar-links a.active {
        color: var(--color-fg);
        background: var(--color-bg);
    }

    :global(.nav-icon) {
        width: 1.5rem;
        height: 1.5rem;
        flex-shrink: 0;
    }

    .nav-label {
        font-size: 1.125rem;
    }

    @media (min-width: 768px) {
        .nav-label {
            display: none;
        }
    }

    .sidebar-spacer {
        flex: 1;
    }

    /* --- Actions --- */
    .sidebar-actions {
        display: flex;
        flex-direction: column;
        gap: 0.25rem;
        width: 100%;
    }

    .nav-btn {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        width: 100%;
        padding: 0.75rem 0.5rem;
        border-radius: 0.375rem;
        color: var(--color-muted);
        background: transparent;
        border: none;
        cursor: pointer;
        font-size: 1rem;
        transition:
            color 0.2s,
            background-color 0.2s;
    }

    .nav-btn:hover {
        color: var(--color-fg);
        background: var(--color-bg);
    }

    /* --- Main content --- */
    .app-main {
        flex: 1;
        padding: 1.5rem;
        min-height: 0;
        overflow-y: auto;
    }
</style>
