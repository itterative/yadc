<script lang="ts">
	import type { Snippet } from 'svelte';
	import { page } from '$app/stores';
	import './layout.css';
	import SettingsDialog from '$lib/components/dialogs/SettingsDialog.svelte';
	import ExportDialog from '$lib/components/dialogs/ExportDialog.svelte';
	import SvgUpload from '$lib/icons/SvgUpload.svelte';
	import SvgSettings from '$lib/icons/SvgSettings.svelte';
	import type { ExportResult } from '$lib/stores/configs';
	import ToastContainer from '$lib/components/ui/ToastContainer.svelte';
	import { toast } from '$lib/stores/toasts';
	import { captioningStatus, resumptionFailed } from '$lib/stores/events';
	import { sendNotification } from '$lib/notifications';
	import { settingsDialog } from '$lib/stores/settings';

	interface Props {
		children: Snippet;
	}

	let { children }: Props = $props();

	let currentHash = $derived($page.url.hash);
	let isDatasetPage = $derived(currentHash === '#/' || currentHash.startsWith('#/datasets/'));
	let isTemplatesPage = $derived(currentHash.startsWith('#/templates'));
	let datasetName = $derived(isDatasetPage ? $page.params.name : null);

	let showExport = $state(false);

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

<div class="app-shell">
	<nav class="app-nav max-sm:gap-3 max-sm:px-4">
		<div class="nav-brand">
			<a href="#/">yadc</a>
		</div>
		<div class="nav-links min-w-0 overflow-hidden max-sm:gap-3">
			<a href="#/" class:active={isDatasetPage}>Datasets</a>
			<a href="#/templates" class:active={isTemplatesPage}>Templates</a>
			{#if isDatasetPage && datasetName}
				<span class="nav-separator">/</span>
				<span class="nav-current min-w-0 truncate">{datasetName}</span>
			{/if}
		</div>
		<div class="nav-actions">
			<button class="nav-btn" onclick={() => (showExport = true)} title="Export">
				<SvgUpload class="h-6 w-6" />
			</button>
			<button
				class="nav-btn"
				onclick={() => settingsDialog.update((d) => ({ ...d, open: true }))}
				title="Settings"
			>
				<SvgSettings class="h-6 w-6" />
			</button>
		</div>
	</nav>

	<main class="app-main">
		{@render children()}
	</main>
</div>

<SettingsDialog
	open={$settingsDialog.open}
	onclose={() => settingsDialog.update((d) => ({ ...d, open: false }))}
/>
<ExportDialog open={showExport} onclose={() => (showExport = false)} onexported={handleExported} />
<ToastContainer />

<style>
	.app-shell {
		display: flex;
		flex-direction: column;
		height: 100vh;
		overflow: hidden;
	}

	.app-nav {
		display: flex;
		align-items: center;
		gap: 2rem;
		padding: 0.75rem 1.5rem;
		background: var(--color-surface);
		border-bottom: 1px solid var(--color-border);
	}

	.nav-brand a {
		font-weight: 700;
		font-size: 1.2rem;
		color: var(--color-accent);
		text-decoration: none;
	}

	.nav-links {
		display: flex;
		align-items: center;
		gap: 1.25rem;
		flex: 1;
	}

	.nav-links a {
		color: var(--color-muted);
		text-decoration: none;
		transition: color 0.2s;
	}

	.nav-links a:hover {
		color: var(--color-fg);
	}

	.nav-links a.active {
		color: var(--color-fg);
	}

	.nav-separator {
		color: var(--color-muted);
		opacity: 0.4;
	}

	.nav-current {
		color: var(--color-fg);
		font-size: 0.875rem;
	}

	.nav-actions {
		display: flex;
		align-items: center;
		gap: 0.75rem;
	}

	.nav-btn {
		display: flex;
		align-items: center;
		justify-content: center;
		width: 2rem;
		height: 2rem;
		border-radius: 0.375rem;
		color: var(--color-muted);
		background: transparent;
		border: none;
		cursor: pointer;
		transition:
			color 0.2s,
			background-color 0.2s;
	}

	.nav-btn:hover {
		color: var(--color-fg);
		background: var(--color-bg);
	}

	.app-main {
		flex: 1;
		padding: 1.5rem;
		min-height: 0;
		overflow-y: auto;
	}
</style>
