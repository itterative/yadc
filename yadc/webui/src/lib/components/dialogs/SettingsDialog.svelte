<script lang="ts">
	import Dialog from '$lib/components/ui/Dialog.svelte';
	import PillTabs from '$lib/components/ui/tabs/PillTabs.svelte';
	import Tab from '$lib/components/ui/tabs/Tab.svelte';
	import Checkbox from '$lib/components/ui/Checkbox.svelte';
	import SvgClose from '$lib/icons/SvgClose.svelte';
	import SvgDelete from '$lib/icons/SvgDelete.svelte';
	import SvgEdit from '$lib/icons/SvgEdit.svelte';
	import SvgPlus from '$lib/icons/SvgPlus.svelte';
	import ConfirmDelete from '$lib/components/ui/ConfirmDelete.svelte';
	import SpinnerBlock from '$lib/components/ui/SpinnerBlock.svelte';
	import { settings, settingsDialog } from '$lib/stores/settings';
	import { deleteEnv, fetchEnv, refreshEnvs, saveEnv, envs, type EnvInfo } from '$lib/stores/envs';
	import { friendlyErrorMessage } from '$lib/api';
	import {
		notificationsSupported,
		notificationPermission,
		requestNotificationPermission
	} from '$lib/notifications';

	interface Props {
		open: boolean;
		onclose: () => void;
	}

	let { open, onclose }: Props = $props();

	// --- Tabs ---
	// NOTE: disabled because we need two way binding
	// eslint-disable-next-line svelte/prefer-writable-derived
	let activeTab: 'general' | 'environments' = $state('general');

	$effect(() => {
		activeTab = $settingsDialog.tab;
	});

	$effect(() => {
		if (activeTab !== $settingsDialog.tab) {
			settingsDialog.update((d) => ({ ...d, tab: activeTab }));
		}
	});

	// --- Env management state ---
	let envLoading = $state(false);
	let envError: string | null = $state(null);
	let editingEnv: EnvInfo | null = $state(null);
	let isNewEnv = $state(false);
	let editName = $state('');
	let editUrl = $state('');
	let editToken = $state('');
	let editModelName = $state('');
	let isSavingEnv = $state(false);
	let saveEnvError: string | null = $state(null);
	let confirmDelete: string | null = $state(null);

	$effect(() => {
		if (open) {
			loadEnvs();
			editingEnv = null;
			isNewEnv = false;
			confirmDelete = null;
		}
	});

	async function loadEnvs() {
		envLoading = true;
		envError = null;
		try {
			await refreshEnvs();
		} catch (e) {
			envError = friendlyErrorMessage(e, 'Failed to load environments');
		} finally {
			envLoading = false;
		}
	}

	function startCreateEnv() {
		isNewEnv = true;
		editingEnv = null;
		editName = '';
		editUrl = '';
		editToken = '';
		editModelName = '';
		saveEnvError = null;
	}

	async function startEditEnv(name: string) {
		saveEnvError = null;
		try {
			const info = await fetchEnv(name);
			isNewEnv = false;
			editingEnv = info;
			editName = info.name;
			editUrl = info.api_url || '';
			editToken = '';
			editModelName = info.api_model_name || '';
		} catch (e) {
			envError = friendlyErrorMessage(e, 'Failed to load environment');
		}
	}

	function cancelEditEnv() {
		editingEnv = null;
		isNewEnv = false;
		saveEnvError = null;
	}

	async function handleSaveEnv() {
		const name = editName.trim();
		if (!name) {
			saveEnvError = 'Name is required';
			return;
		}

		isSavingEnv = true;
		saveEnvError = null;
		try {
			const data: Record<string, string> = { api_url: editUrl.trim() };
			if (editToken) {
				data.api_token = editToken;
			}
			if (editModelName.trim()) {
				data.api_model_name = editModelName.trim();
			}

			await saveEnv(name, data);
			editingEnv = null;
			isNewEnv = false;
			await loadEnvs();
		} catch (e) {
			saveEnvError = friendlyErrorMessage(e, 'Failed to save environment');
		} finally {
			isSavingEnv = false;
		}
	}

	async function handleDeleteEnv(name: string) {
		try {
			await deleteEnv(name);
			confirmDelete = null;
			await loadEnvs();
		} catch (e) {
			envError = friendlyErrorMessage(e, 'Failed to delete environment');
		}
	}

	// --- Notification state ---
	let permStatus = $state<NotificationPermission | 'unsupported'>('default');

	// Local state bound to checkbox; derived from tri-state store value
	let notificationsOn = $state(false);
	let previousStoreValue = $state<'unset' | 'enabled' | 'disabled'>('unset');

	// Sync from store → local state when the dialog opens or the store changes
	$effect(() => {
		const storeVal = $settings.notifications;
		if (storeVal !== previousStoreValue) {
			notificationsOn = storeVal === 'enabled';
			previousStoreValue = storeVal;
		}
		permStatus = notificationPermission();
	});

	// React to checkbox toggle
	$effect(() => {
		const enabled = notificationsOn;
		const currentStore = $settings.notifications;
		const wantsEnable = enabled && currentStore !== 'enabled';
		const wantsDisable = !enabled && currentStore === 'enabled';

		if (!wantsEnable && !wantsDisable) {
			return;
		}

		if (wantsEnable) {
			// Request permission asynchronously, then update
			requestNotificationPermission().then((perm) => {
				permStatus = perm;
				if (perm === 'granted') {
					settings.update((s) => ({ ...s, notifications: 'enabled' }));
				} else {
					// Denied or dismissed — revert the checkbox
					notificationsOn = false;
				}
			});
		} else {
			settings.update((s) => ({ ...s, notifications: 'disabled' }));
		}
	});
</script>

<Dialog class="dialog-panel flex max-h-[85vh] max-w-3xl flex-col overflow-hidden" {open} {onclose}>
	<div class="flex h-full max-h-[85vh] flex-col">
		<!-- Header -->
		<div class="flex items-center justify-between p-5 pb-0">
			<h2 class="dialog-title">Settings</h2>
			<button class="btn-close" onclick={onclose}>
				<SvgClose class="h-5 w-5" />
			</button>
		</div>

		<PillTabs bind:value={activeTab} class="flex-1">
			<Tab id="general" label="General" class="overflow-y-auto">
				<div class="space-y-4 p-5">
					<section class="space-y-4">
						<h3 class="section-heading">Notifications</h3>

						{#if !notificationsSupported()}
							<p class="text-sm text-gray-500">
								Browser notifications are not supported in this environment.
							</p>
						{:else}
							<div class="flex items-start gap-3">
								<Checkbox id="settings-notifications" bind:checked={notificationsOn} />
								<div>
									<label class="cursor-pointer text-sm text-gray-300" for="settings-notifications">
										Browser notifications
									</label>
									<p class="mt-0.5 text-xs text-gray-500">
										Get notified when captioning finishes or encounters an error.
									</p>
									{#if permStatus === 'denied'}
										<p class="mt-1 text-xs text-yellow-400">
											Notification permission is blocked. Enable it in your browser's site settings.
										</p>
									{:else if permStatus === 'unsupported'}
										<p class="mt-1 text-xs text-gray-500">
											Notifications are not available in this browser.
										</p>
									{/if}
								</div>
							</div>
						{/if}
					</section>
				</div>
			</Tab>
			<Tab id="environments" label="Environments" class="overflow-y-auto">
				<div class="space-y-4 p-5">
					<p class="text-sm text-gray-500">
						Environments store API connection settings (URL, token, and default model) so you can
						quickly switch between different providers or local servers when captioning.
					</p>

					{#if envError}
						<div class="alert-error">{envError}</div>
					{/if}

					{#if !editingEnv && !isNewEnv}
						<div class="space-y-2">
							{#if envLoading}
								<SpinnerBlock class="py-8" />
							{:else if $envs.items.length === 0}
								<p class="py-4 text-center text-sm text-gray-500">No environments yet.</p>
							{:else}
								{#each $envs.items as name (name)}
									<div
										class="group flex items-center gap-3 rounded-lg border border-border bg-bg p-3"
									>
										<div class="min-w-0 flex-1">
											<span class="text-sm font-medium text-white">{name}</span>
										</div>
										<div
											class="flex items-center gap-1 opacity-0 transition-opacity group-hover:opacity-100 max-lg:opacity-100"
										>
											<button
												class="cursor-pointer rounded p-1.5 text-gray-400 transition-colors hover:text-accent"
												title="Edit"
												onclick={() => startEditEnv(name)}
											>
												<SvgEdit class="h-4 w-4" />
											</button>
											{#if name !== 'default'}
												<button
													class="cursor-pointer rounded p-1.5 text-gray-400 transition-colors hover:text-error"
													title="Delete"
													onclick={() => (confirmDelete = name)}
												>
													<SvgDelete class="h-4 w-4" />
												</button>
											{/if}
										</div>
									</div>
								{/each}
							{/if}
						</div>

						<button
							class="btn-secondary w-full justify-center py-2.5 text-gray-200"
							onclick={startCreateEnv}
						>
							<SvgPlus class="h-4 w-4" />
							New Environment
						</button>

						<ConfirmDelete
							open={confirmDelete !== null}
							oncancel={() => (confirmDelete = null)}
							onconfirm={() => handleDeleteEnv(confirmDelete!)}
						>
							Delete environment <strong>{confirmDelete}</strong>?
						</ConfirmDelete>
					{:else}
						<div class="space-y-4">
							{#if saveEnvError}
								<div class="alert-error">{saveEnvError}</div>
							{/if}

							<div>
								<label class="label" for="env-name">Name</label>
								<input
									id="env-name"
									type="text"
									bind:value={editName}
									disabled={!isNewEnv}
									class="input disabled:opacity-50"
									placeholder="my-environment"
								/>
							</div>

							<div>
								<label class="label" for="env-url">API URL</label>
								<input
									id="env-url"
									type="text"
									bind:value={editUrl}
									class="input"
									placeholder="https://api.openai.com"
								/>
							</div>

							<div>
								<label class="label" for="env-token">
									API Token
									{#if editingEnv?.api_token}
										<span class="ml-1 text-gray-500">(leave blank to keep current)</span>
									{/if}
								</label>
								<input
									id="env-token"
									type="password"
									bind:value={editToken}
									class="input"
									placeholder={editingEnv?.api_token ? '••••••••' : 'sk-...'}
								/>
							</div>

							<div>
								<label class="label" for="env-model">Default Model</label>
								<input
									id="env-model"
									type="text"
									bind:value={editModelName}
									class="input"
									placeholder="gpt-4o-mini"
								/>
							</div>

							<div class="btn-bar">
								<button class="btn-secondary" onclick={cancelEditEnv} disabled={isSavingEnv}>
									Cancel
								</button>
								<button class="btn-primary" onclick={handleSaveEnv} disabled={isSavingEnv}>
									{isSavingEnv ? 'Saving...' : isNewEnv ? 'Create' : 'Save'}
								</button>
							</div>
						</div>
					{/if}
				</div>
			</Tab>
		</PillTabs>
	</div>
</Dialog>
