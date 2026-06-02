<script lang="ts">
	import Dialog from '$lib/components/ui/Dialog.svelte';
	import TabBar from '$lib/components/ui/TabBar.svelte';
	import EnvManager from '$lib/components/dialogs/EnvManager.svelte';
	import Checkbox from '$lib/components/ui/Checkbox.svelte';
	import SvgClose from '$lib/icons/SvgClose.svelte';
	import { settings } from '$lib/stores/settings';
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
	type Tab = 'general' | 'environments';
	let activeTab: Tab = $state('general');

	// --- Env state ---
	let showEnvManager = $state(false);

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

<!-- Sub-dialogs -->
<EnvManager open={showEnvManager} onclose={() => (showEnvManager = false)} />

<Dialog class="dialog-panel flex max-h-[85vh] max-w-3xl flex-col overflow-hidden" {open} {onclose}>
	<div class="flex h-full max-h-[85vh] flex-col">
		<!-- Header -->
		<div class="flex items-center justify-between p-5 pb-0">
			<h2 class="dialog-title">Settings</h2>
			<button class="btn-close" onclick={onclose}>
				<SvgClose class="h-5 w-5" />
			</button>
		</div>

		<!-- Tabs -->
		<TabBar
			class="px-5 pt-3"
			tabs={[
				{ value: 'general', label: 'General' },
				{ value: 'environments', label: 'Environments' }
			]}
			selected={activeTab}
			onchange={(v) => (activeTab = v as Tab)}
		/>

		<!-- Tab content -->
		<div class="flex-1 space-y-4 overflow-y-auto p-5">
			{#if activeTab === 'general'}
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
			{:else if activeTab === 'environments'}
				<div class="py-8 text-center">
					<p class="mb-4 text-sm text-gray-400">
						Environments store API connection settings (URL, token, default model).
					</p>
					<button class="btn-primary" onclick={() => (showEnvManager = true)}>
						Open Environment Manager
					</button>
				</div>
			{/if}
		</div>
	</div>
</Dialog>
