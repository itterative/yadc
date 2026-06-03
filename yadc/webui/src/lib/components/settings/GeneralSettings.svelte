<script lang="ts">
    import Checkbox from '$lib/components/ui/Checkbox.svelte';
    import { settings } from '$lib/stores/settings';
    import {
        notificationsSupported,
        notificationPermission,
        requestNotificationPermission
    } from '$lib/notifications';

    let permStatus = $state<NotificationPermission | 'unsupported'>('default');
    let notificationsOn = $state(false);
    let previousStoreValue = $state<'unset' | 'enabled' | 'disabled'>('unset');

    $effect(() => {
        const storeVal = $settings.notifications;
        if (storeVal !== previousStoreValue) {
            notificationsOn = storeVal === 'enabled';
            previousStoreValue = storeVal;
        }
        permStatus = notificationPermission();
    });

    $effect(() => {
        const enabled = notificationsOn;
        const currentStore = $settings.notifications;
        const wantsEnable = enabled && currentStore !== 'enabled';
        const wantsDisable = !enabled && currentStore === 'enabled';

        if (!wantsEnable && !wantsDisable) {
            return;
        }

        if (wantsEnable) {
            requestNotificationPermission().then((perm) => {
                permStatus = perm;
                if (perm === 'granted') {
                    settings.update((s) => ({ ...s, notifications: 'enabled' }));
                } else {
                    notificationsOn = false;
                }
            });
        } else {
            settings.update((s) => ({ ...s, notifications: 'disabled' }));
        }
    });
</script>

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
                    <label
                        class="cursor-pointer text-sm text-gray-300"
                        for="settings-notifications"
                    >
                        Browser notifications
                    </label>
                    <p class="mt-0.5 text-xs text-gray-500">
                        Get notified when captioning finishes or encounters an error.
                    </p>
                    {#if permStatus === 'denied'}
                        <p class="mt-1 text-xs text-yellow-400">
                            Notification permission is blocked. Enable it in your browser's site
                            settings.
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
