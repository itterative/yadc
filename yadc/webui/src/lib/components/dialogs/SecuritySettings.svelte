<script lang="ts">
    import { fetchKeyMode, setKeyMode, changeKeyPassword } from '$lib/stores/envs';
    import { friendlyErrorMessage } from '$lib/api';

    let keyMode: 'keyring' | 'password' = $state('keyring');
    let desiredKeyMode: 'keyring' | 'password' = $state('keyring');
    let keyModeLoading = $state(false);
    let keyModeError: string | null = $state(null);
    let keyModePassword = $state('');
    let keyModeOldPassword = $state('');
    let keyModeConfirmPassword = $state('');
    let envPasswordSet = $state(false);

    $effect(() => {
        loadKeyMode();
    });

    async function loadKeyMode() {
        try {
            const result = await fetchKeyMode();
            keyMode = result.mode;
            desiredKeyMode = result.mode;
            envPasswordSet = result.env_password_set;
        } catch {
            // ignore
        }
    }

    async function handleKeyModeAction() {
        keyModeLoading = true;
        keyModeError = null;
        try {
            if (keyMode === 'password' && desiredKeyMode === 'password') {
                await changeKeyPassword(keyModeOldPassword, keyModePassword);
                keyModePassword = '';
                keyModeOldPassword = '';
                keyModeConfirmPassword = '';
            } else {
                const password = desiredKeyMode === 'password' ? keyModePassword : undefined;
                const oldPassword = keyMode === 'password' ? keyModeOldPassword : undefined;
                await setKeyMode(desiredKeyMode, password, oldPassword);
                keyMode = desiredKeyMode;
                keyModePassword = '';
                keyModeOldPassword = '';
                keyModeConfirmPassword = '';
            }
        } catch (e) {
            keyModeError = friendlyErrorMessage(e, 'Failed to update key storage');
        } finally {
            keyModeLoading = false;
        }
    }

    function isKeyModeActionDisabled(): boolean {
        if (keyModeLoading) {
            return true;
        }
        if (desiredKeyMode === keyMode) {
            if (keyMode === 'password') {
                if (!keyModePassword) {
                    return true;
                }
                if (keyModePassword !== keyModeConfirmPassword) {
                    return true;
                }
            }
            return false;
        }
        if (keyMode === 'password' && desiredKeyMode === 'keyring') {
            return !keyModeOldPassword;
        }
        return false;
    }
</script>

<div class="space-y-4 p-5">
    <section class="space-y-4 rounded-lg border border-border bg-bg p-4">
        <h3 class="section-heading">Key Storage</h3>
        <p class="text-xs text-gray-500">Choose how your API token encryption keys are stored.</p>

        {#if keyModeError}
            <div class="alert-error text-sm">{keyModeError}</div>
        {/if}

        {#if envPasswordSet}
            <div class="rounded-lg border border-yellow-700/50 bg-yellow-900/50 px-3 py-2">
                <p class="text-xs text-yellow-200">
                    <strong>YADC_PASSWORD</strong> is set in the server environment. Changing the
                    password here will not update the environment variable — make sure to update
                    <strong>YADC_PASSWORD</strong> as well or the backend will continue using the old
                    password.
                </p>
            </div>
        {/if}

        <div class="flex flex-col gap-2">
            <label class="flex cursor-pointer items-center gap-2">
                <input
                    type="radio"
                    name="key-mode"
                    value="keyring"
                    bind:group={desiredKeyMode}
                    class="h-4 w-4 accent-accent"
                    disabled={keyModeLoading}
                />
                <span class="text-sm text-gray-300">System keyring</span>
            </label>
            <label class="flex cursor-pointer items-center gap-2">
                <input
                    type="radio"
                    name="key-mode"
                    value="password"
                    bind:group={desiredKeyMode}
                    class="h-4 w-4 accent-accent"
                    disabled={keyModeLoading}
                />
                <span class="text-sm text-gray-300">Password-protected</span>
            </label>
        </div>

        {#if keyMode === 'password' && desiredKeyMode === 'password'}
            <div class="space-y-3 border-t border-border pt-3">
                <p class="text-xs font-medium text-gray-400">Change Password</p>
                <div>
                    <label class="label" for="key-mode-old-password">Current Password</label>
                    <input
                        id="key-mode-old-password"
                        type="password"
                        bind:value={keyModeOldPassword}
                        class="input"
                        placeholder="Enter current password"
                        disabled={keyModeLoading}
                    />
                </div>
                <div>
                    <label class="label" for="key-mode-password">New Password</label>
                    <input
                        id="key-mode-password"
                        type="password"
                        bind:value={keyModePassword}
                        class="input"
                        placeholder="Enter new password"
                        disabled={keyModeLoading}
                    />
                </div>
                <div>
                    <label class="label" for="key-mode-confirm-password">Confirm New Password</label
                    >
                    <input
                        id="key-mode-confirm-password"
                        type="password"
                        bind:value={keyModeConfirmPassword}
                        class="input"
                        placeholder="Re-enter new password"
                        disabled={keyModeLoading}
                    />
                </div>
                {#if keyModePassword && keyModePassword !== keyModeConfirmPassword}
                    <p class="text-xs text-error">Passwords do not match.</p>
                {/if}
                <button
                    class="btn-primary"
                    onclick={handleKeyModeAction}
                    disabled={isKeyModeActionDisabled()}
                >
                    {keyModeLoading ? 'Changing…' : 'Change Password'}
                </button>
            </div>
        {:else if keyMode === 'password' && desiredKeyMode === 'keyring'}
            <div class="space-y-3 border-t border-border pt-3">
                <p class="text-xs font-medium text-gray-400">Switch to System Keyring</p>
                <div>
                    <label class="label" for="key-mode-old-password">Current Password</label>
                    <input
                        id="key-mode-old-password"
                        type="password"
                        bind:value={keyModeOldPassword}
                        class="input"
                        placeholder="Required to decrypt existing tokens"
                        disabled={keyModeLoading}
                    />
                </div>
                <div class="rounded-lg border border-yellow-700/50 bg-yellow-900/50 px-3 py-2">
                    <p class="text-xs text-yellow-200">
                        All existing tokens will be re-encrypted with the system keyring.
                    </p>
                </div>
                <button
                    class="btn-primary"
                    onclick={handleKeyModeAction}
                    disabled={isKeyModeActionDisabled()}
                >
                    {keyModeLoading ? 'Switching…' : 'Switch to Keyring'}
                </button>
            </div>
        {:else if keyMode === 'keyring' && desiredKeyMode === 'password'}
            <div class="space-y-3 border-t border-border pt-3">
                <p class="text-xs font-medium text-gray-400">Switch to Password-Protected</p>
                <div>
                    <label class="label" for="key-mode-password">Password</label>
                    <input
                        id="key-mode-password"
                        type="password"
                        bind:value={keyModePassword}
                        class="input"
                        placeholder="Leave blank to use empty password"
                        disabled={keyModeLoading}
                    />
                </div>
                <div class="rounded-lg border border-yellow-700/50 bg-yellow-900/50 px-3 py-2">
                    <p class="text-xs text-yellow-200">
                        All existing tokens will be re-encrypted with a password-protected key.
                    </p>
                </div>
                <button
                    class="btn-primary"
                    onclick={handleKeyModeAction}
                    disabled={isKeyModeActionDisabled()}
                >
                    {keyModeLoading ? 'Switching…' : 'Switch to Password'}
                </button>
            </div>
        {:else}
            <div class="border-t border-border pt-3">
                <p class="text-xs text-gray-500">
                    Currently using system keyring. Select a different option above to switch.
                </p>
            </div>
        {/if}
    </section>
</div>
