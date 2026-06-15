<script lang="ts">
    import { fetchKeyMode, setKeyMode, changeKeyPassword } from '$lib/stores/env';
    import { clearAuthCookie, setAuthCookie, withPasswordRetry } from '$lib/stores/passwordPrompt';
    import { friendlyErrorMessage, PasswordRequiredError } from '$lib/api';
    import Alert from '$lib/components/ui/Alert.svelte';

    let keyMode: 'keyring' | 'password' = $state('keyring');
    let desiredKeyMode: 'keyring' | 'password' = $state('keyring');
    let keyModeLoading = $state(false);
    let keyModeError: string | null = $state(null);
    let keyModeSuccess: string | null = $state(null);
    let keyModePassword = $state('');
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

    /** Change the key storage password (already in password mode).
     *
     *  The current password comes from the ``yadc_password`` session
     *  cookie (with the ``YADC_PASSWORD`` env-var fallback). If the
     *  cookie is missing or wrong, ``withPasswordRetry`` will prompt
     *  the user automatically. */
    async function changePassword() {
        await withPasswordRetry(() => changeKeyPassword(keyModePassword));
        // Refresh the cookie so the next API call uses the new password
        // (otherwise the still-old cookie would 403 the next request).
        await setAuthCookie(keyModePassword);
    }

    /** Switch the key storage mode (``password`` ↔ ``keyring``).
     *
     *  - Switching to ``keyring``: backend reads the current password
     *    from the cookie; the cookie is then cleared (no longer
     *    needed).  ``withPasswordRetry`` handles the no-cookie case.
     *  - Switching to ``password``: backend doesn't need a current
     *    password (current mode is keyring).  The new cookie is set
     *    after success so the next request authenticates. */
    async function switchMode() {
        if (desiredKeyMode === 'password') {
            await setKeyMode('password', keyModePassword);
            await setAuthCookie(keyModePassword);
        } else {
            await withPasswordRetry(() => setKeyMode('keyring'));
            await clearAuthCookie();
        }
        keyMode = desiredKeyMode;
    }

    async function handleKeyModeAction() {
        keyModeLoading = true;
        keyModeError = null;
        keyModeSuccess = null;
        try {
            if (keyMode === 'password' && desiredKeyMode === 'password') {
                await changePassword();
                keyModeSuccess = 'Password changed successfully.';
            } else {
                await switchMode();
                keyModeSuccess =
                    desiredKeyMode === 'password'
                        ? 'Switched to password-protected key storage.'
                        : 'Switched to system keyring.';
            }
            keyModePassword = '';
            keyModeConfirmPassword = '';
        } catch (e) {
            if (e instanceof PasswordRequiredError) {
                keyModeError = 'Current password is incorrect. Please try again.';
            } else {
                keyModeError = friendlyErrorMessage(e, 'Failed to update key storage');
            }
        } finally {
            keyModeLoading = false;
        }
    }

    function isKeyModeActionDisabled(): boolean {
        if (keyModeLoading) {
            return true;
        }
        if (desiredKeyMode === keyMode) {
            // Change-password case
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
        // Mode-switch case
        if (desiredKeyMode === 'password' && !keyModePassword) {
            return true;
        }
        return false;
    }
</script>

<div class="space-y-4 p-5">
    <section class="space-y-4 rounded-lg border border-border bg-bg p-4">
        <h3 class="section-heading">Key Storage</h3>
        <p class="text-xs text-gray-500">Choose how your API token encryption keys are stored.</p>

        {#if keyModeError}
            <Alert
                variant="error"
                class="text-sm"
                dismissable
                ondismiss={() => (keyModeError = null)}
            >
                {keyModeError}
            </Alert>
        {/if}

        {#if keyModeSuccess}
            <Alert
                variant="info"
                class="text-sm"
                dismissable
                ondismiss={() => (keyModeSuccess = null)}
            >
                {keyModeSuccess}
            </Alert>
        {/if}

        {#if envPasswordSet}
            <div class="rounded-lg border border-yellow-700/50 bg-yellow-900/50 px-3 py-2">
                <p class="text-xs text-yellow-200">
                    <strong>YADC_PASSWORD</strong> is set in the server environment. The webui uses a
                    session cookie for password auth, so the env var only matters for the first request
                    of a new session (before the cookie is set) and for non-browser clients. The session
                    cookie wins once it's set.
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
                <p class="text-xs text-gray-500">
                    The current password is read from your session cookie. If you don't have one
                    yet, you'll be prompted for it.
                </p>
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
                <p class="text-xs text-gray-500">
                    Your current password (from the session cookie, or prompted if missing) is used
                    to decrypt existing tokens before re-encrypting them with the keyring.
                </p>
                <div class="rounded-lg border border-yellow-700/50 bg-yellow-900/50 px-3 py-2">
                    <p class="text-xs text-yellow-200">
                        All existing tokens will be re-encrypted with the system keyring. The
                        session cookie will be cleared.
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
                        All existing tokens will be re-encrypted with a password-protected key. The
                        session cookie will be set to this password.
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
