<script lang="ts">
    import Dialog from '$lib/components/ui/Dialog.svelte';

    interface Props {
        open: boolean;
        onsubmit: (password: string) => void;
        oncancel: () => void;
    }

    let { open, onsubmit, oncancel }: Props = $props();

    let password = $state('');
    let inputRef: HTMLInputElement | null = $state(null);

    $effect(() => {
        if (open && inputRef) {
            // Small delay to ensure dialog is visible before focusing
            requestAnimationFrame(() => {
                inputRef?.focus();
            });
        }
    });

    function handleSubmit(ev: SubmitEvent) {
        ev.preventDefault();
        const trimmed = password.trim();
        if (!trimmed) {
            return;
        }
        password = '';
        onsubmit(trimmed);
    }

    function handleCancel() {
        password = '';
        oncancel();
    }

    function handleKeydown(ev: KeyboardEvent) {
        if (ev.key === 'Escape') {
            handleCancel();
        }
    }
</script>

<svelte:window onkeydown={handleKeydown} />

<Dialog class="dialog-surface w-full max-w-sm p-6" {open} onclose={handleCancel}>
    <h2 class="dialog-title mb-3">Password Required</h2>
    <p class="mb-4 text-sm text-gray-300">
        This environment uses a password-protected key. Please enter the password to continue.
    </p>

    <form onsubmit={handleSubmit} class="space-y-4">
        <div>
            <label class="label mb-1 block" for="password-input">Password</label>
            <input
                id="password-input"
                type="password"
                bind:value={password}
                bind:this={inputRef}
                class="input w-full"
                placeholder="Enter password"
                autocomplete="off"
            />
        </div>

        <div class="btn-bar">
            <button type="button" class="btn-secondary px-3 py-1.5" onclick={handleCancel}>
                Cancel
            </button>
            <button type="submit" class="btn-primary px-3 py-1.5" disabled={!password.trim()}>
                Submit
            </button>
        </div>
    </form>
</Dialog>
