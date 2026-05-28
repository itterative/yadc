<script lang="ts">
    import type { Snippet } from 'svelte';
    import Dialog from './Dialog.svelte';

    interface Props {
        open: boolean;
        oncancel: () => void;
        onconfirm: () => void;
        children: Snippet;
    }

    let { open, oncancel, onconfirm, children }: Props = $props();
</script>

<Dialog class="dialog-panel m-auto w-full max-w-sm" {open} onclose={oncancel}>
    <div class="space-y-4 p-5">
        <div class="dialog-header">
            <h2 class="dialog-title">Confirm Delete</h2>
        </div>
        <div class="alert-warning">
            {@render children()}
        </div>
        <div class="btn-bar">
            <button class="btn-secondary px-3 py-1.5" onclick={oncancel}>Cancel</button>
            <button class="btn-danger" onclick={onconfirm}>Delete</button>
        </div>
    </div>
</Dialog>
