<script lang="ts">
    /**
     * Global confirmation dialog driven by the `confirm` store.
     * Mounted once in +layout.svelte.
     *
     * Supports both simple string messages and rich snippet bodies.
     * Escape and backdrop click resolve with `false`.
     */
    import Dialog from './Dialog.svelte';
    import { confirmState, resolveConfirm, type ConfirmVariant } from '$lib/stores/confirm';
    import SvgError from '$lib/icons/SvgError.svelte';
    import SvgInfo from '$lib/icons/SvgInfo.svelte';
    import SvgWarning from '$lib/icons/SvgWarning.svelte';

    const variantAlertClass: Record<ConfirmVariant, string> = {
        danger: 'alert-error',
        warning: 'alert-warning',
        info: 'alert-info'
    };

    const variantIcon: Record<ConfirmVariant, typeof SvgInfo> = {
        danger: SvgError,
        warning: SvgWarning,
        info: SvgInfo
    };

    const variantButtonClass: Record<ConfirmVariant, string> = {
        danger: 'btn-danger',
        warning: 'btn-primary',
        info: 'btn-primary'
    };

    function handleConfirm() {
        resolveConfirm(true);
    }

    function handleCancel() {
        resolveConfirm(false);
    }
</script>

{#if $confirmState.open && $confirmState.options}
    {@const opts = $confirmState.options}
    {@const variant = opts.variant ?? 'warning'}
    {@const Icon = variantIcon[variant]}

    <Dialog
        class="dialog-panel w-full max-w-lg max-md:max-w-md max-sm:max-w-sm"
        open={true}
        onclose={handleCancel}
    >
        <div class="space-y-4 p-5">
            <div class="dialog-header">
                <h2 class="dialog-title">{opts.title ?? 'Confirm'}</h2>
            </div>

            <div class="{variantAlertClass[variant]} flex items-start gap-3">
                <span class="mt-0.5 shrink-0">
                    <Icon class="h-5 w-5" />
                </span>
                <div class="min-w-0 flex-1 wrap-break-word">
                    {#if opts.body}
                        {@render opts.body()}
                    {:else}
                        {opts.message ?? 'Are you sure?'}
                    {/if}
                </div>
            </div>

            <div class="btn-bar">
                <button class="btn-secondary px-3 py-1.5" onclick={handleCancel}>
                    {opts.cancelLabel ?? 'Cancel'}
                </button>
                <button class={variantButtonClass[variant]} onclick={handleConfirm}>
                    {opts.confirmLabel ?? 'Confirm'}
                </button>
            </div>
        </div>
    </Dialog>
{/if}
