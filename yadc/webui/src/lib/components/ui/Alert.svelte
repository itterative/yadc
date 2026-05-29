<script lang="ts">
    /**
     * Inline alert with icon, optional title, actions, and dismiss support.
     *
     * Visibility is controlled by the parent via `visible` (default true).
     * When `dismissable` is true, a close button hides the alert immediately
     * (internal dismissed state) and calls `ondismiss` so the parent can
     * clear the source condition (e.g. a Svelte store).
     *
     * The dismissed state resets automatically when `visible` transitions
     * from false back to true — so the parent must clear its condition in
     * `ondismiss` to ensure the round-trip (visible=true → dismissed=true →
     * visible=false → visible=true again later) works correctly.
     *
     * Usage:
     *   <Alert variant="info">Simple message</Alert>
     *   <Alert variant="warning" visible={$store} dismissable ondismiss={clearStore}>...</Alert>
     */
    import type { Snippet } from 'svelte';
    import SvgCheck from '$lib/icons/SvgCheck.svelte';
    import SvgClose from '$lib/icons/SvgClose.svelte';
    import SvgError from '$lib/icons/SvgError.svelte';
    import SvgInfo from '$lib/icons/SvgInfo.svelte';
    import SvgWarning from '$lib/icons/SvgWarning.svelte';

    type AlertVariant = 'info' | 'warning' | 'error' | 'success';

    interface Props {
        class?: string;
        variant?: AlertVariant;
        visible?: boolean;
        dismissable?: boolean;
        title?: string;
        children: Snippet;
        actions?: Snippet;
        ondismiss?: () => void;
    }

    let {
        class: className = '',
        variant = 'info',
        visible = true,
        dismissable = false,
        title,
        children,
        actions,
        ondismiss
    }: Props = $props();

    const variantIcons: Record<AlertVariant, typeof SvgInfo> = {
        info: SvgInfo,
        success: SvgCheck,
        warning: SvgWarning,
        error: SvgError
    };

    let Icon = $derived(variantIcons[variant]);

    let dismissed = $state(false);

    $effect(() => {
        if (visible) {
            dismissed = false;
        }
    });

    function handleDismiss() {
        dismissed = true;
        ondismiss?.();
    }
</script>

{#if visible && !dismissed}
    <div class="alert-{variant} {className} flex items-start gap-3" role="alert">
        <span class="mt-0.5 shrink-0">
            <Icon class="h-5 w-5" />
        </span>
        <div class="min-w-0 flex-1">
            {#if title}
                <p class="font-medium">{title}</p>
            {/if}
            <div class:text-sm={!title}>
                {@render children()}
            </div>
            {#if actions}
                <div class="mt-2 flex gap-2">
                    {@render actions()}
                </div>
            {/if}
        </div>
        {#if dismissable}
            <button
                class="mt-0.5 shrink-0 cursor-pointer text-current/50 transition-colors hover:text-current"
                onclick={handleDismiss}
                aria-label="Dismiss"
            >
                <SvgClose class="h-4 w-4" />
            </button>
        {/if}
    </div>
{/if}
