<script lang="ts">
    import type { Toast, ToastVariant } from '$lib/stores/toasts';
    import { dismissToast } from '$lib/stores/toasts';

    interface Props {
        toast: Toast;
    }

    let { toast }: Props = $props();

    // --- Auto-dismiss timer ---

    let remaining = $state(0);
    let hovered = $state(false);
    let timer: ReturnType<typeof setInterval> | undefined;
    let started = $state(false);

    // Initialize remaining from the toast's duration once
    $effect(() => {
        if (!started && toast.duration > 0) {
            remaining = toast.duration;
            started = true;
        }
    });

    $effect(() => {
        if (toast.duration <= 0) {
            return;
        }

        timer = setInterval(() => {
            if (!hovered) {
                remaining -= 100;
                if (remaining <= 0) {
                    clearInterval(timer);
                    dismissToast(toast.id);
                }
            }
        }, 100);

        return () => clearInterval(timer);
    });

    // --- Variant styling ---

    const variantStyles: Record<ToastVariant, string> = {
        info: 'bg-blue-900 border-accent/40 text-accent',
        success: 'bg-green-900 border-success/40 text-success',
        warning: 'bg-yellow-900 border-yellow-700/50 text-yellow-200',
        error: 'bg-red-900 border-error/40 text-error'
    };

    const variantIcons: Record<ToastVariant, string> = {
        info: 'ℹ',
        success: '✓',
        warning: '⚠',
        error: '✗'
    };

    const progressBarColors: Record<ToastVariant, string> = {
        info: 'bg-accent',
        success: 'bg-success',
        warning: 'bg-yellow-400',
        error: 'bg-error'
    };

    let pctRemaining = $derived(
        toast.duration > 0 ? Math.max(0, (remaining / toast.duration) * 100) : 0
    );

    function handleDismiss() {
        dismissToast(toast.id);
    }
</script>

<div
    class="relative flex max-w-sm min-w-72 items-start gap-3 rounded-lg border px-4 py-3 shadow-lg {variantStyles[
        toast.variant
    ]}"
    onmouseenter={() => (hovered = true)}
    onmouseleave={() => (hovered = false)}
    role="alert"
>
    <!-- Icon -->
    <span class="mt-0.5 flex-shrink-0 text-base leading-5">{variantIcons[toast.variant]}</span>

    <!-- Message + optional action -->
    <div class="min-w-0 flex-1">
        <p class="text-sm leading-5">{toast.message}</p>
        {#if toast.details?.length}
            <ul class="mt-1 list-disc pl-4 text-xs opacity-80">
                {#each toast.details as detail, i (i)}
                    <li>{detail}</li>
                {/each}
            </ul>
        {/if}
        {#if toast.actions?.length}
            <div class="mt-2 flex gap-2">
                {#each toast.actions as act (act.label)}
                    <button
                        class="cursor-pointer rounded-md border border-current/20 bg-white/10 px-2.5 py-1 text-xs transition-colors hover:bg-white/20"
                        onclick={act.handler}
                    >
                        {act.label}
                    </button>
                {/each}
            </div>
        {:else if toast.action}
            <button
                class="mt-2 cursor-pointer rounded-md border border-current/20 bg-white/10 px-2.5 py-1 text-xs transition-colors hover:bg-white/20"
                onclick={toast.action.handler}
            >
                {toast.action.label}
            </button>
        {/if}
    </div>

    <!-- Dismiss button -->
    <button
        class="mt-0.5 flex-shrink-0 cursor-pointer text-current/50 transition-colors hover:text-current"
        onclick={handleDismiss}
        aria-label="Dismiss"
    >
        <svg class="h-4 w-4" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M4 4l8 8M12 4l-8 8" />
        </svg>
    </button>

    <!-- Progress bar (only for timed toasts) -->
    {#if toast.duration > 0}
        <div class="absolute right-0 bottom-0 left-0 h-0.5 overflow-hidden rounded-b-lg bg-white/5">
            <div
                class="h-full transition-all duration-100 ease-linear {progressBarColors[
                    toast.variant
                ]}"
                style:width="{pctRemaining}%"
            ></div>
        </div>
    {/if}
</div>
