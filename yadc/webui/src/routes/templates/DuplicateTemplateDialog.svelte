<script lang="ts">
    import Dialog from '$lib/components/ui/Dialog.svelte';
    import SvgClose from '$lib/icons/SvgClose.svelte';
    import { duplicateTemplate, type TemplateListItem } from '$lib/stores/templates';
    import { friendlyErrorMessage } from '$lib/api';

    interface Props {
        open: boolean;
        sourceTemplateName: string;
        /** All existing template names — used to detect a name collision
         *  and warn the user before overwriting. */
        existingNames: string[];
        onclose: () => void;
        onduplicated: (template: TemplateListItem) => void;
    }

    let { open, sourceTemplateName, existingNames, onclose, onduplicated }: Props = $props();

    let newName = $state('');
    let isSubmitting = $state(false);
    let error: string | null = $state(null);

    // Collision detection against the names the parent passed in. We
    // derive instead of precomputing so the check stays accurate if
    // `existingNames` ever changes between renders.
    const trimmedName = $derived(newName.trim());
    const isSameAsSource = $derived(trimmedName === sourceTemplateName);
    const collisionName = $derived(
        trimmedName && !isSameAsSource ? existingNames.find((n) => n === trimmedName) : undefined
    );

    // Reset on close / prefill on open
    $effect(() => {
        if (!open) {
            newName = '';
            isSubmitting = false;
            error = null;
            return;
        }
        newName = `${sourceTemplateName}-copy`;
    });

    async function submit() {
        if (!trimmedName) {
            error = 'New template name is required';
            return;
        }
        if (isSameAsSource) {
            error = 'New name must differ from the source name';
            return;
        }
        isSubmitting = true;
        error = null;
        try {
            const info = await duplicateTemplate(sourceTemplateName, trimmedName);
            onduplicated({ name: info.name, source: 'user' });
            onclose();
        } catch (e) {
            error = friendlyErrorMessage(e, 'Failed to duplicate template');
        } finally {
            isSubmitting = false;
        }
    }

    function handleCancel() {
        onclose();
    }
</script>

<Dialog class="dialog-panel max-w-lg" {open} {onclose}>
    <div class="p-5">
        <div class="dialog-header">
            <h2 class="dialog-title">Duplicate "{sourceTemplateName}"</h2>
            <button class="btn-close" onclick={onclose}>
                <SvgClose class="h-5 w-5" />
            </button>
        </div>

        <div class="space-y-4 py-4">
            <div>
                <label class="label" for="duplicate-template-name">New Template Name</label>
                <input
                    id="duplicate-template-name"
                    type="text"
                    bind:value={newName}
                    class="input"
                    placeholder="my-template-copy"
                />
            </div>

            {#if collisionName}
                <div class="alert-warning">
                    <p class="font-semibold">A template named "{collisionName}" already exists</p>
                    <p class="mt-1 text-sm">
                        Duplicating will overwrite it. Pick a different name if you want to keep
                        both.
                    </p>
                </div>
            {:else}
                <p class="text-sm text-muted">
                    The source template's content will be copied to the new template. You can edit
                    the copy without affecting the original.
                </p>
            {/if}

            {#if error}
                <div class="alert-error">{error}</div>
            {/if}

            <div class="flex justify-end gap-2">
                <button
                    class="btn-secondary"
                    onclick={handleCancel}
                    disabled={isSubmitting}
                    type="button"
                >
                    Cancel
                </button>
                <button class="btn-primary" onclick={submit} disabled={isSubmitting} type="button">
                    {isSubmitting ? 'Duplicating…' : 'Duplicate'}
                </button>
            </div>
        </div>
    </div>
</Dialog>
