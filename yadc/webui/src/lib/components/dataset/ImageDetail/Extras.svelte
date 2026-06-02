<script lang="ts">
    import TomlEditor from '$lib/components/ui/TomlEditor.svelte';
    import SvgClose from '$lib/icons/SvgClose.svelte';
    import SvgCheck from '$lib/icons/SvgCheck.svelte';
    import type { CaptionData, ImageInfo } from '$lib/stores/datasetImages';

    interface Props {
        item: ImageInfo;
        captionData: CaptionData | null;
        isSavingExtras: boolean;
        onSaveExtras: (extrasRaw: string) => Promise<void>;
    }

    let { item, captionData, isSavingExtras, onSaveExtras }: Props = $props();

    let editExtrasRaw = $state('');

    // Reset when item changes — picks up the current TOML for the new image
    $effect(() => {
        // Track item.id as a dependency
        void item.id;
        editExtrasRaw = captionData?.extras_raw ?? '';
    });

    let isDirty = $derived(editExtrasRaw !== (captionData?.extras_raw ?? ''));

    function handleCancel() {
        editExtrasRaw = captionData?.extras_raw ?? '';
    }

    async function handleSave() {
        await onSaveExtras(editExtrasRaw);
    }
</script>

<div class="flex h-full flex-col gap-4">
    <div>
        <h3 class="mb-1 text-sm font-medium text-gray-300">Image TOML</h3>
        <p class="text-xs text-gray-500">
            Values are exposed as template variables in the prompt context
            — useful for per-image context like <code class="text-gray-400">style</code>,
            <code class="text-gray-400">characters</code>, or
            <code class="text-gray-400">context</code>.
        </p>
    </div>

    <div class="relative flex flex-1 flex-col overflow-hidden rounded-lg bg-gray-800">
        <div class="relative flex-1 p-2 box-content">
            <TomlEditor
                class="extras-editor min-h-32 rounded-md border-0 text-sm"
                bind:value={editExtrasRaw}
                editable={true}
                autoHeight={false}
            />
        </div>
        <!-- Action bar — mirrors the caption box footer. -->
        <div class="grid grid-cols-2 gap-2 bg-black/15 p-2 text-sm">
            <div class="flex items-center justify-center">
                <button
                    class="flex w-full cursor-pointer items-center justify-center gap-1.5 rounded-lg px-3 py-2 text-gray-400 transition-colors hover:bg-gray-700 hover:text-gray-200 disabled:cursor-not-allowed disabled:opacity-50"
                    onclick={handleCancel}
                    disabled={!isDirty || isSavingExtras}
                >
                    <SvgClose class="h-4 w-4 shrink-0" />
                    <span class="min-w-0 truncate">Cancel</span>
                </button>
            </div>
            <div class="flex items-center justify-center">
                <button
                    class="flex w-full cursor-pointer items-center justify-center gap-1.5 rounded-lg px-3 py-2 text-accent transition-colors hover:bg-gray-700 hover:text-accent-hover disabled:cursor-not-allowed disabled:opacity-50"
                    onclick={handleSave}
                    disabled={!isDirty || isSavingExtras}
                >
                    <SvgCheck class="h-4 w-4 shrink-0" />
                    <span class="min-w-0 truncate">{isSavingExtras ? 'Saving…' : 'Save'}</span>
                </button>
            </div>
        </div>
    </div>
</div>

<style>
    /* Match the caption box: gray-800 background and rounded corners on the
       editor surface. The wrapper has `rounded-md` + `overflow-hidden`, but
       `.cm-editor` paints its own `var(--color-surface)` background on top
       of the wrapper, so we have to override it here. */
    :global(.extras-editor .cm-editor),
    :global(.extras-editor .cm-gutters) {
        background: #1f2937 !important; /* gray-800 */
    }
    :global(.extras-editor .cm-editor) {
        border-radius: 0.375rem; /* rounded-md */
    }
</style>
