<script lang="ts">
    import TomlEditor from '$lib/components/ui/TomlEditor.svelte';
    import Card from '$lib/components/ui/Card.svelte';
    import ActionBar from '$lib/components/ui/ActionBar.svelte';
    import ActionBarItem from '$lib/components/ui/ActionBarItem.svelte';
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
            Values are exposed as template variables in the prompt context — useful for per-image
            context like <code class="text-gray-400">style</code>,
            <code class="text-gray-400">characters</code>, or
            <code class="text-gray-400">context</code>.
        </p>
    </div>

    <Card class="relative flex flex-1 flex-col">
        <div class="relative box-content flex-1 p-2">
            <TomlEditor
                class="min-h-32 rounded-md border-0 bg-gray-800 text-sm"
                bind:value={editExtrasRaw}
                editable={true}
                autoHeight={false}
            />
        </div>
        <ActionBar>
            <ActionBarItem
                onclick={handleCancel}
                disabled={!isDirty || isSavingExtras}
                icon={SvgClose}
                variant="secondary"
            >
                Cancel
            </ActionBarItem>
            <ActionBarItem
                onclick={handleSave}
                disabled={!isDirty || isSavingExtras}
                icon={SvgCheck}
                variant="primary"
            >
                {isSavingExtras ? 'Saving…' : 'Save'}
            </ActionBarItem>
        </ActionBar>
    </Card>
</div>
