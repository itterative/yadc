<script lang="ts">
    import TomlEditor from '$lib/components/ui/TomlEditor.svelte';
    import { configState } from './datasetConfig/state.svelte';

    interface Props {
        /** Whether the dataset is managed (controls the path-warning text). */
        source?: 'upload' | 'import' | 'create';
    }

    let { source }: Props = $props();
</script>

<div class="flex h-full flex-col gap-2">
    {#if configState.simplifiedDirty}
        <div class="alert-info text-xs">
            Form view has unsaved changes that are not reflected here.
        </div>
    {/if}
    <TomlEditor
        class="rounded-md border border-border bg-surface text-sm"
        bind:value={configState.rawContent}
        editable={true}
    />
    {#if source === 'upload'}
        <p class="text-xs text-gray-500">
            Editing <code>[[dataset]]</code> paths for managed datasets will be rejected by the server.
            Use the Upload tab to add or remove images.
        </p>
    {/if}
</div>
