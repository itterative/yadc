<script lang="ts">
    import TomlEditor from '$lib/components/ui/TomlEditor.svelte';
    import CaptionOptionsFields from '$lib/components/settings/CaptionOptionsFields.svelte';
    import KeyValueEditor from '$lib/components/ui/KeyValueEditor.svelte';
    import SvgPlus from '$lib/icons/SvgPlus.svelte';
    import SvgDelete from '$lib/icons/SvgDelete.svelte';
    import { templates } from '$lib/stores/templates';
    import { configState } from './state.svelte';
    import ConfigApiSection from './ConfigApiSection.svelte';

    interface Props {
        /** Dataset source — managed (upload) vs external (import/create). */
        source?: 'upload' | 'import' | 'create';
        /** Dataset name (for displaying the path). */
        datasetName: string;
        /** Path to the loaded config file. */
        loadedConfigPath: string;
        /** Live preview content (rendered by the container). */
        previewContent: string;
    }

    let { source, datasetName, loadedConfigPath, previewContent }: Props = $props();

    function displayConfigPath(configPath: string | undefined): string {
        if (!configPath) {
            return '';
        }
        if (source === 'upload' || source === 'create') {
            return `${datasetName}/config.toml`;
        }
        return configPath;
    }
</script>

<div class="flex flex-col gap-4">
    {#if configState.rawDirty}
        <div class="alert-info text-xs">
            Advanced view has unsaved changes that are not reflected here.
        </div>
    {/if}

    <!-- ═══ Info: Dataset source & path ═══ -->
    <section class="space-y-2">
        <h3 class="section-heading">Dataset</h3>
        <div class="space-y-1">
            <div class="flex items-center gap-2">
                {#if source === 'upload'}
                    <span class="badge-muted badge-sm">Managed</span>
                    <span class="text-xs text-gray-400">Files stored in yadc state directory</span>
                {:else}
                    <span class="badge-muted badge-sm">External</span>
                    <span class="text-xs text-gray-400">References paths outside yadc</span>
                {/if}
            </div>
            {#if loadedConfigPath}
                <p
                    class="truncate font-mono text-xs text-gray-500"
                    title={source === 'upload' || source === 'create'
                        ? loadedConfigPath
                        : undefined}
                >
                    {displayConfigPath(loadedConfigPath)}
                </p>
            {/if}
        </div>
    </section>

    <!-- ═══ Section: Dataset Entries ═══ -->
    <section class="space-y-3">
        <h3 class="section-heading">Paths & Variables</h3>
        <p class="help-text">
            Each entry is a directory of images with optional template variables.
        </p>

        {#each configState.datasetEntries as entry, i (i)}
            <div class="card">
                <div class="card-body space-y-3">
                    <!-- Entry header -->
                    <div class="flex items-start gap-2">
                        <div class="flex-1">
                            <label class="label" for="entry-path-{i}">Path</label>
                            <input
                                id="entry-path-{i}"
                                type="text"
                                class="input font-mono text-sm"
                                value={entry.path}
                                placeholder="/path/to/images"
                                disabled={source === 'upload'}
                                title={source === 'upload'
                                    ? 'Managed dataset paths are set automatically via uploads'
                                    : undefined}
                                oninput={(e) =>
                                    configState.updateEntryPath(i, e.currentTarget.value)}
                            />
                        </div>
                        {#if configState.datasetEntries.length > 1 && source !== 'upload'}
                            <button
                                class="mt-6 cursor-pointer p-1 text-gray-500 transition-colors hover:text-error"
                                onclick={() => configState.removeDatasetEntry(i)}
                                title="Remove entry"
                                aria-label="Remove entry"
                            >
                                <SvgDelete class="h-4 w-4" />
                            </button>
                        {/if}
                    </div>

                    <!-- Inline images badge -->
                    {#if entry.imageCount > 0}
                        <p class="text-xs text-gray-500">
                            {entry.imageCount} inline image{entry.imageCount !== 1 ? 's' : ''} — edit
                            in raw TOML view
                        </p>
                    {/if}

                    <!-- Empty entry warning -->
                    {#if !entry.path && entry.imageCount === 0}
                        <p class="text-xs text-warning">Empty entry — set a path or add images.</p>
                    {/if}

                    <!-- Extras -->
                    <div>
                        <div
                            class="mb-1.5 text-xs font-medium tracking-wide text-gray-300 uppercase"
                        >
                            Variables
                        </div>
                        <KeyValueEditor
                            bind:entries={configState.datasetEntries[i].extras}
                            idPrefix="entry-{i}-extras"
                            keyPlaceholder="variable name"
                        />
                    </div>
                </div>
            </div>
        {/each}

        {#if source !== 'upload'}
            <button class="btn-secondary w-full" onclick={() => configState.addDatasetEntry()}>
                <SvgPlus class="mr-1 inline-block h-4 w-4" />
                Add Path
            </button>
        {/if}
    </section>

    <!-- ═══ Section: API ═══ -->
    <ConfigApiSection />

    <!-- ═══ Section: Prompt ═══ -->
    <section class="space-y-3">
        <h3 class="section-heading">Prompt</h3>

        <div>
            <label class="label" for="config-prompt">Template</label>
            <select
                id="config-prompt"
                class="input cursor-pointer"
                value={configState.promptName}
                onchange={(e) => {
                    configState.promptName = e.currentTarget.value;
                }}
            >
                <option value="">(default)</option>
                {#each $templates.items as t (t.name)}
                    <option value={t.name}
                        >{t.name}{t.source === 'builtin' ? ' (built-in)' : ''}</option
                    >
                {/each}
            </select>
            <p class="help-text">
                Prompt template used during captioning. Leave empty for the default template.
            </p>
        </div>
    </section>

    <!-- ═══ Section: Options & Reasoning (shared component) ═══ -->
    <CaptionOptionsFields
        bind:maxTokens={configState.maxTokens}
        bind:imageQuality={configState.imageQuality}
        bind:rounds={configState.rounds}
        bind:overwrite={configState.overwrite}
        bind:reasoningEnabled={configState.reasoningEnabled}
        bind:reasoningEffort={configState.reasoningEffort}
        bind:storeConversation={configState.storeConversation}
        bind:reasoningExcludeOutput={configState.reasoningExcludeOutput}
        display={{
            idPrefix: 'config',
            nullable: true,
            helpText: true,
            showStoreConversation: true,
            showReasoningExcludeOutput: true
        }}
    />

    <!-- ═══ Section: Preview ═══ -->
    <section class="space-y-3">
        <h3 class="section-heading">Preview</h3>
        {#if configState.simplifiedDirty}
            <p class="text-xs text-accent">Previewing unsaved changes.</p>
        {:else}
            <p class="text-xs text-gray-500">Current config on disk.</p>
        {/if}
        <div class="max-h-[40vh] min-h-30 overflow-y-auto">
            <TomlEditor
                class="rounded-md border border-border bg-surface text-sm"
                value={previewContent}
                editable={false}
            />
        </div>
    </section>

    <div class="h-2"></div>
</div>
