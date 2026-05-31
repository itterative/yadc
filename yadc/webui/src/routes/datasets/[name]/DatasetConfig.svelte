<script lang="ts">
    import TomlEditor from '$lib/components/ui/TomlEditor.svelte';
    import CaptionOptionsFields from '$lib/components/settings/CaptionOptionsFields.svelte';
    import KeyValueEditor, { type KeyValueEntry } from '$lib/components/ui/KeyValueEditor.svelte';
    import SpinnerBlock from '$lib/components/ui/SpinnerBlock.svelte';
    import SvgRefresh from '$lib/icons/SvgRefresh.svelte';
    import SvgPlus from '$lib/icons/SvgPlus.svelte';
    import SvgDelete from '$lib/icons/SvgDelete.svelte';
    import SvgEdit from '$lib/icons/SvgEdit.svelte';
    import SvgFile from '$lib/icons/SvgFile.svelte';
    import CompactPillTabs from '$lib/components/ui/tabs/CompactPillTabs.svelte';
    import Tab from '$lib/components/ui/tabs/Tab.svelte';
    import {
        fetchConfig,
        patchConfig,
        previewConfig,
        updateConfig,
        type Config,
        type ConfigDatasetEntry
    } from '$lib/stores/configs';
    import { templates, refreshTemplates } from '$lib/stores/templates';
    import { toast } from '$lib/stores/toasts';
    import { friendlyErrorMessage } from '$lib/api';
    import ConfigHistory from './ConfigHistory.svelte';
    import SvgHistory from '$lib/icons/SvgHistory.svelte';

    // --- Props ---

    interface Props {
        /** Which dataset this is for. */
        datasetName: string;
        /** Dataset source — managed (upload) vs external (import/create). */
        source?: 'upload' | 'import' | 'create';
        /** Called after the config is saved and rescanned. */
        onsaved?: () => void;
    }

    let { datasetName, source, onsaved }: Props = $props();

    function displayConfigPath(configPath: string | undefined): string {
        if (!configPath) {
            return '';
        }
        if (source === 'upload' || source === 'create') {
            return `${datasetName}/config.toml`;
        }
        return configPath;
    }

    // --- State ---

    let isLoading = $state(false);
    let isSaving = $state(false);
    let error: string | null = $state(null);
    let saveError: string | null = $state(null);
    let validationErrors: Array<{ loc: string[]; msg: string }> = $state([]);
    let loadedConfigPath: string = $state('');

    // (template list comes from the shared store)

    // --- Structured fields (parsed from config) ---

    // API
    let apiUrl = $state('');
    let apiModelName = $state('');

    // Prompt
    let promptName = $state('');

    // Options (nullable fields: maxTokens, imageQuality, rounds — null means "not in TOML")
    let maxTokens: number | null = $state(null);
    let imageQuality: 'auto' | 'high' | 'low' | null = $state(null);
    let rounds: number | null = $state(null);
    let overwrite = $state(false);
    let storeConversation = $state(false);

    // Reasoning
    let reasoningEnabled = $state(false);
    let reasoningEffort: 'low' | 'medium' | 'high' = $state('low');
    let reasoningExcludeOutput = $state(true);

    // Environment
    let envName = $state('');

    // Dataset entries
    interface DatasetEntry {
        path: string;
        extras: KeyValueEntry[];
        imageCount: number;
    }

    let datasetEntries: DatasetEntry[] = $state([]);

    // --- View mode ---

    /** Current view mode — driven by CompactPillTabs. */
    let activeView: 'simplified' | 'advanced' = $state('simplified');

    /** Raw TOML content for Advanced mode editing. */
    let rawContent = $state('');
    /** Snapshot of raw content at load / last save, for Advanced mode dirty tracking. */
    let loadedRawContent = $state('');

    /** Whether the Advanced mode editor has unsaved changes. */
    let rawDirty = $derived(rawContent !== loadedRawContent);

    // --- Snapshot of last-loaded values for dirty tracking ---

    let loadedApiUrl = $state('');
    let loadedApiModelName = $state('');
    let loadedPromptName = $state('');
    let loadedMaxTokens: number | null = $state(null);
    let loadedImageQuality: 'auto' | 'high' | 'low' | null = $state(null);
    let loadedRounds: number | null = $state(null);
    let loadedOverwrite = $state(false);
    let loadedStoreConversation = $state(false);
    let loadedReasoningEnabled = $state(false);
    let loadedReasoningEffort: 'low' | 'medium' | 'high' = $state('low');
    let loadedReasoningExcludeOutput = $state(true);
    let loadedEnvName = $state('');
    let loadedDatasetEntries: DatasetEntry[] = $state([]);

    /** Deep-compare two dataset entry arrays for equality. */
    function entriesEqual(a: DatasetEntry[], b: DatasetEntry[]): boolean {
        if (a.length !== b.length) {
            return false;
        }
        return a.every((entry, i) => {
            const other = b[i];
            return (
                entry.path === other.path &&
                entry.imageCount === other.imageCount &&
                entry.extras.length === other.extras.length &&
                entry.extras.every(
                    (e, j) => e.key === other.extras[j].key && e.value === other.extras[j].value
                )
            );
        });
    }

    let simplifiedDirty = $derived(
        apiUrl !== loadedApiUrl ||
            apiModelName !== loadedApiModelName ||
            promptName !== loadedPromptName ||
            maxTokens !== loadedMaxTokens ||
            imageQuality !== loadedImageQuality ||
            rounds !== loadedRounds ||
            overwrite !== loadedOverwrite ||
            storeConversation !== loadedStoreConversation ||
            reasoningEnabled !== loadedReasoningEnabled ||
            reasoningEffort !== loadedReasoningEffort ||
            reasoningExcludeOutput !== loadedReasoningExcludeOutput ||
            envName !== loadedEnvName ||
            !entriesEqual(datasetEntries, loadedDatasetEntries)
    );

    let dirty = $derived(activeView === 'simplified' ? simplifiedDirty : rawDirty);

    // --- Live preview ---

    let previewContent = $state('');
    let previewTimer: ReturnType<typeof setTimeout> | null = null;
    const DEBOUNCE_MS = 400;

    /** Build the patch dict from current field values. Nullable fields are omitted when null. */
    function buildPatch(): Partial<Config> {
        const settings: Record<string, unknown> = {
            store_conversation: storeConversation
        };
        if (maxTokens !== null) {
            settings.max_tokens = maxTokens;
        }
        if (imageQuality !== null) {
            settings.image_quality = imageQuality;
        }

        // Build dataset entries, preserving inline images from original config
        const dataset = datasetEntries.map((entry, i) => {
            const obj: Record<string, unknown> = { path: entry.path };
            const extras: Record<string, string> = {};
            for (const { key, value } of entry.extras) {
                if (key) {
                    extras[key] = value;
                }
            }
            if (Object.keys(extras).length > 0) {
                obj.extras = extras;
            }
            // Preserve inline images from the original parsed config (read-only, not edited)
            const raw = parsedDatasetRaw[i];
            if (raw?.images && raw.images.length > 0) {
                obj.images = raw.images;
            }
            return obj;
        });

        const patch: Record<string, unknown> = {
            api: {
                url: apiUrl,
                model_name: apiModelName
            },
            prompt: {
                name: promptName
            },
            settings,
            overwrite_captions: overwrite,
            reasoning: {
                enable: reasoningEnabled,
                thinking_effort: reasoningEffort,
                exclude_from_output: reasoningExcludeOutput
            },
            env: envName,
            dataset
        };
        if (rounds !== null) {
            patch.rounds = rounds;
        }
        return patch;
    }

    /** Debounced: request a dry-run preview from the backend. */
    function schedulePreview() {
        if (previewTimer !== null) {
            clearTimeout(previewTimer);
        }
        previewTimer = setTimeout(() => {
            previewTimer = null;
            previewConfig(datasetName, buildPatch())
                .then((result) => {
                    previewContent = result.content;
                })
                .catch(() => {
                    /* preview failure is non-critical */
                });
        }, DEBOUNCE_MS);
    }

    // --- Parsed dataset entries (for preserving images on save) ---

    let parsedDatasetRaw: ConfigDatasetEntry[] = $state([]);

    /** Convert extras dict to KeyValueEntry array (sorted by key for stable ordering). */
    function extrasToEntries(extras?: Record<string, unknown>): KeyValueEntry[] {
        if (!extras) {
            return [];
        }
        return Object.entries(extras)
            .sort(([a], [b]) => a.localeCompare(b))
            .map(([key, value]) => ({ key, value: String(value) }));
    }

    // --- Load on mount / dataset change ---

    $effect(() => {
        const name = datasetName;
        if (!name) {
            return;
        }
        loadConfig();
    });

    /** Populate structured fields from parsed config. */
    function populateFields(p: Config, content: string, configPath: string) {
        previewContent = content;
        rawContent = loadedRawContent = content;
        loadedConfigPath = configPath;

        apiUrl = loadedApiUrl = p.api?.url ?? '';
        apiModelName = loadedApiModelName = p.api?.model_name ?? '';
        promptName = loadedPromptName = p.prompt?.name ?? '';
        maxTokens = loadedMaxTokens = p.settings?.max_tokens ?? null;
        imageQuality = loadedImageQuality = p.settings?.image_quality ?? null;
        rounds = loadedRounds = p.rounds ?? null;
        overwrite = loadedOverwrite = p.overwrite_captions ?? false;
        storeConversation = loadedStoreConversation = p.settings?.store_conversation ?? false;
        reasoningEnabled = loadedReasoningEnabled = p.reasoning?.enable ?? false;
        reasoningEffort = loadedReasoningEffort = p.reasoning?.thinking_effort ?? 'low';
        reasoningExcludeOutput = loadedReasoningExcludeOutput =
            p.reasoning?.exclude_from_output ?? true;
        envName = loadedEnvName = p.env ?? '';

        // Parse dataset entries
        const rawEntries = p.dataset ?? [];
        parsedDatasetRaw = rawEntries;
        datasetEntries = loadedDatasetEntries = rawEntries.map((entry) => ({
            path: entry.path ?? '',
            extras: extrasToEntries(entry.extras as Record<string, unknown> | undefined),
            imageCount: entry.images?.length ?? 0
        }));
    }

    async function loadConfig() {
        isLoading = true;
        error = null;

        try {
            const templatesReady = $templates.loaded ? Promise.resolve() : refreshTemplates();
            const [config] = await Promise.all([fetchConfig(datasetName), templatesReady]);
            validationErrors = config.validation_error ?? [];
            populateFields(config.parsed, config.content, config.config_path);
        } catch (e) {
            error = friendlyErrorMessage(e, 'Failed to load config');
        } finally {
            isLoading = false;
        }
    }

    // --- Trigger live preview on any field change ---

    $effect(() => {
        // Touch all reactive values so Svelte tracks them
        void apiUrl;
        void apiModelName;
        void promptName;
        void maxTokens;
        void imageQuality;
        void rounds;
        void overwrite;
        void storeConversation;
        void reasoningEnabled;
        void reasoningEffort;
        void reasoningExcludeOutput;
        void envName;
        void datasetName;
        // Touch dataset entries (array + each entry's fields)
        void datasetEntries;
        for (const entry of datasetEntries) {
            void entry.path;
            for (const e of entry.extras) {
                void e.key;
                void e.value;
            }
        }

        if (!isLoading && !error) {
            schedulePreview();
        }
    });

    // --- Save structured fields via PATCH ---

    function addDatasetEntry() {
        datasetEntries = [...datasetEntries, { path: '', extras: [], imageCount: 0 }];
    }

    function removeDatasetEntry(index: number) {
        datasetEntries = datasetEntries.filter((_, i) => i !== index);
    }

    function updateEntryPath(index: number, newPath: string) {
        datasetEntries = datasetEntries.map((e, i) => (i === index ? { ...e, path: newPath } : e));
    }

    // --- View mode switching ---

    let previousView: 'simplified' | 'advanced' = $state('simplified');

    $effect(() => {
        if (activeView === previousView) {
            return;
        }
        if (activeView === 'advanced') {
            // Transfer current preview content (includes unsaved simplified changes)
            rawContent = previewContent;
            loadedRawContent = previewContent;
        } else if (rawDirty) {
            // Switching back with unsaved raw changes — reload from server
            loadConfig();
        }
        previousView = activeView;
    });

    async function handleSave() {
        isSaving = true;
        saveError = null;

        try {
            if (activeView === 'advanced') {
                // PUT: full content replacement
                const result = await updateConfig(datasetName, rawContent);
                previewContent = result.content;
                rawContent = loadedRawContent = result.content;
                // Re-populate structured fields from the new content
                populateFields(result.parsed, result.content, result.config_path);
                validationErrors = result.validation_error ?? [];
            } else {
                // PATCH: merge structured fields
                const result = await patchConfig(datasetName, buildPatch());
                previewContent = result.content;
                rawContent = loadedRawContent = result.content;
                validationErrors = result.validation_error ?? [];
                // Update loaded snapshot so dirty resets
                loadedApiUrl = apiUrl;
                loadedApiModelName = apiModelName;
                loadedPromptName = promptName;
                loadedMaxTokens = maxTokens as number | null;
                loadedImageQuality = imageQuality as 'auto' | 'high' | 'low' | null;
                loadedRounds = rounds as number | null;
                loadedOverwrite = overwrite;
                loadedStoreConversation = storeConversation;
                loadedReasoningEnabled = reasoningEnabled;
                loadedReasoningEffort = reasoningEffort;
                loadedReasoningExcludeOutput = reasoningExcludeOutput;
                loadedEnvName = envName;
                loadedDatasetEntries = datasetEntries.map((e) => ({
                    ...e,
                    extras: e.extras.map((kv) => ({ ...kv }))
                }));
            }

            handleConfigSaved();
        } catch (e) {
            saveError = friendlyErrorMessage(e, 'Failed to save config');
        } finally {
            isSaving = false;
        }
    }

    // --- Config history ---

    let configVersion = $state(0);

    function handleConfigSaved() {
        configVersion++;
        toast.success('Config saved');
        onsaved?.();
    }
</script>

<div class="flex h-full flex-col">
    <!-- Scrollable content -->
    <div class="flex-1 space-y-5 overflow-y-auto p-4">
        {#if isLoading}
            <SpinnerBlock class="py-8" size="h-5 w-5" label="Loading config…" />
        {:else if error}
            <div class="alert-error">{error}</div>
        {:else}
            {#if saveError}
                <div class="alert-error">{saveError}</div>
            {/if}
            {#if validationErrors.length > 0}
                <div class="alert-warning">
                    <p class="font-medium">Config validation issues:</p>
                    <ul class="mt-1 list-inside list-disc">
                        {#each validationErrors as err (err.loc.join('.'))}
                            <li>{err.loc.join('.')}: {err.msg}</li>
                        {/each}
                    </ul>
                </div>
            {/if}

            <!-- ═══ View mode toggle ═══ -->
            <CompactPillTabs bind:value={activeView} class="h-full">
                <Tab id="simplified" label="Form" icon={SvgFile} class="flex flex-col gap-4">
                    <!-- ═══ Simplified mode: structured form ═══ -->

                    <!-- ═══ Info: Dataset source & path ═══ -->
                    <section class="space-y-2">
                        <h3 class="section-heading">Dataset</h3>
                        <div class="space-y-1">
                            <div class="flex items-center gap-2">
                                {#if source === 'upload'}
                                    <span class="badge-muted badge-sm">Managed</span>
                                    <span class="text-xs text-gray-400"
                                        >Files stored in yadc state directory</span
                                    >
                                {:else}
                                    <span class="badge-muted badge-sm">External</span>
                                    <span class="text-xs text-gray-400"
                                        >References paths outside yadc</span
                                    >
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

                        {#each datasetEntries as entry, i (i)}
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
                                                oninput={(e) =>
                                                    updateEntryPath(i, e.currentTarget.value)}
                                            />
                                        </div>
                                        {#if datasetEntries.length > 1}
                                            <button
                                                class="mt-6 cursor-pointer p-1 text-gray-500 transition-colors hover:text-error"
                                                onclick={() => removeDatasetEntry(i)}
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
                                            {entry.imageCount} inline image{entry.imageCount !== 1
                                                ? 's'
                                                : ''}
                                            — edit in raw TOML view
                                        </p>
                                    {/if}

                                    <!-- Empty entry warning -->
                                    {#if !entry.path && entry.imageCount === 0}
                                        <p class="text-xs text-warning">
                                            Empty entry — set a path or add images.
                                        </p>
                                    {/if}

                                    <!-- Extras -->
                                    <div>
                                        <div
                                            class="mb-1.5 text-xs font-medium tracking-wide text-gray-300 uppercase"
                                        >
                                            Variables
                                        </div>
                                        <KeyValueEditor
                                            bind:entries={datasetEntries[i].extras}
                                            idPrefix="entry-{i}-extras"
                                            keyPlaceholder="variable name"
                                            valuePlaceholder="value"
                                        />
                                    </div>
                                </div>
                            </div>
                        {/each}

                        <button class="btn-secondary w-full" onclick={addDatasetEntry}>
                            <SvgPlus class="mr-1 inline-block h-4 w-4" />
                            Add Path
                        </button>
                    </section>

                    <!-- ═══ Section: API ═══ -->
                    <section class="space-y-3">
                        <h3 class="section-heading">API</h3>

                        <div class="grid grid-cols-2 gap-3">
                            <div class="col-span-2">
                                <label class="label" for="config-api-url">API URL</label>
                                <input
                                    id="config-api-url"
                                    type="text"
                                    bind:value={apiUrl}
                                    class="input"
                                    placeholder="http://localhost:11434"
                                />
                                <p class="help-text">
                                    Hostname and port only — no additional path segments.
                                </p>
                            </div>

                            <div>
                                <label class="label" for="config-model">Model Name</label>
                                <input
                                    id="config-model"
                                    type="text"
                                    bind:value={apiModelName}
                                    class="input"
                                    placeholder="gemma3"
                                />
                            </div>

                            <div>
                                <label class="label" for="config-env">Environment</label>
                                <input
                                    id="config-env"
                                    type="text"
                                    bind:value={envName}
                                    class="input"
                                    placeholder="default"
                                />
                                <p class="help-text">
                                    Set this to configure API settings outside the TOML file.
                                </p>
                            </div>
                        </div>
                    </section>

                    <!-- ═══ Section: Prompt ═══ -->
                    <section class="space-y-3">
                        <h3 class="section-heading">Prompt</h3>

                        <div>
                            <label class="label" for="config-prompt">Template</label>
                            <select
                                id="config-prompt"
                                class="input cursor-pointer"
                                value={promptName}
                                onchange={(e) => {
                                    promptName = e.currentTarget.value;
                                }}
                            >
                                <option value="">(default)</option>
                                {#each $templates.items as t (t.name)}
                                    <option value={t.name}
                                        >{t.name}{t.source === 'builtin'
                                            ? ' (built-in)'
                                            : ''}</option
                                    >
                                {/each}
                            </select>
                            <p class="help-text">
                                Prompt template used during captioning. Leave empty for the default
                                template.
                            </p>
                        </div>
                    </section>

                    <!-- ═══ Section: Options & Reasoning (shared component) ═══ -->
                    <CaptionOptionsFields
                        bind:maxTokens
                        bind:imageQuality
                        bind:rounds
                        bind:overwrite
                        bind:reasoningEnabled
                        bind:reasoningEffort
                        bind:storeConversation
                        bind:reasoningExcludeOutput
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
                        {#if simplifiedDirty}
                            <p class="text-xs text-accent">Previewing unsaved changes.</p>
                        {:else}
                            <p class="text-xs text-gray-500">Current config on disk.</p>
                        {/if}
                        <div class="max-h-[40vh] min-h-30 overflow-y-auto">
                            <TomlEditor value={previewContent} editable={false} />
                        </div>
                    </section>

                    <div class="h-2"></div>
                </Tab>
                <Tab id="advanced" label="Advanced" icon={SvgEdit} class="h-full">
                    <TomlEditor
                        value={rawContent}
                        editable={true}
                        onchange={(v) => (rawContent = v)}
                    />
                </Tab>
                <Tab id="history" label="History" icon={SvgHistory} class="h-full">
                    <ConfigHistory {datasetName} {configVersion} onsaved={handleConfigSaved} />
                </Tab>
            </CompactPillTabs>
        {/if}
    </div>

    <!-- Footer: sticky save button -->
    {#if !isLoading && !error}
        <div class="flex-shrink-0 border-t border-border p-4">
            <div class="flex gap-2">
                <button class="btn-secondary flex-1" onclick={loadConfig} disabled={isSaving}>
                    <SvgRefresh class="mr-1 inline-block h-4 w-4" />
                    Reload
                </button>
                <button
                    class="btn-primary flex-1"
                    onclick={handleSave}
                    disabled={isSaving || !dirty}
                >
                    {isSaving ? 'Saving…' : 'Save Config'}
                </button>
            </div>
        </div>
    {/if}
</div>
