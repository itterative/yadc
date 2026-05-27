<script lang="ts">
    import TomlEditor from '$lib/components/ui/TomlEditor.svelte';
    import {
        fetchPromptPreview,
        type PromptPreview as PromptPreviewData
    } from '$lib/stores/datasetImages';
    import { templates, refreshTemplates } from '$lib/stores/templates';
    import { friendlyErrorMessage } from '$lib/api';

    interface Props {
        datasetName: string;
        imageId: number | null;
    }

    let { datasetName, imageId }: Props = $props();

    let promptPreview: PromptPreviewData | null = $state(null);
    let isLoadingPreview = $state(false);
    let previewError: string | null = $state(null);
    let previewTemplateName = $state('');
    let showPreview = $state(true);

    // Load preview when image changes
    $effect(() => {
        const id = imageId;
        const dsName = datasetName;
        if (id === null) {
            promptPreview = null;
            previewError = null;
            return;
        }

        let cancelled = false;

        (async () => {
            try {
                if (!$templates.loaded) {
                    await refreshTemplates();
                }
                const data = await fetchPromptPreview(dsName, id);
                if (cancelled) {
                    return;
                }
                promptPreview = data;
                previewError = null;
            } catch (e) {
                if (cancelled) {
                    return;
                }
                if (!$templates.loaded) {
                    /* store stays empty */
                }
                previewError = friendlyErrorMessage(e, 'Failed to preview prompt');
            }
        })();

        return () => {
            cancelled = true;
        };
    });

    async function handlePreviewPrompt() {
        if (imageId === null) {
            return;
        }
        isLoadingPreview = true;
        previewError = null;
        promptPreview = null;
        showPreview = true;

        try {
            const opts = previewTemplateName ? { template_name: previewTemplateName } : {};
            promptPreview = await fetchPromptPreview(datasetName, imageId, opts);
        } catch (e) {
            previewError = friendlyErrorMessage(e, 'Failed to preview prompt');
        } finally {
            isLoadingPreview = false;
        }
    }
</script>

<div class="pt-2">
    <div class="mb-2 flex items-center gap-2">
        <button
            class="cursor-pointer text-sm text-accent hover:text-accent-hover"
            onclick={() => (showPreview = !showPreview)}
        >
            {showPreview ? '▾' : '▸'} Preview Prompt
        </button>
    </div>

    {#if showPreview}
        <div class="space-y-3">
            <div class="flex items-end gap-2">
                <div class="flex-1">
                    <label class="label mb-1" for="preview-template">Template</label>
                    <select
                        id="preview-template"
                        bind:value={previewTemplateName}
                        class="input-sm"
                        onchange={() => handlePreviewPrompt()}
                    >
                        <option value="">(default)</option>
                        {#each $templates.items as t (t.name)}
                            <option value={t.name}
                                >{t.name} {t.source === 'builtin' ? '(built-in)' : ''}</option
                            >
                        {/each}
                    </select>
                </div>
                <button
                    class="btn-primary px-3 py-1.5"
                    onclick={handlePreviewPrompt}
                    disabled={isLoadingPreview}
                >
                    {isLoadingPreview ? 'Loading...' : 'Render'}
                </button>
            </div>

            {#if previewError}
                <p class="text-sm text-error">{previewError}</p>
            {/if}

            {#if promptPreview}
                {@const ctxEntries = Object.entries(promptPreview.template_context).filter(
                    ([k]) => k !== 'caption_suffix' && k !== 'toml_suffix' && k !== 'history_suffix'
                )}
                <div>
                    <h4 class="mb-1 text-xs font-medium text-gray-400">System Prompt</h4>
                    <pre
                        class="max-h-40 overflow-y-auto rounded-lg bg-gray-800 p-3 text-sm whitespace-pre-wrap text-gray-200">{promptPreview.system_prompt}</pre>
                </div>
                <div>
                    <h4 class="mb-1 text-xs font-medium text-gray-400">User Prompt</h4>
                    <pre
                        class="max-h-60 overflow-y-auto rounded-lg bg-gray-800 p-3 text-sm whitespace-pre-wrap text-gray-200">{promptPreview.user_prompt}</pre>
                </div>
                <details class="group">
                    <summary class="cursor-pointer text-xs text-gray-400 hover:text-gray-300"
                        >Template context ({ctxEntries.length} variables)</summary
                    >
                    <div class="mt-1">
                        <TomlEditor
                            class="text-sm"
                            value={promptPreview.template_context_toml}
                            editable={false}
                        />
                    </div>
                </details>
            {/if}
        </div>
    {/if}
</div>
