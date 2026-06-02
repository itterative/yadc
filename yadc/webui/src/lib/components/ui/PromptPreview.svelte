<script lang="ts">
    import TomlEditor from '$lib/components/ui/TomlEditor.svelte';
    import {
        fetchPromptPreview,
        type PromptPreview as PromptPreviewData
    } from '$lib/stores/datasetImages';
    import { captionOptions } from '$lib/stores/captionActions';
    import { friendlyErrorMessage } from '$lib/api';

    interface Props {
        datasetName: string;
        imageId: number | null;
    }

    let { datasetName, imageId }: Props = $props();

    let promptPreview: PromptPreviewData | null = $state(null);
    let isLoadingPreview = $state(false);
    let previewError: string | null = $state(null);

    // Track the previously rendered image so we only clear the preview on image
    // changes, not on template edits (keeps the old preview visible during the
    // 200 ms debounce window so the user doesn't see a flash of empty state).
    let lastRenderedImageId: number | null = null;

    $effect(() => {
        const id = imageId;
        const dsName = datasetName;
        // Read template fields so the effect re-runs when they change.
        const templateContent = $captionOptions?.prompt_template;
        const templateName = $captionOptions?.prompt_name;

        if (id === null) {
            promptPreview = null;
            previewError = null;
            lastRenderedImageId = null;
            return;
        }

        // Clear previous preview only when the image actually changes.
        if (id !== lastRenderedImageId) {
            promptPreview = null;
            previewError = null;
            lastRenderedImageId = id;
        }

        let cancelled = false;
        let timer: ReturnType<typeof setTimeout> | null = null;

        const options: { template?: string; template_name?: string } = {};
        if (templateContent) {
            options.template = templateContent;
        } else if (templateName) {
            options.template_name = templateName;
        }

        timer = setTimeout(() => {
            if (cancelled) {
                return;
            }
            isLoadingPreview = true;
            fetchPromptPreview(dsName, id, options)
                .then((data) => {
                    if (cancelled) {
                        return;
                    }
                    promptPreview = data;
                })
                .catch((e) => {
                    if (cancelled) {
                        return;
                    }
                    previewError = friendlyErrorMessage(e, 'Failed to preview prompt');
                })
                .finally(() => {
                    if (cancelled) {
                        return;
                    }
                    isLoadingPreview = false;
                });
        }, 200);

        return () => {
            cancelled = true;
            if (timer !== null) {
                clearTimeout(timer);
            }
        };
    });
</script>

<div>
    <div class="mb-2 flex items-center justify-between">
        <h3 class="text-sm font-medium text-gray-300">Preview Prompt</h3>
        {#if isLoadingPreview}
            <span class="text-xs text-gray-500">Rendering…</span>
        {/if}
    </div>
    <p class="mb-2 text-xs text-gray-500">
        Live preview of the system and user prompts sent to the model. Updates automatically when
        the image or prompt template changes.
    </p>
    <div class="space-y-3">
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
                        class="rounded-md border border-border text-sm"
                        value={promptPreview.template_context_toml}
                        editable={false}
                    />
                </div>
            </details>
        {/if}
    </div>
</div>
