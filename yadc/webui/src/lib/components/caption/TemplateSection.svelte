<script lang="ts">
    import { templates, refreshTemplates, fetchTemplate, saveTemplate } from '$lib/stores/templates';
    import { friendlyErrorMessage } from '$lib/api';
    import JinjaEditor from '$lib/components/ui/JinjaEditor.svelte';
    import SvgPlus from '$lib/icons/SvgPlus.svelte';
    import SpinnerBlock from '$lib/components/ui/SpinnerBlock.svelte';

    interface Props {
        /** Currently selected template name. Bindable so the parent can seed it from
         *  localStorage / dataset config defaults. */
        selectedTemplate?: string;
        /** Current content of the template editor. Bindable — the parent reads it
         *  to assemble `prompt_template` for the captioning request. */
        templateContent?: string;
        /** True when the content has been edited since the last load/save. Bindable —
         *  the parent uses it to decide between `prompt_name` and `prompt_template`. */
        templateDirty?: boolean;
        /** True when the user is creating a new template (vs. editing an existing one).
         *  Bindable — the parent reads it to compute `effectiveTemplateName`. */
        isNewTemplate?: boolean;
        /** Name of the new template being created. Bindable — the parent reads it
         *  to compute `effectiveTemplateName` and to persist. */
        newTemplateName?: string;
        /** The dataset's default template name (from TOML config). Used for the
         *  diff dot next to the template dropdown. */
        datasetDefaultTemplate?: string;
    }

    let {
        selectedTemplate = $bindable(''),
        templateContent = $bindable(''),
        templateDirty = $bindable(false),
        isNewTemplate = $bindable(false),
        newTemplateName = $bindable(''),
        datasetDefaultTemplate = ''
    }: Props = $props();

    // --- Section-internal state (not exposed to the parent) ---

    let templateSource: 'user' | 'builtin' | '' = $state('');
    let isLoadingTemplate = $state(false);
    let isSavingTemplate = $state(false);
    let templateSaveError: string | null = $state(null);
    let templatesError: string | null = $state(null);
    let isLoadingTemplates = $state(false);

    let templateOverridden = $derived(selectedTemplate !== datasetDefaultTemplate);

    // Load templates list on mount. May seed `selectedTemplate = 'default'` if the
    // list has a default and nothing is selected. The parent's `loadDatasetDefaults`
    // races with this — but its writes always win because they fire after (and
    // check the same emptiness precondition).
    $effect(() => {
        void loadTemplateList();
    });

    // Load the selected template's content when `selectedTemplate` changes
    // (and we're not in new-template mode).
    $effect(() => {
        const name = selectedTemplate;
        if (!name || isNewTemplate) {
            return;
        }

        let cancelled = false;
        isLoadingTemplate = true;
        (async () => {
            try {
                const info = await fetchTemplate(name);
                if (cancelled) {
                    return;
                }
                templateContent = info.content;
                templateSource = info.source;
                templateDirty = false;
            } catch {
                if (cancelled) {
                    return;
                }
                templateContent = '';
                templateSource = '';
            } finally {
                if (!cancelled) {
                    isLoadingTemplate = false;
                }
            }
        })();

        return () => {
            cancelled = true;
        };
    });

    async function loadTemplateList() {
        isLoadingTemplates = true;
        templatesError = null;
        try {
            const list = await refreshTemplates();
            const hasDefault = list.some((t) => t.name === 'default');
            if (hasDefault && !selectedTemplate) {
                selectedTemplate = 'default';
            }
        } catch {
            templatesError = 'Failed to load templates';
        } finally {
            isLoadingTemplates = false;
        }
    }

    function handleTemplateChange() {
        templateDirty = false;
        isNewTemplate = false;
        newTemplateName = '';
        templateSaveError = null;
    }

    function startNewTemplate() {
        isNewTemplate = true;
        newTemplateName = '';
        templateContent = '';
        templateSource = '';
        templateDirty = false;
        templateSaveError = null;
    }

    function cancelNewTemplate() {
        isNewTemplate = false;
        newTemplateName = '';
        templateSaveError = null;
        if (selectedTemplate) {
            // Force a re-load of the previously-selected template
            const name = selectedTemplate;
            selectedTemplate = '';
            selectedTemplate = name;
        }
    }

    function handleTemplateContentChange() {
        templateDirty = true;
    }

    async function handleSaveTemplate() {
        const name = newTemplateName.trim();
        if (!name) {
            templateSaveError = 'Template name is required';
            return;
        }

        isSavingTemplate = true;
        templateSaveError = null;
        try {
            const info = await saveTemplate(name, templateContent);
            isNewTemplate = false;
            selectedTemplate = info.name;
            templateSource = info.source;
            templateDirty = false;
            await loadTemplateList();
        } catch (e) {
            templateSaveError = friendlyErrorMessage(e, 'Failed to save template');
        } finally {
            isSavingTemplate = false;
        }
    }
</script>

<section class="space-y-3">
    <h3 class="section-heading">Template</h3>

    {#if templatesError}
        <div class="alert-error">{templatesError}</div>
    {/if}

    <div class="flex items-end gap-2">
        <div class="flex-1">
            {#if isNewTemplate}
                <label class="label" for="new-tpl-name">New Template Name</label>
                <input
                    id="new-tpl-name"
                    type="text"
                    bind:value={newTemplateName}
                    class="input"
                    placeholder="my-template"
                />
            {:else}
                <div class="mb-1 flex items-center gap-1.5">
                    <label class="label mb-0" for="caption-template">Template</label>
                    {#if templateOverridden}
                        <span class="diff-dot" title="Differs from dataset config"></span>
                    {/if}
                </div>
                <select
                    id="caption-template"
                    class="input cursor-pointer"
                    bind:value={selectedTemplate}
                    onchange={handleTemplateChange}
                    disabled={isLoadingTemplates}
                >
                    {#each $templates.items as t (t.name)}
                        <option value={t.name}>
                            {t.name}{#if t.source === 'builtin'}
                                (built-in){/if}
                        </option>
                    {/each}
                </select>
            {/if}
        </div>

        <div class="flex gap-2 pb-px">
            {#if isNewTemplate}
                <button class="btn-secondary px-3 py-2" onclick={cancelNewTemplate}>
                    Cancel
                </button>
                <button
                    class="btn-primary px-3 py-2"
                    onclick={handleSaveTemplate}
                    disabled={isSavingTemplate || !newTemplateName.trim()}
                >
                    {isSavingTemplate ? 'Saving…' : 'Save'}
                </button>
            {:else}
                <button
                    class="btn-secondary px-3 py-2"
                    onclick={startNewTemplate}
                    title="Create new template"
                >
                    <SvgPlus class="h-4 w-4" />
                </button>
                {#if templateDirty && selectedTemplate}
                    <button
                        class="btn-primary px-3 py-2"
                        onclick={handleSaveTemplate}
                        disabled={isSavingTemplate}
                        title="Save changes to template"
                    >
                        {isSavingTemplate ? '…' : 'Save'}
                    </button>
                {/if}
            {/if}
        </div>
    </div>

    {#if templateSaveError}
        <p class="text-xs text-error">{templateSaveError}</p>
    {/if}

    <div class="relative h-64">
        {#if isLoadingTemplate}
            <SpinnerBlock class="py-8" size="h-4 w-4" label="Loading template…" />
        {:else}
            <JinjaEditor
                class="h-full rounded-md border border-border bg-surface text-sm"
                bind:value={templateContent}
                onchange={handleTemplateContentChange}
            />
        {/if}
    </div>

    {#if templateSource === 'builtin' && !templateDirty}
        <p class="text-xs text-gray-500">
            Built-in template. Edit above and save to create a user override.
        </p>
    {/if}
</section>
