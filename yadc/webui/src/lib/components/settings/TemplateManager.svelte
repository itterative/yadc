<script lang="ts">
    import JinjaEditor from '$lib/components/ui/JinjaEditor.svelte';
    import { confirmDialog } from '$lib/stores/confirm';
    import SpinnerBlock from '$lib/components/ui/SpinnerBlock.svelte';
    import SvgDelete from '$lib/icons/SvgDelete.svelte';
    import SvgPlus from '$lib/icons/SvgPlus.svelte';
    import { friendlyErrorMessage } from '$lib/api';
    import {
        templates,
        refreshTemplates,
        fetchTemplate,
        saveTemplate,
        deleteTemplate
    } from '$lib/stores/templates';

    interface Props {
        /** Reload data when this becomes true (e.g. dialog opens). */
        open: boolean;
    }

    let { open }: Props = $props();

    let selectedTemplateName = $state('');
    let templateContent = $state('');
    let templateSource: 'user' | 'builtin' | '' = $state('');
    let isLoadingTemplate = $state(false);
    let templateDirty = $state(false);
    let isSavingTemplate = $state(false);
    let templateError: string | null = $state(null);
    let isNewTemplate = $state(false);
    let newTemplateName = $state('');

    $effect(() => {
        if (open) {
            templateError = null;
            templateDirty = false;
            isNewTemplate = false;
            loadTemplates();
        }
    });

    async function loadTemplates() {
        try {
            await refreshTemplates();
        } catch (e) {
            templateError = friendlyErrorMessage(e, 'Failed to load templates');
        }
    }

    async function selectTemplate(name: string) {
        selectedTemplateName = name;
        isLoadingTemplate = true;
        templateError = null;
        templateDirty = false;
        isNewTemplate = false;
        try {
            const info = await fetchTemplate(name);
            templateContent = info.content;
            templateSource = info.source;
        } catch (e) {
            templateError = friendlyErrorMessage(e, 'Failed to load template');
            templateContent = '';
            templateSource = '';
        } finally {
            isLoadingTemplate = false;
        }
    }

    function handleTemplateContentChange(value: string) {
        templateContent = value;
        templateDirty = true;
    }

    async function saveTemplateContent() {
        const name = isNewTemplate ? newTemplateName.trim() : selectedTemplateName;
        if (!name) {
            templateError = 'Template name is required';
            return;
        }

        isSavingTemplate = true;
        templateError = null;
        try {
            const info = await saveTemplate(name, templateContent);
            isNewTemplate = false;
            selectedTemplateName = info.name;
            templateSource = info.source;
            templateDirty = false;
            await refreshTemplates();
        } catch (e) {
            templateError = friendlyErrorMessage(e, 'Failed to save template');
        } finally {
            isSavingTemplate = false;
        }
    }

    async function handleDeleteTemplate(name: string) {
        const ok = await confirmDialog.danger(`Delete template "${name}"?`);
        if (!ok) {
            return;
        }
        try {
            await deleteTemplate(name);
            if (selectedTemplateName === name) {
                selectedTemplateName = '';
                templateContent = '';
                templateSource = '';
            }
            await refreshTemplates();
        } catch (e) {
            templateError = friendlyErrorMessage(e, 'Failed to delete template');
        }
    }

    function startNewTemplate() {
        isNewTemplate = true;
        newTemplateName = '';
        templateContent = '';
        templateSource = '';
        templateDirty = false;
        templateError = null;
        selectedTemplateName = '';
    }

    function cancelNewTemplate() {
        isNewTemplate = false;
        newTemplateName = '';
        templateError = null;
    }
</script>

{#if templateError}
    <div class="alert-error">{templateError}</div>
{/if}

<div class="flex h-[50vh] gap-4">
    <!-- Template list sidebar -->
    <div class="w-48 shrink-0 space-y-1 overflow-y-auto">
        {#each $templates.items as t (t.name)}
            <div
                role="button"
                tabindex="0"
                class="group w-full cursor-pointer rounded-lg px-3 py-2 text-left text-sm transition-colors {selectedTemplateName ===
                    t.name && !isNewTemplate
                    ? 'border border-border bg-bg text-accent'
                    : 'text-gray-300 hover:bg-bg/50'}"
                onclick={() => selectTemplate(t.name)}
                onkeydown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                        selectTemplate(t.name);
                    }
                }}
            >
                <div class="flex items-center justify-between gap-1">
                    <span class="truncate">{t.name}</span>
                    <span class="flex shrink-0 items-center gap-0.5">
                        {#if t.source === 'builtin'}
                            <span class="text-[10px] text-gray-500 uppercase">built-in</span>
                        {/if}
                        {#if t.source === 'user'}
                            <button
                                class="cursor-pointer p-0.5 text-gray-500 opacity-0 transition-opacity group-hover:opacity-100 hover:text-error max-lg:opacity-100"
                                title="Delete template"
                                onclick={(e) => {
                                    e.stopPropagation();
                                    handleDeleteTemplate(t.name);
                                }}
                            >
                                <SvgDelete class="h-3 w-3" />
                            </button>
                        {/if}
                    </span>
                </div>
            </div>
        {/each}

        <!-- New template button -->
        <button
            class="flex w-full cursor-pointer items-center justify-center gap-2 rounded-lg px-3 py-2 text-sm text-gray-400 transition-colors hover:bg-bg/50 hover:text-gray-200"
            onclick={startNewTemplate}
        >
            <SvgPlus class="h-3.5 w-3.5" />
            New
        </button>
    </div>

    <!-- Template editor -->
    <div class="flex min-w-0 flex-1 flex-col">
        {#if isNewTemplate}
            <div class="mb-2 flex items-center gap-2">
                <input
                    type="text"
                    bind:value={newTemplateName}
                    class="input-sm"
                    placeholder="Template name"
                />
                <button class="btn-secondary px-3 py-1.5 text-xs" onclick={cancelNewTemplate}>
                    Cancel
                </button>
            </div>
        {:else if selectedTemplateName}
            <div class="mb-2 flex items-center justify-between">
                <h3 class="text-sm font-medium text-gray-300">
                    {selectedTemplateName}.jinja
                    {#if templateSource === 'builtin'}
                        <span class="ml-1 text-xs text-gray-500">(built-in)</span>
                    {/if}
                </h3>
                {#if templateDirty}
                    <button
                        class="btn-primary px-3 py-1.5 text-xs"
                        onclick={saveTemplateContent}
                        disabled={isSavingTemplate}
                    >
                        {isSavingTemplate ? 'Saving…' : 'Save'}
                    </button>
                {/if}
            </div>
        {/if}

        {#if isLoadingTemplate}
            <SpinnerBlock class="py-12" />
        {:else if selectedTemplateName || isNewTemplate}
            <div class="min-h-0 flex-1 overflow-hidden">
                <JinjaEditor
                    class="rounded-md border border-border"
                    value={templateContent}
                    onchange={handleTemplateContentChange}
                />
            </div>
            {#if templateSource === 'builtin' && !templateDirty}
                <p class="mt-2 text-xs text-gray-500">
                    Built-in template. Edit and save to create a user override.
                </p>
            {/if}
        {:else}
            <div class="flex h-full items-center justify-center text-sm text-gray-500">
                Select or create a template
            </div>
        {/if}
    </div>
</div>
