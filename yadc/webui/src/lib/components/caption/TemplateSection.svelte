<script lang="ts">
    import {
        templates,
        refreshTemplates,
        fetchTemplate,
        deleteTemplate
    } from '$lib/stores/templates';
    import { friendlyErrorMessage } from '$lib/api';
    import { confirmDialog } from '$lib/stores/confirm';
    import { toast } from '$lib/stores/toasts';
    import JinjaEditor from '$lib/components/ui/JinjaEditor.svelte';
    import Card from '$lib/components/ui/Card.svelte';
    import ActionBar from '$lib/components/ui/ActionBar.svelte';
    import ActionBarItem from '$lib/components/ui/ActionBarItem.svelte';
    import SpinnerBlock from '$lib/components/ui/SpinnerBlock.svelte';
    import SvgPlus from '$lib/icons/SvgPlus.svelte';
    import SvgEdit from '$lib/icons/SvgEdit.svelte';
    import SvgDelete from '$lib/icons/SvgDelete.svelte';
    import { EditTemplateDialog } from '$lib/components/templates';

    interface Props {
        /** Currently selected template name. Bindable so the parent can seed it from
         *  localStorage / dataset config defaults. */
        selectedTemplate?: string;
        /** The dataset's default template name (from TOML config). Used for the
         *  diff dot next to the template dropdown. */
        datasetDefaultTemplate?: string;
    }

    let { selectedTemplate = $bindable(''), datasetDefaultTemplate = '' }: Props = $props();

    // --- Section-internal state (not exposed to the parent) ---

    let templatesError: string | null = $state(null);
    let isLoadingTemplates = $state(false);

    /** Content of the currently selected template, loaded from the API for the
     *  readonly card. Refreshed when `selectedTemplate` changes or after edits. */
    let currentContent = $state('');
    let isLoadingContent = $state(false);
    let contentError: string | null = $state(null);

    /** Dialog visibility. Two flows share the same `EditTemplateDialog`:
     *  edit (templateName set) and add (templateName=null). */
    let editDialogOpen = $state(false);
    let addDialogOpen = $state(false);

    let templateOverridden = $derived(selectedTemplate !== datasetDefaultTemplate);
    let selectedListItem = $derived(
        $templates.items.find((t) => t.name === selectedTemplate) ?? null
    );
    let canDelete = $derived(selectedListItem?.source === 'user');

    // Load templates list on mount. May seed `selectedTemplate = 'default'` if the
    // list has a default and nothing is selected. The parent's `loadDatasetDefaults`
    // races with this — but its writes always win because they fire after (and
    // check the same emptiness precondition).
    $effect(() => {
        void loadTemplateList();
    });

    // Load the selected template's content when `selectedTemplate` changes
    // (used to render the readonly card).
    $effect(() => {
        const name = selectedTemplate;
        if (!name) {
            currentContent = '';
            contentError = null;
            return;
        }

        let cancelled = false;
        isLoadingContent = true;
        contentError = null;
        (async () => {
            try {
                const info = await fetchTemplate(name);
                if (cancelled) {
                    return;
                }
                currentContent = info.content;
            } catch (e) {
                if (cancelled) {
                    return;
                }
                currentContent = '';
                contentError = friendlyErrorMessage(e, 'Failed to load template');
            } finally {
                if (!cancelled) {
                    isLoadingContent = false;
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

    async function handleDelete() {
        if (!selectedListItem || !canDelete) {
            return;
        }
        const ok = await confirmDialog.danger(
            `Delete template "${selectedListItem.name}"? This action cannot be undone.`
        );
        if (!ok) {
            return;
        }
        try {
            await deleteTemplate(selectedListItem.name);
            await loadTemplateList();
            // Pick a sensible fallback so the captioning options stay valid.
            const remaining = $templates.items;
            const fallback = remaining.find((t) => t.name === 'default') ?? remaining[0];
            selectedTemplate = fallback?.name ?? '';
        } catch (e) {
            toast.error(friendlyErrorMessage(e, 'Failed to delete template'));
        }
    }

    async function handleEditSaved() {
        editDialogOpen = false;
        await loadTemplateList();
        // Trigger a content reload by clearing and re-setting (avoids depending on
        // the API returning the latest content; forces the load effect to re-run).
        const name = selectedTemplate;
        selectedTemplate = '';
        selectedTemplate = name;
    }

    function handleAddSaved(name: string) {
        addDialogOpen = false;
        void loadTemplateList().then(() => {
            selectedTemplate = name;
        });
    }
</script>

<section class="space-y-3">
    <h3 class="section-heading">Template</h3>

    {#if templatesError}
        <div class="alert-error">{templatesError}</div>
    {/if}

    <div class="flex items-end gap-2">
        <div class="flex-1">
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
                disabled={isLoadingTemplates}
            >
                {#each $templates.items as t (t.name)}
                    <option value={t.name}>
                        {t.name}{#if t.source === 'builtin'}
                            (built-in){/if}
                    </option>
                {/each}
            </select>
        </div>

        <div class="pb-px">
            <button
                class="btn-secondary px-3 py-2"
                onclick={() => (addDialogOpen = true)}
                title="Create new template"
            >
                <SvgPlus class="h-4 w-4" />
            </button>
        </div>
    </div>

    <Card>
        <div class="relative h-64">
            {#if !selectedTemplate}
                <div class="flex h-full items-center justify-center text-sm text-gray-500">
                    No template selected
                </div>
            {:else if isLoadingContent}
                <SpinnerBlock class="py-8" size="h-4 w-4" label="Loading template…" />
            {:else if contentError}
                <div class="flex h-full items-center justify-center p-4 text-sm text-error">
                    {contentError}
                </div>
            {:else}
                <JinjaEditor
                    class="h-full rounded-md border border-border bg-surface text-sm"
                    editable={false}
                    value={currentContent}
                />
            {/if}
        </div>

        <ActionBar>
            <ActionBarItem
                onclick={() => (editDialogOpen = true)}
                disabled={!selectedTemplate || isLoadingContent}
                icon={SvgEdit}
                variant="primary"
            >
                Edit
            </ActionBarItem>
            <ActionBarItem
                onclick={handleDelete}
                disabled={!canDelete}
                icon={SvgDelete}
                variant="danger"
            >
                Delete
            </ActionBarItem>
        </ActionBar>
    </Card>
</section>

<!-- Edit dialog (existing template). Reuses the same dialog as the /templates route. -->
<EditTemplateDialog
    open={editDialogOpen}
    templateName={selectedTemplate || null}
    onclose={() => (editDialogOpen = false)}
    onsaved={handleEditSaved}
/>

<!-- Add dialog (new template). The dialog's own `onsaved` closes it; we just
     switch into the new template after the list is refreshed. -->
<EditTemplateDialog
    open={addDialogOpen}
    templateName={null}
    onclose={() => (addDialogOpen = false)}
    onsaved={handleAddSaved}
/>
