<script lang="ts">
    import { onMount } from 'svelte';
    import { browser } from '$app/environment';
    import {
        templates,
        refreshTemplates,
        deleteTemplate,
        type TemplateListItem
    } from '$lib/stores/templates';
    import EditTemplateDialog from './EditTemplateDialog.svelte';
    import ConfirmDelete from '$lib/components/ui/ConfirmDelete.svelte';
    import SvgDelete from '$lib/icons/SvgDelete.svelte';
    import SvgEdit from '$lib/icons/SvgEdit.svelte';
    import SvgFile from '$lib/icons/SvgFile.svelte';
    import SvgPlus from '$lib/icons/SvgPlus.svelte';
    import { friendlyErrorMessage } from '$lib/api';

    let loading = $state(true);
    let error = $state('');
    let showAddTemplate = $state(false);

    let deletingTemplate: TemplateListItem | null = $state(null);

    // Edit state
    let editingTemplate: TemplateListItem | null = $state(null);

    onMount(() => {
        if (!browser) {
            return;
        }
        loadTemplates();
    });

    async function loadTemplates() {
        try {
            await refreshTemplates();
        } catch (e) {
            error = friendlyErrorMessage(e, 'Failed to load templates');
        } finally {
            loading = false;
        }
    }

    async function handleDeleteConfirm() {
        if (!deletingTemplate) {
            return;
        }
        try {
            await deleteTemplate(deletingTemplate.name);
            await refreshTemplates();
        } catch (e) {
            error = friendlyErrorMessage(e, 'Failed to delete template');
        } finally {
            deletingTemplate = null;
        }
    }

    function handleTemplateSaved() {
        loadTemplates();
        editingTemplate = null;
    }

    function handleTemplateCreated(name: string) {
        loadTemplates();
        showAddTemplate = false;
        // Open the new template for editing
        editingTemplate = { name, source: 'user' };
    }
</script>

{#if loading}
    <p class="text-muted">Loading templates...</p>
{:else if error}
    <p class="text-error">Error: {error}</p>
{:else if $templates.items.length === 0}
    <div class="empty-state py-12">
        <h2>No templates found</h2>
        <p>Create a new Jinja2 template to get started.</p>
        <button class="btn-primary mt-4" onclick={() => (showAddTemplate = true)}>
            Add Template
        </button>
    </div>
{:else}
    <div class="grid grid-cols-[repeat(auto-fill,minmax(260px,1fr))] gap-4">
        {#each $templates.items as template (template.name)}
            <div
                class="card group relative cursor-pointer transition-opacity hover:opacity-90"
                onclick={() => (editingTemplate = template)}
                onkeydown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                        editingTemplate = template;
                    }
                }}
                role="button"
                tabindex="0"
            >
                <div class="flex aspect-video w-full items-center justify-center bg-border/30">
                    <SvgFile class="h-8 w-8 text-muted opacity-50" />
                </div>
                <div class="card-body">
                    <h3 class="mb-1 text-fg">{template.name}</h3>
                    <span
                        class="badge {template.source === 'builtin'
                            ? 'badge-muted'
                            : 'badge-accent'}"
                    >
                        {template.source === 'builtin' ? 'Built-in' : 'User'}
                    </span>
                </div>

                <!-- Action buttons (top-right corner) -->
                <div
                    class="absolute top-2 right-2 flex gap-1 opacity-0 transition-opacity group-hover:opacity-100 max-lg:opacity-100"
                >
                    <button
                        class="cursor-pointer rounded-md bg-black/60 p-1.5 text-gray-300 hover:bg-black/80 hover:text-white"
                        title="Edit template"
                        onclick={(e) => {
                            e.preventDefault();
                            e.stopPropagation();
                            editingTemplate = template;
                        }}
                    >
                        <SvgEdit class="h-4 w-4" />
                    </button>
                    {#if template.source === 'user'}
                        <button
                            class="cursor-pointer rounded-md bg-black/60 p-1.5 text-gray-300 hover:bg-black/80 hover:text-error"
                            title="Delete template"
                            onclick={(e) => {
                                e.preventDefault();
                                e.stopPropagation();
                                deletingTemplate = template;
                            }}
                        >
                            <SvgDelete class="h-4 w-4" />
                        </button>
                    {/if}
                </div>
            </div>
        {/each}
        <button
            class="card flex min-h-56 cursor-pointer items-center justify-center border-dashed border-gray-600 hover:border-accent"
            onclick={() => (showAddTemplate = true)}
            title="Add template"
        >
            <span class="flex items-center gap-2 text-sm text-gray-400">
                <SvgPlus class="h-4 w-4" />
                Add Template
            </span>
        </button>
    </div>

    <!-- Delete confirmation -->
    {#if deletingTemplate}
        <ConfirmDelete
            open={true}
            oncancel={() => (deletingTemplate = null)}
            onconfirm={handleDeleteConfirm}
        >
            Delete template "{deletingTemplate.name}"? This action cannot be undone.
        </ConfirmDelete>
    {/if}
{/if}

<!-- Edit dialog -->
{#if editingTemplate}
    <EditTemplateDialog
        open={true}
        templateName={editingTemplate.name}
        onclose={() => (editingTemplate = null)}
        onsaved={handleTemplateSaved}
    />
{/if}

<!-- Add dialog -->
<EditTemplateDialog
    open={showAddTemplate}
    templateName={null}
    onclose={() => (showAddTemplate = false)}
    onsaved={(name) => handleTemplateCreated(name)}
/>
