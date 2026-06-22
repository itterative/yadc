<script lang="ts">
    import SvgPhoto from '$lib/icons/SvgPhoto.svelte';
    import SvgUpload from '$lib/icons/SvgUpload.svelte';
    import SvgPlus from '$lib/icons/SvgPlus.svelte';
    import SvgEdit from '$lib/icons/SvgEdit.svelte';
    import SvgDelete from '$lib/icons/SvgDelete.svelte';
    import { readFileAsDataUrl } from '$lib/stores/prompts';
    import type { ExamplePair } from '$lib/stores/prompts';
    import { friendlyErrorMessage } from '$lib/api';
    import Card from '$lib/components/ui/Card.svelte';
    import ActionBar from '$lib/components/ui/ActionBar.svelte';
    import ActionBarItem from '$lib/components/ui/ActionBarItem.svelte';
    import EditExampleDialog from './EditExampleDialog.svelte';
    import DatasetPickerDialog from './DatasetPickerDialog.svelte';

    interface Props {
        examples: ExamplePair[];
        disabled?: boolean;
    }

    let { examples = $bindable([]), disabled = false }: Props = $props();

    let fileInput: HTMLInputElement | null = $state(null);
    let fileError: string | null = $state(null);

    // EditExampleDialog state. ``editIndex`` is null when composing a new
    // example (manual add) and an index when editing an existing one —
    // the dialog itself is identical either way; the index only decides
    // append vs update on save.
    let editDialogOpen = $state(false);
    let editIndex: number | null = $state(null);
    let editImage = $state('');
    let editSubject = $state('');
    let editCaption = $state('');

    let pickerOpen = $state(false);

    function openEditExisting(i: number) {
        const ex = examples[i];
        editIndex = i;
        editImage = ex.image_data_url;
        editSubject = ex.subject;
        editCaption = ex.caption;
        editDialogOpen = true;
    }

    function handleEditSave(patch: { subject: string; caption: string }) {
        if (editIndex === null) {
            examples = [...examples, { ...patch, image_data_url: editImage }];
        } else {
            examples = examples.map((ex, i) => (i === editIndex ? { ...ex, ...patch } : ex));
        }
        editDialogOpen = false;
    }

    function deleteExample(i: number) {
        examples = examples.filter((_, idx) => idx !== i);
    }

    function pickFiles() {
        fileError = null;
        fileInput?.click();
    }

    async function handleFiles(event: Event) {
        const input = event.target as HTMLInputElement;
        const file = input.files?.[0];
        // Reset so the same file can be re-selected.
        input.value = '';
        if (!file) {
            return;
        }
        try {
            const { dataUrl, mime } = await readFileAsDataUrl(file);
            if (!mime.startsWith('image/')) {
                fileError = `${file.name}: not an image file`;
                return;
            }
            // Seed the edit dialog with the filename stem as the subject;
            // the user adjusts subject + caption before the example is
            // committed. ``editIndex = null`` → append on save.
            editIndex = null;
            editImage = dataUrl;
            editSubject = file.name.replace(/\.[^.]+$/, '');
            editCaption = '';
            editDialogOpen = true;
        } catch (e) {
            fileError = friendlyErrorMessage(e, `Failed to read ${file.name}`);
        }
    }

    function handleDatasetAdd(newExamples: ExamplePair[]) {
        examples = [...examples, ...newExamples];
        pickerOpen = false;
    }
</script>

<input bind:this={fileInput} type="file" accept="image/*" class="hidden" onchange={handleFiles} />

<div class="space-y-2">
    {#if examples.length === 0}
        <div
            class="flex flex-col items-center gap-2 rounded-lg border border-dashed border-gray-600 bg-bg/30 px-4 py-6 text-center"
        >
            <SvgPhoto class="h-6 w-6 text-muted" />
            <p class="text-sm text-gray-400">
                No examples yet — add a few to improve style transfer.
            </p>
        </div>
    {:else}
        <ul class="space-y-2">
            {#each examples as ex, i (i)}
                <li data-testid="example-row">
                    <Card class="group relative">
                        <div class="flex gap-3 p-2">
                            <div
                                class="flex h-16 w-16 shrink-0 items-center justify-center overflow-hidden rounded-md bg-gray-900"
                            >
                                <img
                                    src={ex.image_data_url}
                                    alt={ex.subject}
                                    class="h-full w-full object-cover"
                                />
                            </div>
                            <div class="flex min-w-0 flex-1 flex-col justify-center gap-0.5 pr-10">
                                <p class="truncate text-sm font-medium text-gray-200">
                                    {ex.subject || '(no subject)'}
                                </p>
                                <p class="line-clamp-2 text-xs text-gray-400">
                                    {ex.caption || '(no caption)'}
                                </p>
                            </div>
                        </div>
                        <!-- Corner actions (templates-page pattern):
                             hover-revealed on desktop, always visible on
                             touch via max-lg:opacity-100. The bg-black/60
                             chips stay legible over any thumbnail. -->
                        <div
                            class="absolute top-1.5 right-1.5 flex gap-1 opacity-0 transition-opacity group-hover:opacity-100 max-lg:opacity-100"
                        >
                            <button
                                class="cursor-pointer rounded-md bg-black/60 p-1.5 text-gray-300 transition-colors hover:bg-black/80 hover:text-white disabled:cursor-not-allowed disabled:opacity-40"
                                title="Edit example"
                                aria-label="Edit example"
                                onclick={() => openEditExisting(i)}
                                {disabled}
                            >
                                <SvgEdit class="h-4 w-4" />
                            </button>
                            <button
                                class="cursor-pointer rounded-md bg-black/60 p-1.5 text-gray-300 transition-colors hover:bg-black/80 hover:text-error disabled:cursor-not-allowed disabled:opacity-40"
                                title="Delete example"
                                aria-label="Delete example"
                                onclick={() => deleteExample(i)}
                                {disabled}
                            >
                                <SvgDelete class="h-4 w-4" />
                            </button>
                        </div>
                    </Card>
                </li>
            {/each}
        </ul>
    {/if}

    {#if fileError}
        <p class="text-xs text-error">{fileError}</p>
    {/if}

    <div class="overflow-hidden rounded-lg">
        <ActionBar>
            <ActionBarItem onclick={pickFiles} {disabled} icon={SvgUpload} variant="secondary"
                >Add manual</ActionBarItem
            >
            <ActionBarItem
                onclick={() => (pickerOpen = true)}
                {disabled}
                icon={SvgPlus}
                variant="secondary">From dataset</ActionBarItem
            >
        </ActionBar>
    </div>
</div>

<EditExampleDialog
    open={editDialogOpen}
    title={editIndex === null ? 'Add example' : 'Edit example'}
    imageDataUrl={editImage}
    initialSubject={editSubject}
    initialCaption={editCaption}
    onsave={handleEditSave}
    onclose={() => (editDialogOpen = false)}
/>

<DatasetPickerDialog
    open={pickerOpen}
    onadd={handleDatasetAdd}
    onclose={() => (pickerOpen = false)}
/>
