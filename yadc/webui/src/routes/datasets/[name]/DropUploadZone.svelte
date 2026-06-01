<script lang="ts">
    import type { Snippet } from 'svelte';
    import SvgUpload from '$lib/icons/SvgUpload.svelte';
    import SvgWarning from '$lib/icons/SvgWarning.svelte';

    interface Props {
        /** Whether uploads are currently allowed. Drives the overlay's
         *  visual state — accent when true, warning when false. */
        canUpload: boolean;
        /** Reason to show in the warning overlay. Ignored when canUpload is true. */
        blockedReason: string | null;
        /** Dataset name — shown in the success message and used in the dialog. */
        datasetName: string;
        /** Called with the dropped files when a valid drop happens. */
        ondrop: (files: File[]) => void;
        /** Called when a drop is attempted but blocked. Parent typically
         *  surfaces this as a toast. */
        onblockeddrop?: (reason: string) => void;
        children: Snippet;
    }

    let { canUpload, blockedReason, datasetName, ondrop, onblockeddrop, children }: Props =
        $props();

    // True while a file drag is over the drop zone. Filtered by
    // `dataTransfer.types.includes('Files')` so text/links don't trigger it.
    let isDraggingFiles = $state(false);
    // Counter pattern (matches FileDropZone) — handles dragenter/dragleave
    // firing on child elements as the cursor moves across the zone.
    let dragCounter = 0;

    // Visible overlay: 'normal' for allowed drops, 'warning' for blocked drops,
    // null when no drag is in progress or the drag isn't a file drag.
    let dragOverlayState: 'normal' | 'warning' | null = $derived(
        isDraggingFiles ? (canUpload ? 'normal' : 'warning') : null
    );

    function handleDragEnter(event: DragEvent) {
        // Always preventDefault so the browser doesn't treat the zone as a
        // foreign drop target (e.g. navigating to a dropped URL).
        event.preventDefault();
        if (!event.dataTransfer?.types.includes('Files')) {
            return;
        }
        dragCounter++;
        isDraggingFiles = true;
    }

    function handleDragOver(event: DragEvent) {
        event.preventDefault();
        if (!isDraggingFiles) {
            return;
        }
        // Visual cursor feedback: 'copy' when the drop will work, 'none' when blocked.
        if (event.dataTransfer) {
            event.dataTransfer.dropEffect = canUpload ? 'copy' : 'none';
        }
    }

    function handleDragLeave(event: DragEvent) {
        event.preventDefault();
        if (!isDraggingFiles) {
            return;
        }
        dragCounter--;
        if (dragCounter <= 0) {
            dragCounter = 0;
            isDraggingFiles = false;
        }
    }

    async function handleDrop(event: DragEvent) {
        event.preventDefault();
        dragCounter = 0;
        if (!isDraggingFiles) {
            return;
        }
        isDraggingFiles = false;

        if (!canUpload) {
            // Surface the reason in case the user missed the overlay
            // (e.g. drag-cancelled mid-flight).
            if (blockedReason) {
                onblockeddrop?.(blockedReason);
            }
            return;
        }

        const items = event.dataTransfer?.items;
        if (!items) {
            return;
        }
        const files = await collectDroppedFiles(items);
        if (files.length === 0) {
            return;
        }

        ondrop(files);
    }

    // Recursive folder/file collection from a DataTransferItemList. Mirrors
    // FileDropZone's logic — kept here so the zone is self-contained.
    async function collectDroppedFiles(items: DataTransferItemList): Promise<File[]> {
        const files: File[] = [];
        for (let i = 0; i < items.length; i++) {
            const entry = items[i].webkitGetAsEntry();
            if (entry) {
                files.push(...(await collectFilesFromEntry(entry)));
            }
        }
        return files;
    }

    async function collectFilesFromEntry(entry: FileSystemEntry, path = ''): Promise<File[]> {
        const fullPath = path ? `${path}/${entry.name}` : entry.name;

        if (entry.isFile) {
            const file = await new Promise<File>((resolve, reject) => {
                (entry as FileSystemFileEntry).file(resolve, reject);
            });
            Object.defineProperty(file, 'webkitRelativePath', {
                value: fullPath,
                writable: false
            });
            return [file];
        }

        if (entry.isDirectory) {
            const dirEntry = entry as FileSystemDirectoryEntry;
            const reader = dirEntry.createReader();
            const collected: File[] = [];

            let entries: FileSystemEntry[];
            do {
                entries = await new Promise<FileSystemEntry[]>((resolve, reject) => {
                    reader.readEntries(resolve, reject);
                });
                for (const child of entries) {
                    if (child.isFile) {
                        const childPath = `${fullPath}/${child.name}`;
                        const file = await new Promise<File>((resolve, reject) => {
                            (child as FileSystemFileEntry).file(resolve, reject);
                        });
                        Object.defineProperty(file, 'webkitRelativePath', {
                            value: childPath,
                            writable: false
                        });
                        collected.push(file);
                    }
                }
            } while (entries.length > 0);

            return collected;
        }

        return [];
    }
</script>

<div
    class="relative min-w-0 flex-1 overflow-y-auto"
    ondragenter={handleDragEnter}
    ondragover={handleDragOver}
    ondragleave={handleDragLeave}
    ondrop={handleDrop}
    role="presentation"
>
    {@render children()}

    <!-- Drop-to-upload overlay. Three visual states:
         - 'normal': accent, drop opens the upload dialog
         - 'warning': warning color, explains why uploads are blocked
         - null: no drag in progress
         The overlay is pointer-events-none so it doesn't interfere with the
         underlying grid (e.g. clicking an image tile). The border outlines
         the entire drop zone. -->
    {#if dragOverlayState}
        <div
            class="pointer-events-none absolute inset-0 z-20 flex items-center justify-center rounded-2xl border-4 border-dashed backdrop-blur-xs transition-opacity
                {dragOverlayState === 'normal'
                ? 'border-accent bg-accent/5'
                : 'border-warning bg-warning/5'}"
        >
            <div class="rounded-xl bg-bg/80 px-8 py-4 text-center shadow-lg">
                {#if dragOverlayState === 'normal'}
                    <SvgUpload class="mx-auto h-12 w-12 text-accent" />
                    <p class="mt-3 text-lg font-semibold text-white">
                        Drop files to add to this dataset
                    </p>
                    <p class="mt-1 text-sm text-gray-400">
                        Files will be uploaded and appended to “{datasetName}”
                    </p>
                {:else}
                    <SvgWarning class="mx-auto h-12 w-12 text-warning" />
                    <p class="mt-3 text-lg font-semibold text-white">Cannot upload files</p>
                    <p class="mt-1 text-sm text-gray-400">{blockedReason}</p>
                {/if}
            </div>
        </div>
    {/if}
</div>
