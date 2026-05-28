<script lang="ts">
    const DEFAULT_UPLOAD_EXTENSIONS = [
        '.jpg',
        '.jpeg',
        '.png',
        '.gif',
        '.bmp',
        '.webp',
        '.tiff',
        '.tif',
        '.ico',
        '.txt',
        '.toml'
    ];

    import Alert from '$lib/components/ui/Alert.svelte';
    import type { Snippet } from 'svelte';
    import { formatBytes } from '$lib/format';

    interface Props {
        class?: string;
        disabled?: boolean;
        accept?: string;
        allowedExtensions?: string[];
        files?: File[];
        children?: Snippet<[]>;
    }

    let {
        class: className = '',
        disabled = false,
        accept = 'image/*',
        allowedExtensions = DEFAULT_UPLOAD_EXTENSIONS,
        files = $bindable([]),
        children
    }: Props = $props();

    let isDragOver = $state(false);
    let dragCounter = 0;
    let fileInput: HTMLInputElement | null = $state(null);
    let folderInput: HTMLInputElement | null = $state(null);
    let validationWarning: string | null = $state(null);

    function getExtension(filename: string): string {
        const lastDot = filename.lastIndexOf('.');
        return lastDot >= 0 ? filename.slice(lastDot).toLowerCase() : '';
    }

    function isAllowedFile(file: File): boolean {
        const ext = getExtension(file.name);
        return allowedExtensions.length === 0 || allowedExtensions.includes(ext);
    }

    function addFiles(newFiles: File[]) {
        validationWarning = null;

        const accepted: File[] = [];
        const rejected: string[] = [];

        for (const file of newFiles) {
            if (isAllowedFile(file)) {
                accepted.push(file);
            } else {
                rejected.push(file.name);
            }
        }

        if (rejected.length > 0) {
            const names = rejected.slice(0, 3).join(', ');
            const more = rejected.length > 3 ? ` and ${rejected.length - 3} more` : '';
            validationWarning = `Skipped unsupported files: ${names}${more}`;
        }

        if (accepted.length > 0) {
            files = [...files, ...accepted];
        }
    }

    function handleFileSelect(event: Event) {
        const input = event.target as HTMLInputElement;
        if (input.files) {
            const allFiles = Array.from(input.files);
            // webkitdirectory may include nested files; we only support one level
            const flatFiles = allFiles.filter((f) => {
                const rel = f.webkitRelativePath || f.name;
                return (rel.match(/\//g) || []).length <= 1;
            });
            addFiles(flatFiles);
        }
        input.value = '';
    }

    async function readDirectoryEntries(
        reader: FileSystemDirectoryReader
    ): Promise<FileSystemEntry[]> {
        return new Promise((resolve, reject) => {
            reader.readEntries((entries) => {
                resolve(entries);
            }, reject);
        });
    }

    async function getFileFromEntry(entry: FileSystemFileEntry): Promise<File> {
        return new Promise((resolve, reject) => {
            entry.file(resolve, reject);
        });
    }

    async function collectFilesFromEntry(entry: FileSystemEntry, path = ''): Promise<File[]> {
        const fullPath = path ? `${path}/${entry.name}` : entry.name;

        if (entry.isFile) {
            const file = await getFileFromEntry(entry as FileSystemFileEntry);
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
                entries = await readDirectoryEntries(reader);
                for (const child of entries) {
                    // Only collect direct file children, skip subdirectories
                    if (child.isFile) {
                        const childPath = `${fullPath}/${child.name}`;
                        const file = await getFileFromEntry(child as FileSystemFileEntry);
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

    async function handleDrop(event: DragEvent) {
        event.preventDefault();
        dragCounter = 0;
        isDragOver = false;

        const items = event.dataTransfer?.items;
        if (!items) {
            return;
        }

        const newFiles: File[] = [];
        for (let i = 0; i < items.length; i++) {
            const entry = items[i].webkitGetAsEntry();
            if (entry) {
                newFiles.push(...(await collectFilesFromEntry(entry)));
            }
        }

        if (newFiles.length > 0) {
            addFiles(newFiles);
        }
    }

    function handleDragEnter(event: DragEvent) {
        event.preventDefault();
        dragCounter++;
        isDragOver = true;
    }

    function handleDragOver(event: DragEvent) {
        event.preventDefault();
    }

    function handleDragLeave(event: DragEvent) {
        event.preventDefault();
        dragCounter--;
        if (dragCounter <= 0) {
            dragCounter = 0;
            isDragOver = false;
        }
    }

    function removeFile(index: number) {
        files = files.filter((_, i) => i !== index);
    }

    function clearFiles() {
        files = [];
        validationWarning = null;
    }
</script>

<div class={className}>
    <div
        class="rounded-lg border-2 border-dashed p-6 text-center transition-colors {isDragOver
            ? 'border-accent bg-accent/5'
            : 'border-border bg-surface'}"
        ondragenter={handleDragEnter}
        ondragover={handleDragOver}
        ondragleave={handleDragLeave}
        ondrop={handleDrop}
        role="region"
        aria-label="Drop files here"
    >
        <div class="space-y-3">
            <p class="text-sm text-gray-400">
                {#if isDragOver}
                    <span class="text-accent">Drop files here</span>
                {:else}
                    Drag and drop files or folders here
                {/if}
            </p>
            <div class="flex justify-center gap-2">
                <button
                    class="btn-secondary"
                    onclick={() => fileInput?.click()}
                    {disabled}
                    type="button"
                >
                    Choose Files
                </button>
                <button
                    class="btn-secondary"
                    onclick={() => folderInput?.click()}
                    {disabled}
                    type="button"
                >
                    Choose Folder
                </button>
            </div>
        </div>
        <input
            bind:this={fileInput}
            type="file"
            multiple
            {accept}
            class="hidden"
            onchange={handleFileSelect}
        />
        <input
            bind:this={folderInput}
            type="file"
            webkitdirectory
            class="hidden"
            onchange={handleFileSelect}
        />
        {#if children}
            <div class="mt-2">{@render children()}</div>
        {:else}
            <p class="help-text mt-2">Select individual images or an entire folder.</p>
        {/if}
    </div>

    {#if validationWarning}
        <Alert
            variant="warning"
            class="mt-3"
            dismissable
            ondismiss={() => (validationWarning = null)}
        >
            {validationWarning}
        </Alert>
    {/if}

    {#if files.length > 0}
        <div class="mt-3 max-h-40 overflow-y-auto rounded-lg border border-border bg-bg">
            {#each files as file, i (file.name + '-' + i)}
                <div class="flex items-center justify-between px-3 py-2 text-sm">
                    <span
                        class="truncate text-gray-200"
                        title={file.webkitRelativePath || file.name}
                    >
                        {file.webkitRelativePath || file.name}
                    </span>
                    <div class="ml-2 flex shrink-0 items-center gap-3">
                        <span class="text-gray-400">{formatBytes(file.size)}</span>
                        <button
                            class="text-gray-500 hover:text-red-400"
                            onclick={() => removeFile(i)}
                            type="button"
                            aria-label="Remove file"
                        >
                            ×
                        </button>
                    </div>
                </div>
            {/each}
        </div>
        <div class="mt-2 flex items-center justify-between text-sm text-gray-400">
            <span>
                {files.length} file{files.length === 1 ? '' : 's'},
                {formatBytes(files.reduce((sum, f) => sum + f.size, 0))} total
            </span>
            <button class="text-red-400 hover:text-red-300" onclick={clearFiles} type="button">
                Clear all
            </button>
        </div>
    {/if}
</div>
