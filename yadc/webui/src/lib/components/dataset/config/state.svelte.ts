/**
 * Shared state for the dataset config editor.
 *
 * Owns the field values, the loaded snapshots (for dirty tracking), and the
 * raw TOML content. The Form, Advanced, and container components all read
 * from and write to this single state instance so the data flows through
 * one source of truth.
 *
 * The "loaded*" mirrors are deep-copied from the live values on load and
 * on save, so mutations to the live fields don't leak into the snapshot
 * (which would defeat dirty tracking — see plan todo 7.1).
 */

import type { KeyValueEntry } from '$lib/components/ui/KeyValueEditor.svelte';
import type { Config, ConfigDatasetEntry } from '$lib/stores/config';

export interface DatasetEntry {
    path: string;
    extras: KeyValueEntry[];
    imageCount: number;
}

/** Detect the TOML type of a parsed extra value. */
function detectType(value: unknown): KeyValueEntry['type'] {
    if (typeof value === 'boolean') {
        return 'boolean';
    }
    if (typeof value === 'number') {
        return 'number';
    }
    if (typeof value === 'object' && value !== null) {
        return 'object';
    }
    return 'string';
}

/** Convert an extras dict to a sorted ``KeyValueEntry[]``. */
function extrasToEntries(extras?: Record<string, unknown>): KeyValueEntry[] {
    if (!extras) {
        return [];
    }
    return Object.entries(extras)
        .sort(([a], [b]) => a.localeCompare(b))
        .map(([key, value]) => ({
            key,
            value: value as KeyValueEntry['value'],
            type: detectType(value)
        }));
}

/** Deep-copy dataset entries for a snapshot (preserves dirty tracking). */
function snapshotEntries(entries: DatasetEntry[]): DatasetEntry[] {
    return entries.map((e) => ({
        ...e,
        extras: e.extras.map((kv) => ({ ...kv, value: kv.value, type: kv.type }))
    }));
}

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
                (e, j) =>
                    e.key === other.extras[j].key &&
                    e.type === other.extras[j].type &&
                    JSON.stringify(e.value) === JSON.stringify(other.extras[j].value)
            )
        );
    });
}

class ConfigState {
    // --- Form fields ---

    apiUrl = $state('');
    apiModelName = $state('');
    promptName = $state('');
    maxTokens: number | null = $state(null);
    imageQuality: 'auto' | 'high' | 'low' | null = $state(null);
    rounds: number | null = $state(null);
    overwrite = $state(false);
    storeConversation = $state(false);
    reasoningEnabled = $state(false);
    reasoningEffort: 'low' | 'medium' | 'high' = $state('low');
    reasoningExcludeOutput = $state(true);
    envName = $state('');
    datasetEntries: DatasetEntry[] = $state([]);

    // --- Loaded snapshots (for dirty tracking) ---

    loadedApiUrl = $state('');
    loadedApiModelName = $state('');
    loadedPromptName = $state('');
    loadedMaxTokens: number | null = $state(null);
    loadedImageQuality: 'auto' | 'high' | 'low' | null = $state(null);
    loadedRounds: number | null = $state(null);
    loadedOverwrite = $state(false);
    loadedStoreConversation = $state(false);
    loadedReasoningEnabled = $state(false);
    loadedReasoningEffort: 'low' | 'medium' | 'high' = $state('low');
    loadedReasoningExcludeOutput = $state(true);
    loadedEnvName = $state('');
    loadedDatasetEntries: DatasetEntry[] = $state([]);

    // --- Raw TOML (Advanced view) ---

    rawContent = $state('');
    loadedRawContent = $state('');

    /**
     * Parsed dataset entries, kept alongside the form-friendly
     * ``datasetEntries``. Used at save time to preserve read-only
     * fields (inline ``images``) that the form doesn't expose.
     */
    parsedDatasetRaw: ConfigDatasetEntry[] = $state([]);

    // --- Dirty derivations ---

    get rawDirty(): boolean {
        return this.rawContent !== this.loadedRawContent;
    }

    get simplifiedDirty(): boolean {
        return (
            this.apiUrl !== this.loadedApiUrl ||
            this.apiModelName !== this.loadedApiModelName ||
            this.promptName !== this.loadedPromptName ||
            this.maxTokens !== this.loadedMaxTokens ||
            this.imageQuality !== this.loadedImageQuality ||
            this.rounds !== this.loadedRounds ||
            this.overwrite !== this.loadedOverwrite ||
            this.storeConversation !== this.loadedStoreConversation ||
            this.reasoningEnabled !== this.loadedReasoningEnabled ||
            this.reasoningEffort !== this.loadedReasoningEffort ||
            this.reasoningExcludeOutput !== this.loadedReasoningExcludeOutput ||
            this.envName !== this.loadedEnvName ||
            !entriesEqual(this.datasetEntries, this.loadedDatasetEntries)
        );
    }

    // --- Mutations used by the form ---

    addDatasetEntry(): void {
        this.datasetEntries = [...this.datasetEntries, { path: '', extras: [], imageCount: 0 }];
    }

    removeDatasetEntry(index: number): void {
        this.datasetEntries = this.datasetEntries.filter((_, i) => i !== index);
    }

    updateEntryPath(index: number, newPath: string): void {
        this.datasetEntries = this.datasetEntries.map((e, i) =>
            i === index ? { ...e, path: newPath } : e
        );
    }

    // --- Populate / reset (called from loadConfig and after save) ---

    /** Populate all fields from a parsed config and raw TOML content. */
    populateFields(p: Config, content: string): void {
        this.rawContent = content;
        this.loadedRawContent = content;

        this.apiUrl = this.loadedApiUrl = p.api?.url ?? '';
        this.apiModelName = this.loadedApiModelName = p.api?.model_name ?? '';
        this.promptName = this.loadedPromptName = p.prompt?.name ?? '';
        this.maxTokens = this.loadedMaxTokens = p.settings?.max_tokens ?? null;
        this.imageQuality = this.loadedImageQuality = p.settings?.image_quality ?? null;
        this.rounds = this.loadedRounds = p.rounds ?? null;
        this.overwrite = this.loadedOverwrite = p.overwrite_captions ?? false;
        this.storeConversation = this.loadedStoreConversation =
            p.settings?.store_conversation ?? false;
        this.reasoningEnabled = this.loadedReasoningEnabled = p.reasoning?.enable ?? false;
        this.reasoningEffort = this.loadedReasoningEffort = p.reasoning?.thinking_effort ?? 'low';
        this.reasoningExcludeOutput = this.loadedReasoningExcludeOutput =
            p.reasoning?.exclude_from_output ?? true;
        this.envName = this.loadedEnvName = p.env ?? '';

        const rawEntries = p.dataset ?? [];
        this.parsedDatasetRaw = rawEntries;
        const entries = rawEntries.map((entry) => ({
            path: entry.path ?? '',
            extras: extrasToEntries(entry.extras as Record<string, unknown> | undefined),
            imageCount: entry.images?.length ?? 0
        }));
        this.datasetEntries = entries;
        this.loadedDatasetEntries = snapshotEntries(entries);
    }

    /** Update only the simplified-view loaded snapshots (after a successful PATCH). */
    resetSimplifiedDirty(): void {
        this.loadedApiUrl = this.apiUrl;
        this.loadedApiModelName = this.apiModelName;
        this.loadedPromptName = this.promptName;
        this.loadedMaxTokens = this.maxTokens;
        this.loadedImageQuality = this.imageQuality;
        this.loadedRounds = this.rounds;
        this.loadedOverwrite = this.overwrite;
        this.loadedStoreConversation = this.storeConversation;
        this.loadedReasoningEnabled = this.reasoningEnabled;
        this.loadedReasoningEffort = this.reasoningEffort;
        this.loadedReasoningExcludeOutput = this.reasoningExcludeOutput;
        this.loadedEnvName = this.envName;
        this.loadedDatasetEntries = snapshotEntries(this.datasetEntries);
    }
}

/** Singleton shared by all DatasetConfig views. */
export const configState = new ConfigState();
