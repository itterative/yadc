/**
 * Patch builder for the dataset config editor.
 *
 * ``buildPatch()`` converts the current form-field state into the JSON
 * shape the backend's ``PATCH /configs/<name>`` endpoint expects.
 * ``createPreviewScheduler()`` debounces dry-run preview requests.
 *
 * Pure functions, no Svelte dependencies — easy to test in isolation.
 */

import { previewConfig, type Config, type ConfigDatasetEntry } from '$lib/stores/config';
import { configState, type DatasetEntry } from './state.svelte';

/** Build the patch dict from current field values.
 *
 * Nullable fields are omitted when null. Sections whose values are all
 * empty/default are also omitted so the PATCH doesn't add meaningless
 * empty tables to the TOML.
 */
export function buildPatch(): Partial<Config> {
    const {
        apiUrl,
        apiModelName,
        promptName,
        maxTokens,
        imageQuality,
        rounds,
        overwrite,
        storeConversation,
        reasoningEnabled,
        reasoningEffort,
        reasoningExcludeOutput,
        envName,
        datasetEntries
    } = configState;

    const patch: Record<string, unknown> = {};

    // API — only include if at least one field is non-empty
    if (apiUrl || apiModelName) {
        patch.api = { url: apiUrl, model_name: apiModelName };
    }

    // Prompt — only include if a template is selected
    if (promptName) {
        patch.prompt = { name: promptName };
    }

    // Settings — only include if at least one value is non-default
    const settings: Record<string, unknown> = {};
    if (maxTokens !== null) {
        settings.max_tokens = maxTokens;
    }
    if (imageQuality !== null) {
        settings.image_quality = imageQuality;
    }
    if (storeConversation) {
        settings.store_conversation = true;
    }
    if (Object.keys(settings).length > 0) {
        patch.settings = settings;
    }

    // Overwrite — only include if true (default is false)
    if (overwrite) {
        patch.overwrite_captions = true;
    }

    // Reasoning — only include if enabled
    if (reasoningEnabled) {
        patch.reasoning = {
            enable: true,
            thinking_effort: reasoningEffort,
            exclude_from_output: reasoningExcludeOutput
        };
    }

    // Environment — only include if non-empty
    if (envName) {
        patch.env = envName;
    }

    // Rounds — only include when explicitly set
    if (rounds !== null) {
        patch.rounds = rounds;
    }

    // Dataset entries — always included since a config needs at least one
    patch.dataset = datasetEntries.map((entry: DatasetEntry, i: number) => {
        const obj: Record<string, unknown> = { path: entry.path };
        const extras: Record<string, unknown> = {};
        for (const { key, value } of entry.extras) {
            if (key) {
                extras[key] = value;
            }
        }
        if (Object.keys(extras).length > 0) {
            obj.extras = extras;
        }
        // Preserve inline images from the original parsed config (read-only, not edited)
        const raw: ConfigDatasetEntry | undefined = configState.parsedDatasetRaw[i];
        if (raw?.images && raw.images.length > 0) {
            obj.images = raw.images;
        }
        return obj;
    });

    return patch;
}

/** Factory for a debounced preview scheduler.
 *
 * Schedules a dry-run PATCH against the backend and writes the resulting
 * content to ``onResult``. Multiple calls within ``debounceMs`` coalesce
 * into a single request. Failures are swallowed (preview is best-effort).
 */
export function createPreviewScheduler(
    datasetName: () => string,
    onResult: (content: string) => void,
    debounceMs: number = 400
): {
    schedule: () => void;
    cancel: () => void;
} {
    let timer: ReturnType<typeof setTimeout> | null = null;

    function schedule() {
        if (timer !== null) {
            clearTimeout(timer);
        }
        timer = setTimeout(() => {
            timer = null;
            previewConfig(datasetName(), buildPatch())
                .then((result) => {
                    onResult(result.content);
                })
                .catch(() => {
                    /* preview failure is non-critical */
                });
        }, debounceMs);
    }

    function cancel() {
        if (timer !== null) {
            clearTimeout(timer);
            timer = null;
        }
    }

    return { schedule, cancel };
}
