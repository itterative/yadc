import { writable, get, type Writable } from 'svelte/store';
import {
    cancelTagging as apiCancelTagging,
    saveImageTags as apiSaveImageTags,
    startTagJob as apiStartTagJob,
    swapTaggerModel as apiSwapTaggerModel,
    tagImage as apiTagImage
} from './api';
import { addCurrentlyTagging, removeCurrentlyTagging } from './inflight';
import { resetTaggingStatus, setTaggingStatus, taggingStatuses } from './status';
import { tagSettings } from './settings';
import { registerJobId } from '../caption/jobs';
import { toast } from '../toasts';
import { friendlyErrorMessage } from '$lib/api';
import type {
    CancelResult,
    SwapTaggerBody,
    TagJobInfo,
    TaggerResult,
    TagSaveOptions
} from './types';

// --- Assembled options store ---

/** Threshold + save options assembled by the batch panel and read by the
 *  action functions. ``null`` thresholds mean "use server default" (omitted
 *  from the request). Written by TagSettingsPanel. */
export interface TagOptions {
    rating_threshold?: number;
    general_threshold?: number;
    character_threshold?: number;
    replace_underscores?: boolean;
    save?: TagSaveOptions;
    source?: string;
}

export const tagOptions: Writable<TagOptions> = writable({});

/** Job ID of the last tagging job started by this tab. Used by
 *  +page.svelte to fire completion toasts only for jobs this tab
 *  started, not for stale jobs on page load. */
export const lastStartedTagJobId: Writable<string> = writable('');

// --- Actions ---

/** Tag a single image synchronously and return the result. Used by the
 *  interactive Tags tab — the returned value populates the prune grid
 *  immediately. The result is also cached server-side (see
 *  ``TaggingService._tag_results``) so a fresh tab / page load sees it
 *  via :func:`fetchCachedTagResult`. Tracks the image as in-flight while
 *  the request is pending. Throws on failure (caller surfaces the toast). */
export async function tagSingleImage(
    datasetName: string,
    imageId: number,
    source?: string
): Promise<TaggerResult> {
    const options = get(tagOptions);
    addCurrentlyTagging(datasetName, imageId);
    try {
        return await apiTagImage(datasetName, imageId, {
            rating_threshold: options.rating_threshold,
            general_threshold: options.general_threshold,
            character_threshold: options.character_threshold,
            replace_underscores: options.replace_underscores,
            source
        });
    } finally {
        removeCurrentlyTagging(datasetName, imageId);
    }
}

/** Start a batch tagging job. Reads current options from the store, seeds
 *  the SSE status store for immediate UI feedback. Returns the job ID. */
export async function startBatchTagging(datasetName: string): Promise<string> {
    setTaggingStatus({
        status: 'running',
        dataset_name: datasetName,
        job_id: '',
        processed: 0,
        total: 0,
        errors: 0,
        error: null,
        error_messages: [],
        source: '',
        elapsed: 0
    });
    const options = get(tagOptions);
    try {
        const info: TagJobInfo = await apiStartTagJob(datasetName, {
            rating_threshold: options.rating_threshold,
            general_threshold: options.general_threshold,
            character_threshold: options.character_threshold,
            replace_underscores: options.replace_underscores,
            save: options.save,
            source: options.source
        });
        // Seed real status immediately so the progress bar appears without
        // waiting for the first SSE event.
        setTaggingStatus(info);
        // The no-op case (overwrite=False + every image already has saved
        // tags) returns ``done`` synchronously with no background task, so
        // no progress bar ever appears. Surface it as a toast instead.
        if (info.status === 'done' && info.total > 0) {
            toast.info(`All ${info.total} images already have tags — nothing to do`);
        }
        // Register the job_id so dataset_changed events caused by our own
        // tagging writes are suppressed on this tab (the backend tags the
        // watcher events with the job_id via expect_changes).
        registerJobId(info.job_id);
        // Track for the page-level completion-toast effect (mirrors
        // captioning's ``lastStartedJobId``). The +page.svelte effect
        // matches this id against incoming ``tag_job_status`` events and
        // fires a toast when the job reaches a terminal state.
        lastStartedTagJobId.set(info.job_id);
        return info.job_id;
    } catch (e) {
        resetTaggingStatus(datasetName);
        throw e;
    }
}

/** Stop a running tagging job (graceful-first, kill-as-fallback). Reads the
 *  current job_id from the status store so the cancel targets the right job
 *  and refuses if it's stale. Toasts the outcome. */
export async function stopTagging(
    datasetName: string,
    onError?: (message: string) => void
): Promise<boolean> {
    const status = get(taggingStatuses).get(datasetName);
    const jobId = status?.job_id;
    try {
        const result = await apiCancelTagging(jobId);
        toastCancelResult(result);
        return result.outcome !== 'stale_job';
    } catch {
        toast.error('Failed to stop tagging: request failed');
        onError?.('request failed');
        return false;
    }
}

/** Cancel an in-flight single-image tag (the interactive Tags tab Cancel
 *  button). No job to stop — cancel escalates to killing the subprocess if
 *  the request is mid-inference. Removes the image from the in-flight set
 *  immediately so the button flips back without waiting on the network. */
export async function cancelTaggingAction(datasetName: string, imageId: number): Promise<void> {
    removeCurrentlyTagging(datasetName, imageId);
    try {
        const result = await apiCancelTagging();
        toastCancelResult(result);
    } catch {
        toast.error('Failed to cancel tagging: request failed');
    }
}

function toastCancelResult(result: CancelResult): void {
    switch (result.outcome) {
        case 'killed':
            toast.info('Tagger process killed — it will respawn on the next request');
            break;
        case 'stopped':
            toast.info('Tagging stopped');
            break;
        case 'stale_job':
            toast.warning('That job is no longer running');
            break;
        case 'nothing_running':
            // Nothing to cancel — stay quiet.
            break;
    }
}

/** Write user-pruned tags for an image to a draft / extras without re-running
 *  the model. Used by the interactive Tags tab's save bar. The save options
 *  default to the persisted settings when not provided. */
export async function saveImageTagsAction(
    datasetName: string,
    imageId: number,
    result: TaggerResult,
    save?: TagSaveOptions
): Promise<void> {
    const settings = get(tagSettings);
    const resolvedSave: TagSaveOptions = save ?? {
        mode: settings.saveMode,
        draft_name: settings.draftName || 'tags',
        draft_format: settings.draftFormat || 'comma',
        overwrite: true
    };
    try {
        await apiSaveImageTags(datasetName, imageId, result, resolvedSave);
    } catch (e) {
        throw new Error(friendlyErrorMessage(e, 'Failed to save tags'));
    }
}

/** Swap the active tagger selection. Dispatches the right toast for each
 *  outcome (success / batch running / concurrent swap / error) and returns
 *  the new active payload on success, or ``null`` when the server refused
 *  the swap. The caller decides what to do with ``null`` (typically: leave
 *  the picker showing the prior value). */
export async function swapActiveModelAction(body: SwapTaggerBody): Promise<SwapTaggerBody | null> {
    const result = await apiSwapTaggerModel(body);
    switch (result.status) {
        case 'ok':
            toast.info(`Tagger swapped to ${result.response.active?.source ?? body.repo_id}`);
            return result.response.active
                ? {
                      kind: result.response.active.kind,
                      repo_id: result.response.active.repo_id,
                      repo_model_filename: result.response.active.repo_model_filename,
                      repo_label_filename: result.response.active.repo_label_filename,
                      model_path: result.response.active.model_path,
                      label_path: result.response.active.label_path,
                      preproc_profile: result.response.active.preproc_profile,
                      default_size: result.response.active.default_size
                  }
                : null;
        case 'busy':
            toast.warning('A batch tagging job is running — stop it before swapping models', {
                details: [result.message]
            });
            return null;
        case 'in_progress':
            toast.warning('Another swap is in progress', {
                details: [
                    result.message,
                    `Retry in ~${Math.ceil(result.retryAfterS)}s — the previous swap is still draining.`
                ]
            });
            return null;
        case 'error':
            toast.error('Failed to swap tagger model', { details: [result.message] });
            return null;
    }
}
