import { writable, get, type Writable } from 'svelte/store';
import {
    startCaptioning,
    captionSingleImage as apiCaptionSingleImage,
    stopCaptioning as apiStopCaptioning
} from '../dataset/api';
import { registerJobId, setCaptioningStatus, setCurrentlyCaptioning } from '../events';
import { withPasswordRetry } from '../passwordPrompt';
import { toast } from '../toasts';
import type { CaptionOptions } from './options';

// --- Caption options store ---

/** Reactive store holding the current assembled caption options.
 *  Written by CaptionSettings, read by action functions. */
export const captionOptions: Writable<CaptionOptions> = writable({});

/** Job ID of the last captioning job started by this tab (batch or single-image).
 *  Used by +page.svelte to fire completion toasts. */
export const lastStartedJobId: Writable<string> = writable('');

// --- Actions ---

/** Start a batch captioning job. Reads current options from the store,
 *  handles password retry, registers the job, and seeds the SSE stores
 *  for immediate UI feedback. Returns the job ID. */
export async function startBatchCaptioning(datasetName: string): Promise<string> {
    setCaptioningStatus({
        status: 'starting',
        dataset_name: datasetName,
        processed: 0,
        total: 0,
        errors: 0,
        job_id: '',
        error: null,
        error_messages: []
    });
    const options = get(captionOptions);
    const info = await withPasswordRetry(() =>
        startCaptioning(datasetName, options as Record<string, unknown>)
    );
    registerJobId(info.job_id);
    lastStartedJobId.set(info.job_id);
    // Seed the status immediately so the progress bar / ETA appear without
    // waiting for the first SSE event. Fast completions may finish before
    // the HTTP response, so only seed when the job is actually running.
    if (info.status === 'running' || info.status === 'stopping') {
        setCaptioningStatus({
            status: info.status,
            dataset_name: info.dataset_name,
            processed: info.processed,
            total: info.total,
            errors: info.errors,
            job_id: info.job_id,
            error: info.error,
            error_messages: [],
            api_url: info.api_url,
            api_model_name: info.api_model_name
        });
    }
    return info.job_id;
}

/** Start a single-image captioning job. Reads current options from the store,
 *  handles password retry, registers the job, and seeds the SSE stores
 *  for immediate spinner feedback in ImageDetail. */
export async function captionSingleImage(datasetName: string, imageId: number): Promise<string> {
    setCaptioningStatus({
        status: 'starting',
        dataset_name: datasetName,
        processed: 0,
        total: 1,
        errors: 0,
        job_id: '',
        error: null,
        error_messages: []
    });
    setCurrentlyCaptioning({ dataset_name: datasetName, image_id: imageId });
    const options = get(captionOptions);
    const info = await withPasswordRetry(() =>
        apiCaptionSingleImage(datasetName, imageId, options as Record<string, unknown>)
    );
    registerJobId(info.job_id);

    // Seed stores so the spinner appears immediately, but only if the job
    // hasn't already finished. Fast completions (e.g. 0 images due to
    // no-overwrite) may finish before the HTTP response arrives; seeding
    // a stale 'running' state can race with the SSE 'done' event and leave
    // the UI stuck.
    if (info.status === 'running' || info.status === 'stopping') {
        setCaptioningStatus({
            status: info.status,
            dataset_name: info.dataset_name,
            processed: info.processed,
            total: info.total,
            errors: info.errors,
            job_id: info.job_id,
            error: info.error,
            error_messages: [],
            api_url: info.api_url,
            api_model_name: info.api_model_name
        });
        setCurrentlyCaptioning({ dataset_name: info.dataset_name, image_id: imageId });
    }

    lastStartedJobId.set(info.job_id);
    return info.job_id;
}

/** Stop a running captioning job. Shows a toast on failure.
 *  @param onError Optional callback receiving the error message for caller-side handling. */
export async function stopCaptioning(
    datasetName: string,
    onError?: (message: string) => void
): Promise<boolean> {
    try {
        const ok = await apiStopCaptioning(datasetName);
        if (!ok) {
            toast.error('Failed to stop captioning');
            onError?.('Failed to stop captioning');
        }
        return ok;
    } catch {
        toast.error('Failed to stop captioning: request failed');
        onError?.('request failed');
        return false;
    }
}
