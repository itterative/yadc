/**
 * Self-connecting SSE event store.
 *
 * On module load (browser only), opens a ``TypedEventSource`` to
 * ``GET /api/events``, validates every event with Zod, and pipes the
 * result into the corresponding writable stores.  Components just
 * import the stores they need — the connection is managed here.
 *
 * The server sends ``id:`` fields on every non-ping event and supports
 * ``Last-Event-ID`` resumption.  The browser's built-in ``EventSource``
 * automatically stores the last event ID and sends it on reconnect, so
 * no manual bookkeeping is needed.
 */

import { browser } from '$app/environment';
import { API_BASE } from '$lib/api';
import { TypedEventSource } from '$lib/events';
import { writable, readonly, type Readable } from 'svelte/store';
import { z } from 'zod';

// --- Zod schemas ---

export const CaptioningStatusZ = z.object({
    status: z.enum(['idle', 'running', 'stopping', 'error', 'done']),
    dataset_name: z.string(),
    processed: z.number(),
    total: z.number(),
    errors: z.number(),
    job_id: z.string().default(''),
    error: z.string().nullable(),
    error_messages: z.array(z.string()).default([])
});

export const PingEventZ = z.object({
    time: z.string()
});

export const DatasetChangedEventZ = z.object({
    dataset_name: z.string(),
    job_id: z.string().nullable().optional()
});

export const ResumptionFailedEventZ = z.object({
    requested_event_id: z.number()
});

export const ImageCaptionedEventZ = z.object({
    dataset_name: z.string(),
    job_id: z.string(),
    id: z.number(),
    file_name: z.string(),
    path: z.string(),
    has_caption: z.boolean(),
    has_toml: z.boolean(),
    width: z.number(),
    height: z.number(),
    draft_names: z.array(z.string()),
    last_modified_t: z.number().nullable()
});

export const ImageCaptionErrorEventZ = z.object({
    dataset_name: z.string(),
    job_id: z.string(),
    image_id: z.number(),
    error: z.string()
});

// --- Types ---

export type CaptioningStatus = z.infer<typeof CaptioningStatusZ>;
export type DatasetChangedEvent = z.infer<typeof DatasetChangedEventZ>;
export type ImageCaptionedEvent = z.infer<typeof ImageCaptionedEventZ>;
export type ImageCaptionErrorEvent = z.infer<typeof ImageCaptionErrorEventZ>;

// --- Internal writable stores ---

const _captioningStatus = writable<CaptioningStatus>({
    status: 'idle',
    dataset_name: '',
    processed: 0,
    total: 0,
    errors: 0,
    job_id: '',
    error: null,
    error_messages: []
});

const _pendingDatasetChanges = writable<Set<string>>(new Set());

/** Job IDs of captioning operations initiated by this frontend (bounded ring). */
const _activeJobIds = writable<string[]>([]);

/** Set to true when the server signals that SSE resumption failed. */
const _resumptionFailed = writable(false);

/** Latest per-image captioned event (null when no event has been received yet). */
const _lastCaptionedImage = writable<ImageCaptionedEvent | null>(null);

/** Latest per-image caption error event. */
const _lastCaptionError = writable<ImageCaptionErrorEvent | null>(null);

const MAX_ACTIVE_JOB_IDS = 16;

// --- Public readonly stores ---

/** Latest captioning job status received via SSE. */
export const captioningStatus: Readable<CaptioningStatus> = readonly(_captioningStatus);

/** Set of dataset names that have pending filesystem changes (not yet refreshed). */
export const pendingDatasetChanges: Readable<Set<string>> = readonly(_pendingDatasetChanges);

/** True when the SSE event history was too old to resume after a reconnect. */
export const resumptionFailed: Readable<boolean> = readonly(_resumptionFailed);

/** Most recent per-image captioned event. Set to null after consumption if needed. */
export const lastCaptionedImage: Readable<ImageCaptionedEvent | null> =
    readonly(_lastCaptionedImage);

/** Most recent per-image caption error event. */
export const lastCaptionError: Readable<ImageCaptionErrorEvent | null> =
    readonly(_lastCaptionError);

// --- Public actions ---

/** Clear the pending-change flag for a dataset (call after the user refreshes). */
export function clearPendingDatasetChange(datasetName: string): void {
    _pendingDatasetChanges.update((set) => {
        const next = new Set(set);
        next.delete(datasetName);
        return next;
    });
}

/** Register a captioning job ID initiated by this frontend. */
export function registerJobId(jobId: string): void {
    _activeJobIds.update((ids) => {
        if (ids.length >= MAX_ACTIVE_JOB_IDS) {
            return [...ids.slice(ids.length - MAX_ACTIVE_JOB_IDS + 1), jobId];
        }
        return [...ids, jobId];
    });
}

/** Dismiss the resumption-failed warning (e.g. after the user refreshes). */
export function clearResumptionFailed(): void {
    _resumptionFailed.set(false);
}

/** Explicitly set the captioning status (e.g. from an initial HTTP poll). */
export function setCaptioningStatus(status: CaptioningStatus): void {
    _captioningStatus.set(status);
}

// --- Self-connecting SSE lifecycle ---

let _eventSource: TypedEventSource | null = null;

function connect() {
    if (_eventSource !== null && _eventSource.readyState !== EventSource.CLOSED) {
        return;
    }

    try {
        _eventSource = new TypedEventSource(`${API_BASE}/api/events`);
    } catch (e) {
        console.warn('Failed to connect to event stream', { error: e });
        return;
    }

    _eventSource.listen('captioning_status', CaptioningStatusZ, (data) => {
        _captioningStatus.set(data);
    });

    _eventSource.listen('ping', PingEventZ, () => {
        // keepalive
    });

    _eventSource.listen('dataset_changed', DatasetChangedEventZ, (data) => {
        // Suppress events caused by our own captioning jobs (but not other clients')
        if (data.job_id) {
            let suppress = false;
            _activeJobIds.subscribe((ids) => {
                suppress = ids.includes(data.job_id!);
            })();
            if (suppress) {
                return;
            }
        }
        _pendingDatasetChanges.update((set) => {
            const next = new Set(set);
            next.add(data.dataset_name);
            return next;
        });
    });

    _eventSource.listen('resumption_failed', ResumptionFailedEventZ, (data) => {
        console.warn(
            'SSE resumption failed: server could not replay from event id %d',
            data.requested_event_id
        );
        _resumptionFailed.set(true);
    });

    _eventSource.listen('image_captioned', ImageCaptionedEventZ, (data) => {
        _lastCaptionedImage.set(data);
    });

    _eventSource.listen('image_caption_error', ImageCaptionErrorEventZ, (data) => {
        _lastCaptionError.set(data);
    });

    // Let the browser handle reconnection automatically.  The server sends a
    // ``retry:`` directive so the browser waits 5 s before reconnecting.  On
    // reconnect the browser sends ``Last-Event-ID`` automatically, allowing the
    // server to replay missed events.
    _eventSource.onerror = () => {
        // No-op: the browser will reconnect on its own.  We deliberately do NOT
        // call close() here — closing would discard the internal last-event-id
        // state and prevent automatic resumption.
    };
}

if (browser) {
    connect();

    // Watch for a permanently closed connection (e.g. server shutdown) and
    // attempt to reconnect.  The normal reconnect path is the browser's
    // built-in auto-reconnect (which preserves Last-Event-ID), so this is only
    // a fallback for edge cases.
    let reconnecting = false;
    window.setInterval(() => {
        if (_eventSource === null) {
            connect();
        } else if (!reconnecting && _eventSource.readyState === EventSource.CLOSED) {
            reconnecting = true;
            connect();
            reconnecting = false;
        }
    }, 5000);
}
