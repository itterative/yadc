/**
 * Self-connecting SSE event router.
 *
 * On module load (browser only), opens a ``TypedEventSource`` to
 * ``GET /api/events``, validates every event with Zod, and routes the
 * result into the appropriate domain stores. Domain state lives in
 * its respective store (see ``$lib/stores/caption``,
 * ``$lib/stores/dataset``) — this file is concerned only with the
 * SSE transport and the routing itself.
 *
 * The server sends ``id:`` fields on every non-ping event and supports
 * ``Last-Event-ID`` resumption.  The browser's built-in ``EventSource``
 * automatically stores the last event ID and sends it on reconnect, so
 * no manual bookkeeping is needed.
 */

import { browser } from '$app/environment';
import { generateId } from '$lib/random';
import { API_BASE } from '$lib/api';
import { TypedEventSource } from '$lib/events';
import {
    addCurrentlyCaptioning,
    clearCurrentlyCaptioning,
    isOwnJobId,
    recordCaptionTiming,
    removeCurrentlyCaptioning,
    setCaptioningStatus,
    setImageRefined
} from '$lib/stores/caption';
import { addPendingDatasetChange } from '$lib/stores/dataset';
import { refreshEnvs } from '$lib/stores/env';
import { refreshTemplates } from '$lib/stores/templates';
import { writable, readonly, type Readable } from 'svelte/store';
import { z } from 'zod';

// --- Zod schemas ---

export const CaptioningStatusZ = z.object({
    status: z.enum(['idle', 'starting', 'running', 'stopping', 'error', 'done', 'cancelled']),
    dataset_name: z.string(),
    processed: z.number(),
    total: z.number(),
    errors: z.number(),
    job_id: z.string().default(''),
    error: z.string().nullable(),
    error_messages: z.array(z.string()).default([]),
    api_url: z.string().optional(),
    api_model_name: z.string().optional(),
    // Seconds since the job started. 0 before the job has actually
    // started. Used for the "elapsed" display.
    elapsed: z.number().default(0),
    // Configured concurrency for the job. The frontend divides the
    // per-image ETA by this so the estimate reflects actual wall-clock
    // throughput (max_concurrent requests in flight).
    max_concurrent: z.number().default(1)
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
    last_modified_t: z.number().nullable(),
    caption: z.string().default(''),
    duration_ms: z.number().default(0),
    api_url: z.string().optional(),
    api_model_name: z.string().optional()
});

export const ImageCaptionErrorEventZ = z.object({
    dataset_name: z.string(),
    job_id: z.string(),
    image_id: z.number(),
    error: z.string(),
    duration_ms: z.number().default(0),
    api_url: z.string().optional(),
    api_model_name: z.string().optional()
});

export const ImageCaptionStartedEventZ = z.object({
    dataset_name: z.string(),
    job_id: z.string(),
    image_id: z.number(),
    file_name: z.string()
});

export const ImageRefinedEventZ = z.object({
    dataset_name: z.string(),
    job_id: z.string(),
    image_id: z.number(),
    caption: z.string().default(''),
    source: z.enum(['caption', 'draft']).default('caption'),
    draft_name: z.string().default('')
});

export const EnvironmentsChangedEventZ = z.object({
    envs: z.array(z.string())
});

export const TemplatesChangedEventZ = z.object({
    templates: z.array(z.string())
});

// --- Types ---

export type CaptioningStatus = z.infer<typeof CaptioningStatusZ>;
export type DatasetChangedEvent = z.infer<typeof DatasetChangedEventZ>;
export type ImageCaptionedEvent = z.infer<typeof ImageCaptionedEventZ>;
export type ImageCaptionErrorEvent = z.infer<typeof ImageCaptionErrorEventZ>;
export type ImageCaptionStartedEvent = z.infer<typeof ImageCaptionStartedEventZ>;
export type EnvironmentsChangedEvent = z.infer<typeof EnvironmentsChangedEventZ>;
export type TemplatesChangedEvent = z.infer<typeof TemplatesChangedEventZ>;
export type ImageRefinedEvent = z.infer<typeof ImageRefinedEventZ>;

// --- Internal writable stores (pure event mirrors) ---

/** Set to true when the server signals that SSE resumption failed. */
const _resumptionFailed = writable(false);

/** Latest per-image captioned event (null when no event has been received yet). */
const _lastCaptionedImage = writable<ImageCaptionedEvent | null>(null);

/** Latest per-image caption error event. */
const _lastCaptionError = writable<ImageCaptionErrorEvent | null>(null);

// --- Public readonly stores ---

/** True when the SSE event history was too old to resume after a reconnect. */
export const resumptionFailed: Readable<boolean> = readonly(_resumptionFailed);

/** Most recent per-image captioned event. Set to null after consumption if needed. */
export const lastCaptionedImage: Readable<ImageCaptionedEvent | null> =
    readonly(_lastCaptionedImage);

/** Most recent per-image caption error event. */
export const lastCaptionError: Readable<ImageCaptionErrorEvent | null> =
    readonly(_lastCaptionError);

// --- Public actions ---

/** Dismiss the resumption-failed warning (e.g. after the user refreshes). */
export function clearResumptionFailed(): void {
    _resumptionFailed.set(false);
}

// --- Identity ---

/** Unique ID for this browser tab, used to tag webui-originated changes
 *  so only this tab suppresses the resulting DatasetChangedEvent.
 *  Uses Math.random() since crypto.randomUUID() is unavailable over HTTP. */
export const clientId = browser ? `ui:${generateId()}` : '';

// --- Self-connecting SSE lifecycle ---

let _eventSource: TypedEventSource | null = null;

/** Tracked across reconnections so manual reconnects can resume from the right position. */
let _lastEventId: string = '';

function connect() {
    if (_eventSource !== null && _eventSource.readyState !== EventSource.CLOSED) {
        return;
    }

    // Preserve the last event ID from the previous EventSource so the server
    // can replay missed events after a manual reconnection.  The browser's
    // built-in auto-reconnect preserves this internally, but creating a *new*
    // EventSource does not.
    let url = `${API_BASE}/api/events`;
    if (_lastEventId) {
        url += `?lastEventId=${_lastEventId}`;
    }

    try {
        _eventSource = new TypedEventSource(url);
    } catch (e) {
        console.warn('Failed to connect to event stream', { error: e });
        return;
    }

    _eventSource.listen('captioning_status', CaptioningStatusZ, (data) => {
        setCaptioningStatus(data);
        // Clear in-flight entries for the dataset when the job reaches
        // a terminal state. Catches cancellation (in-flight tasks
        // don't fire per-image completion events) and fresh starts on
        // a still-warm set.
        if (
            data.status === 'done' ||
            data.status === 'error' ||
            data.status === 'cancelled' ||
            data.status === 'idle'
        ) {
            clearCurrentlyCaptioning(data.dataset_name);
        }
    });

    _eventSource.listen('ping', PingEventZ, () => {
        // keepalive
    });

    _eventSource.listen('dataset_changed', DatasetChangedEventZ, (data) => {
        // Suppress events caused by our own captioning jobs, webui edits
        // (tagged with clientId), or legacy 'self' jobs.
        if (data.job_id) {
            if (isOwnJobId(data.job_id) || data.job_id === clientId || data.job_id === 'self') {
                return;
            }
        }
        addPendingDatasetChange(data.dataset_name);
    });

    _eventSource.listen('resumption_failed', ResumptionFailedEventZ, (data) => {
        console.warn(
            'SSE resumption failed: server could not replay from event id %d',
            data.requested_event_id
        );
        _resumptionFailed.set(true);
        // We may have missed change events while disconnected — refresh as a safety net.
        refreshEnvs();
        refreshTemplates();
    });

    _eventSource.listen('image_captioned', ImageCaptionedEventZ, (data) => {
        _lastCaptionedImage.set(data);
        if (data.api_url && data.api_model_name) {
            recordCaptionTiming(data.api_url, data.api_model_name, data.duration_ms);
        }
        removeCurrentlyCaptioning(data.dataset_name, data.id);
    });

    _eventSource.listen('image_caption_error', ImageCaptionErrorEventZ, (data) => {
        _lastCaptionError.set(data);
        removeCurrentlyCaptioning(data.dataset_name, data.image_id);
    });

    _eventSource.listen('image_caption_started', ImageCaptionStartedEventZ, (data) => {
        addCurrentlyCaptioning(data.dataset_name, data.image_id);
    });

    _eventSource.listen('image_refined', ImageRefinedEventZ, (data) => {
        setImageRefined(data);
        removeCurrentlyCaptioning(data.dataset_name, data.image_id);
    });

    _eventSource.listen('environments_changed', EnvironmentsChangedEventZ, () => {
        refreshEnvs();
    });

    _eventSource.listen('templates_changed', TemplatesChangedEventZ, () => {
        refreshTemplates();
    });

    // Let the browser handle reconnection automatically.  The server sends a
    // ``retry:`` directive so the browser waits 5 s before reconnecting.  On
    // reconnect the browser sends ``Last-Event-ID`` automatically, allowing the
    // server to replay missed events.
    _eventSource.onerror = () => {
        // Capture the last event ID before the EventSource potentially becomes
        // unusable, so that a manual reconnection can resume from the right
        // position.
        if (_eventSource) {
            _lastEventId = _eventSource.lastEventId || _lastEventId;
        }
        // We deliberately do NOT call close() here — closing would discard the
        // internal last-event-id state and prevent automatic resumption.
    };
}

if (browser) {
    connect();

    // --- Fallback reconnect for permanently closed connections ---
    //
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

    // --- Mobile background/foreground recovery ---
    //
    // Mobile browsers aggressively kill background TCP connections.  When the
    // user switches back to the browser, the EventSource may be in a stale
    // state (OPEN with a dead socket) or CLOSED without auto-reconnecting.
    // Force-reconnect with the tracked last-event-id so the server can replay
    // any events that were sent while the tab was suspended.
    let _hiddenAt: number | null = null;

    document.addEventListener('visibilitychange', () => {
        if (document.visibilityState === 'hidden') {
            _hiddenAt = Date.now();
        } else if (document.visibilityState === 'visible') {
            const wasHiddenMs = _hiddenAt ? Date.now() - _hiddenAt : 0;
            _hiddenAt = null;

            // Only force-reconnect if the page was hidden for longer than the
            // SSE retry interval (5 s).  Brief background switches (e.g.
            // notification shade pull-down) should not trigger a reconnect.
            if (wasHiddenMs < 5000 || _eventSource === null) {
                return;
            }

            // Capture the last event ID before closing.
            _lastEventId = _eventSource.lastEventId || _lastEventId;
            _eventSource.close();
            connect();
        }
    });
}
