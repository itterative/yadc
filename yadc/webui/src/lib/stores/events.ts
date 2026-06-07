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
import { generateId } from '$lib/random';

/** Unique ID for this browser tab, used to tag webui-originated changes
 *  so only this tab suppresses the resulting DatasetChangedEvent.
 *  Uses Math.random() since crypto.randomUUID() is unavailable over HTTP. */
export const clientId = browser ? `ui:${generateId()}` : '';
import { API_BASE } from '$lib/api';
import { TypedEventSource } from '$lib/events';
import { refreshEnvs } from '$lib/stores/env';
import { refreshTemplates } from '$lib/stores/templates';
import { writable, readonly, derived, get, type Readable } from 'svelte/store';
import storable from '$lib/storable.js';
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

// --- Internal writable stores ---

const _captioningStatus = writable<CaptioningStatus>({
    status: 'idle',
    dataset_name: '',
    processed: 0,
    total: 0,
    errors: 0,
    job_id: '',
    error: null,
    error_messages: [],
    elapsed: 0,
    max_concurrent: 1
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

/** Images currently being captioned, across all datasets. Empty when idle.
 *
 *  Holds at most ``max_concurrent`` entries per dataset — one per
 *  in-flight parallel task. Entries are added on
 *  ``image_caption_started`` and removed on ``image_caption_captioned``
 *  or ``image_caption_error``. Entries for a dataset are also cleared
 *  when the job for that dataset reaches a terminal state (done /
 *  error / cancelled), which catches images whose per-image events
 *  were never delivered (e.g. cancelled in-flight tasks).
 *
 *  Internally a ``Map`` keyed by ````${dataset_name}#${image_id}````
 *  so the SSE ``image_captioned`` handler can reliably remove the
 *  entry that ``image_caption_started`` added (plain ``Set`` would
 *  fail because every fresh object literal has a different identity).
 *  The public store exposes a ``Set`` of the same entries.
 */
type CaptioningTarget = { dataset_name: string; image_id: number };
const _currentlyCaptioningMap = writable<ReadonlyMap<string, CaptioningTarget>>(new Map());

function _key(dataset_name: string, image_id: number): string {
    return `${dataset_name}#${image_id}`;
}

/**
 * Captions received via SSE, keyed by image ID.
 *
 * Bounded LRU cache — oldest entries are evicted when the capacity is
 * exceeded.  Accessed entries are promoted so recently-viewed captions
 * survive eviction.
 *
 * NOTE: This feature may be refined before committing to the final
 * implementation.  The capacity and eviction strategy are subject to
 * change.
 */
const _storedCaptions = writable<Map<number, string>>(new Map());

const MAX_STORED_CAPTIONS = 64;

const MAX_ACTIVE_JOB_IDS = 16;

const MAX_TIMING_SAMPLES = 32;

interface TimingRingData {
    $version: number;
    timings: Record<string, number[]>;
}

/** Per-API+model ring buffer of successful caption durations (ms), persisted to localStorage. */
const _captionTimingRing = storable<TimingRingData>('yadc/captionTimingRing', {
    $version: 1,
    timings: {}
});

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

/** Images currently being captioned, across all datasets. Empty when idle.
 *  Under ``max_concurrent > 1`` this holds multiple entries at once. */
export const currentlyCaptioning: Readable<ReadonlySet<CaptioningTarget>> = derived(
    _currentlyCaptioningMap,
    ($map) => new Set($map.values())
);

/** Captions received via SSE, keyed by image ID. */
export const storedCaptions: Readable<Map<number, string>> = readonly(_storedCaptions);

/** Per-API+model ring buffer of successful caption durations (ms). */
export const captionTimingRing: Readable<Record<string, number[]>> = derived(
    _captionTimingRing,
    ($r) => $r.timings
);

/** Return a caption received via SSE for the given image, if any.
 *  Promotes the entry to most-recently-used (LRU eviction ordering). */
export function getStoredCaption(imageId: number): string | undefined {
    const map = get(_storedCaptions);
    const value = map.get(imageId);
    if (value !== undefined) {
        // Promote to end of iteration order (most-recently-used).
        _storedCaptions.update((m) => {
            m.delete(imageId);
            m.set(imageId, value);
            return m;
        });
    }
    return value;
}

/** Remove a stored caption after authoritative data has been fetched. */
export function clearStoredCaption(imageId: number): void {
    _storedCaptions.update((map) => {
        const next = new Map(map);
        next.delete(imageId);
        return next;
    });
}

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

/** Add an image to the currently-captioning set (idempotent).
 *  Used to seed immediate UI feedback before the SSE
 *  ``image_caption_started`` event arrives. The SSE handler removes
 *  the entry on completion. */
export function addCurrentlyCaptioning(datasetName: string, imageId: number): void {
    const k = _key(datasetName, imageId);
    _currentlyCaptioningMap.update((map) => {
        if (map.has(k)) {
            return map;
        }
        const next = new Map(map);
        next.set(k, { dataset_name: datasetName, image_id: imageId });
        return next;
    });
}

/** Remove all currently-captioning entries for a dataset. Called when
 *  a job reaches a terminal state so cancelled in-flight tasks don't
 *  leave stale entries behind. */
export function clearCurrentlyCaptioning(datasetName: string): void {
    _currentlyCaptioningMap.update((map) => {
        let changed = false;
        const next = new Map(map);
        for (const k of map.keys()) {
            if (k.startsWith(`${datasetName}#`)) {
                next.delete(k);
                changed = true;
            }
        }
        return changed ? next : map;
    });
}

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
        _captioningStatus.set(data);
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
            _currentlyCaptioningMap.update((map) => {
                let changed = false;
                const next = new Map(map);
                const prefix = `${data.dataset_name}#`;
                for (const k of map.keys()) {
                    if (k.startsWith(prefix)) {
                        next.delete(k);
                        changed = true;
                    }
                }
                return changed ? next : map;
            });
        }
    });

    _eventSource.listen('ping', PingEventZ, () => {
        // keepalive
    });

    _eventSource.listen('dataset_changed', DatasetChangedEventZ, (data) => {
        // Suppress events caused by our own captioning jobs or webui edits (but not other clients')
        if (data.job_id) {
            let suppress = false;
            _activeJobIds.subscribe((ids) => {
                suppress = ids.includes(data.job_id!);
            })();
            if (suppress) {
                return;
            }
            // Suppress events originating from this tab (webui edits, deletes, uploads)
            if (data.job_id === clientId) {
                return;
            }
            // Legacy: suppress events from older clients that still send "self"
            if (data.job_id === 'self') {
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
        // We may have missed change events while disconnected — refresh as a safety net.
        refreshEnvs();
        refreshTemplates();
    });

    _eventSource.listen('image_captioned', ImageCaptionedEventZ, (data) => {
        _lastCaptionedImage.set(data);
        _storedCaptions.update((map) => {
            const next = new Map(map);
            next.set(data.id, data.caption);
            if (next.size > MAX_STORED_CAPTIONS) {
                const firstKey = next.keys().next().value;
                if (firstKey !== undefined) {
                    next.delete(firstKey);
                }
            }
            return next;
        });
        // Record successful caption timing for ETA estimation.
        if (data.duration_ms > 0 && data.api_url && data.api_model_name) {
            const key = `${data.api_url}#${data.api_model_name}`;
            _captionTimingRing.update((ring) => {
                const arr = [...(ring.timings[key] ?? []), data.duration_ms];
                if (arr.length > MAX_TIMING_SAMPLES) {
                    arr.shift();
                }
                return { ...ring, timings: { ...ring.timings, [key]: arr } };
            });
        }
        // Remove this image from the in-flight map.  No-op if absent
        // (tolerates missed started events on reconnect).
        const k = _key(data.dataset_name, data.id);
        _currentlyCaptioningMap.update((map) => {
            if (!map.has(k)) {
                return map;
            }
            const next = new Map(map);
            next.delete(k);
            return next;
        });
    });

    _eventSource.listen('image_caption_error', ImageCaptionErrorEventZ, (data) => {
        _lastCaptionError.set(data);
        // Remove this image from the in-flight map (same idempotence
        // guarantee as for captioned).
        const k = _key(data.dataset_name, data.image_id);
        _currentlyCaptioningMap.update((map) => {
            if (!map.has(k)) {
                return map;
            }
            const next = new Map(map);
            next.delete(k);
            return next;
        });
    });

    _eventSource.listen('image_caption_started', ImageCaptionStartedEventZ, (data) => {
        const k = _key(data.dataset_name, data.image_id);
        _currentlyCaptioningMap.update((map) => {
            if (map.has(k)) {
                return map;
            }
            const next = new Map(map);
            next.set(k, { dataset_name: data.dataset_name, image_id: data.image_id });
            return next;
        });
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
