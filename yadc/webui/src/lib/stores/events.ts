/**
 * Self-connecting SSE event store.
 *
 * On module load (browser only), opens a ``TypedEventSource`` to
 * ``GET /api/events``, validates every event with Zod, and pipes the
 * result into the corresponding writable stores.  Components just
 * import the stores they need — the connection is managed here.
 */

import { browser } from "$app/environment";
import { API_BASE } from "$lib/api";
import { TypedEventSource } from "$lib/events";
import { writable, readonly, type Readable } from "svelte/store";
import { z } from "zod";

// --- Zod schemas ---

export const CaptioningStatusZ = z.object({
  status: z.enum(["idle", "running", "stopping", "error", "done"]),
  dataset_name: z.string(),
  processed: z.number(),
  total: z.number(),
  errors: z.number(),
  job_id: z.string().default(""),
  error: z.string().nullable(),
});

export const PingEventZ = z.object({
  time: z.string(),
});

export const DatasetChangedEventZ = z.object({
  dataset_name: z.string(),
  job_id: z.string().nullable().optional(),
});

// --- Types ---

export type CaptioningStatus = z.infer<typeof CaptioningStatusZ>;
export type DatasetChangedEvent = z.infer<typeof DatasetChangedEventZ>;

// --- Internal writable stores ---

const _captioningStatus = writable<CaptioningStatus>({
  status: "idle",
  dataset_name: "",
  processed: 0,
  total: 0,
  errors: 0,
  job_id: "",
  error: null,
});

const _pendingDatasetChanges = writable<Set<string>>(new Set());

/** Job IDs of captioning operations initiated by this frontend (bounded ring). */
const _activeJobIds = writable<string[]>([]);

const MAX_ACTIVE_JOB_IDS = 16;

// --- Public readonly stores ---

/** Latest captioning job status received via SSE. */
export const captioningStatus: Readable<CaptioningStatus> = readonly(_captioningStatus);

/** Set of dataset names that have pending filesystem changes (not yet refreshed). */
export const pendingDatasetChanges: Readable<Set<string>> = readonly(_pendingDatasetChanges);

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

// --- Self-connecting SSE lifecycle ---

let _eventSource: TypedEventSource | null = null;

function connect() {
  if (_eventSource !== null && _eventSource.readyState !== EventSource.CLOSED) {
    return;
  }

  try {
    _eventSource = new TypedEventSource(`${API_BASE}/api/events`);
  } catch (e) {
    console.warn("Failed to connect to event stream", { error: e });
    return;
  }

  _eventSource.listen("captioning_status", CaptioningStatusZ, (data) => {
    _captioningStatus.set(data);
  });

  _eventSource.listen("ping", PingEventZ, () => {
    // keepalive
  });

  _eventSource.listen("dataset_changed", DatasetChangedEventZ, (data) => {
    // Suppress events caused by our own captioning jobs (but not other clients')
    if (data.job_id) {
      let suppress = false;
      _activeJobIds.subscribe((ids) => { suppress = ids.includes(data.job_id!); })();
      if (suppress) return;
    }
    _pendingDatasetChanges.update((set) => {
      const next = new Set(set);
      next.add(data.dataset_name);
      return next;
    });
  });

  _eventSource.onerror = () => {
    _eventSource?.close();
  };
}

if (browser) {
  connect();

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
