import { derived, type Readable } from 'svelte/store';
import storable from '$lib/storable.js';
import { z } from 'zod';

const TimingRingSchema = z.object({
    $version: z.number(),
    timings: z.record(z.string(), z.array(z.number())).default({})
});

/** Per-API+model ring buffer of successful caption durations (ms),
 *  persisted to localStorage. Drives the ETA estimate in the topbar
 *  by averaging the per-image duration and dividing by
 *  ``max_concurrent``. */
const _captionTimingRing = storable(
    'yadc/captionTimingRing',
    { $version: 1, timings: {} },
    null,
    TimingRingSchema
);

const MAX_TIMING_SAMPLES = 32;

/** Per-API+model ring buffer of successful caption durations (ms). */
export const captionTimingRing: Readable<Record<string, number[]>> = derived(
    _captionTimingRing,
    ($r) => $r.timings
);

/** Record a successful caption duration for ETA estimation.
 *  Silently drops non-positive durations. */
export function recordCaptionTiming(apiUrl: string, modelName: string, durationMs: number): void {
    if (durationMs <= 0) {
        return;
    }
    const key = `${apiUrl}#${modelName}`;
    _captionTimingRing.update((ring) => {
        const arr = [...(ring.timings[key] ?? []), durationMs];
        if (arr.length > MAX_TIMING_SAMPLES) {
            arr.shift();
        }
        return { ...ring, timings: { ...ring.timings, [key]: arr } };
    });
}
