/**
 * ETA estimation for running captioning jobs.
 *
 * The model: each image takes a roughly stable per-image time, so
 * ``throughput = max_concurrent / per_image_time`` images per second,
 * and ``remaining = (total - processed) / throughput`` wall-clock
 * seconds. Concurrency is factored in via ``max_concurrent`` (sourced
 * from the status event), not via the ring buffer (which stays
 * per-model — per-image duration is roughly constant regardless of
 * concurrency).
 *
 * Per-image durations are smoothed with an exponential moving average
 * over the ring buffer. EMA weights more recent samples higher, so
 * the estimate adapts quickly to model warmup, rate-limit changes,
 * and similar transient effects.
 */

/** EMA discount factor. Higher = more reactive to recent samples.
 *  0.3 gives an effective window of ~3-4 samples. */
const EMA_ALPHA = 0.3;

/** Compute the EMA-smoothed duration from a chronological list of
 *  per-image ``duration_ms`` samples. The list is treated as a single
 *  fold: ``ema_k = α * x_k + (1-α) * ema_{k-1}``, with ``ema_0 = x_0``.
 *  Returns ``null`` for an empty list. */
export function emaFromSamples(samples: readonly number[]): number | null {
    if (samples.length === 0) {
        return null;
    }
    let ema = samples[0];
    for (let i = 1; i < samples.length; i++) {
        ema = EMA_ALPHA * samples[i] + (1 - EMA_ALPHA) * ema;
    }
    return ema;
}

export interface EtaInput {
    /** Total images to caption in this job. */
    total: number;
    /** Images already captioned (cumulative). */
    processed: number;
    /** Per-image ``duration_ms`` samples for the current (api_url, model).
     *  Empty = no estimate yet. */
    ring: readonly number[];
    /** Configured concurrency (1 for sequential, >1 for parallel). */
    maxConcurrent: number;
}

/** Estimate the wall-clock seconds remaining for a running captioning job.
 *  Returns ``null`` when we don't have enough information yet (no
 *  samples, or the job is already done). */
export function computeEtaSeconds(input: EtaInput): number | null {
    if (input.total <= input.processed) {
        return 0;
    }
    const smoothedMs = emaFromSamples(input.ring);
    if (smoothedMs === null || smoothedMs <= 0) {
        return null;
    }
    // Per-image wall-clock time, accounting for concurrency. The
    // ring buffer is in "per-image" units (one image, regardless of
    // how many were in flight), so throughput = max_concurrent /
    // per_image_seconds.
    const concurrent = Math.max(1, input.maxConcurrent);
    const perImageSec = smoothedMs / 1000;
    const throughput = concurrent / perImageSec;
    const remaining = input.total - input.processed;
    return Math.round(remaining / throughput);
}
