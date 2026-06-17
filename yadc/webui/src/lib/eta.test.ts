import { describe, expect, it } from 'vitest';
import { computeEtaSeconds, effectiveConcurrency } from '$lib/eta';

describe('effectiveConcurrency', () => {
    it('is the configured value when enough work remains', () => {
        expect(effectiveConcurrency(2, 10, 0)).toBe(2);
        expect(effectiveConcurrency(4, 10, 6)).toBe(4);
    });

    it('caps at the remaining images (last image of a parallel job)', () => {
        // 1 image left of a 2-concurrent job runs alone.
        expect(effectiveConcurrency(2, 10, 9)).toBe(1);
        expect(effectiveConcurrency(3, 10, 8)).toBe(2);
    });

    it('is 1 for a single-image job regardless of max_concurrent', () => {
        expect(effectiveConcurrency(4, 1, 0)).toBe(1);
    });

    it('clamps to at least 1 when nothing remains (divisor safety)', () => {
        expect(effectiveConcurrency(2, 10, 10)).toBe(1);
        expect(effectiveConcurrency(2, 10, 12)).toBe(1);
    });
});

describe('computeEtaSeconds with capped concurrency', () => {
    // Regression: the last image of a 2-concurrent job runs alone, but
    // the status still reports max_concurrent=2. Feeding that straight
    // into the ETA halves the per-image time (5s instead of 10s).
    // Capping first restores the correct single-image estimate.
    const ring = [10000]; // 10s per image

    it('overstates speed by 2x without the cap, correct with it', () => {
        const capped = effectiveConcurrency(2, 10, 9);
        expect(capped).toBe(1);
        expect(computeEtaSeconds({ total: 10, processed: 9, ring, maxConcurrent: capped })).toBe(
            10
        );
        // The bug: the uncapped value would have returned half.
        expect(computeEtaSeconds({ total: 10, processed: 9, ring, maxConcurrent: 2 })).toBe(5);
    });
});
