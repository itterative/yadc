/** Tests for ``setCaptioningStatus`` — guards against late responses overwriting terminal SSE state. */

import { get } from 'svelte/store';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import {
    INITIAL_CAPTIONING_STATUS,
    resetCaptioningStatus,
    setCaptioningStatus,
    captioningStatuses
} from './status';
import type { CaptioningStatus } from '../events';

function makeStatus(overrides: Partial<CaptioningStatus> = {}): CaptioningStatus {
    return {
        ...INITIAL_CAPTIONING_STATUS,
        dataset_name: 'anime',
        job_id: 'job-1',
        status: 'running',
        total: 10,
        processed: 0,
        ...overrides
    };
}

describe('setCaptioningStatus — terminal vs non-terminal gating', () => {
    beforeEach(() => {
        resetCaptioningStatus('anime', true);
    });

    afterEach(() => {
        resetCaptioningStatus('anime', true);
    });

    it('lets a running status overwrite idle', () => {
        setCaptioningStatus(makeStatus({ status: 'running', processed: 0 }));
        expect(get(captioningStatuses).get('anime')?.status).toBe('running');
    });

    it('lets a terminal status overwrite running', () => {
        setCaptioningStatus(makeStatus({ status: 'running' }));
        setCaptioningStatus(makeStatus({ status: 'done', processed: 10 }));
        expect(get(captioningStatuses).get('anime')?.status).toBe('done');
    });

    it('ignores a non-terminal response that races after a terminal SSE for the same job', () => {
        // SSE ``captioning_status`` ('done') arrives first, then the
        // HTTP response from ``POST /captioner/start`` lands late with
        // 'running'. The response must NOT yank the UI back.
        setCaptioningStatus(makeStatus({ status: 'done', processed: 10, job_id: 'job-1' }));
        setCaptioningStatus(makeStatus({ status: 'running', processed: 0, job_id: 'job-1' }));
        const after = get(captioningStatuses).get('anime');
        expect(after?.status).toBe('done');
        expect(after?.processed).toBe(10);
    });

    it('ignores a non-terminal response after each terminal kind', () => {
        for (const terminal of ['done', 'error', 'cancelled'] as const) {
            resetCaptioningStatus('anime', true);
            setCaptioningStatus(makeStatus({ status: terminal, job_id: 'job-1' }));
            setCaptioningStatus(makeStatus({ status: 'running', job_id: 'job-1' }));
            expect(get(captioningStatuses).get('anime')?.status).toBe(terminal);
        }
    });

    it('lets a new job (different job_id) overwrite a terminal entry', () => {
        setCaptioningStatus(makeStatus({ status: 'done', job_id: 'job-1' }));
        setCaptioningStatus(makeStatus({ status: 'running', job_id: 'job-2' }));
        expect(get(captioningStatuses).get('anime')?.status).toBe('running');
        expect(get(captioningStatuses).get('anime')?.job_id).toBe('job-2');
    });

    it('lets the optimistic seed (empty job_id) be overwritten by the real response', () => {
        // The action seeds with job_id='' before the POST returns. The
        // response carries the real job_id. The gate must allow this.
        setCaptioningStatus(makeStatus({ status: 'starting', job_id: '' }));
        setCaptioningStatus(makeStatus({ status: 'running', job_id: 'job-real' }));
        expect(get(captioningStatuses).get('anime')?.job_id).toBe('job-real');
    });

    it('lets intermediate progress updates through while a job is still running', () => {
        setCaptioningStatus(makeStatus({ status: 'running', processed: 0 }));
        setCaptioningStatus(makeStatus({ status: 'running', processed: 4 }));
        setCaptioningStatus(makeStatus({ status: 'running', processed: 8 }));
        expect(get(captioningStatuses).get('anime')?.processed).toBe(8);
    });
});
