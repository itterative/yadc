/** Tests for ``setTaggingStatus`` — guards against late responses overwriting terminal SSE state. */

import { get } from 'svelte/store';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import {
    INITIAL_TAGGING_STATUS,
    resetTaggingStatus,
    setTaggingStatus,
    taggingStatuses
} from './status';
import type { TagJobInfo } from './types';

function makeStatus(overrides: Partial<TagJobInfo> = {}): TagJobInfo {
    return {
        ...INITIAL_TAGGING_STATUS,
        dataset_name: 'anime',
        job_id: 'job-1',
        status: 'running',
        total: 41,
        processed: 0,
        ...overrides
    };
}

describe('setTaggingStatus — terminal vs non-terminal gating', () => {
    beforeEach(() => {
        // Each test starts from a clean map. ``force: true`` clears any
        // leftover entry from a previous test (the action seed may have
        // a different job_id).
        resetTaggingStatus('anime', true);
    });

    afterEach(() => {
        resetTaggingStatus('anime', true);
    });

    it('lets a running status overwrite idle', () => {
        setTaggingStatus(makeStatus({ status: 'running', processed: 0 }));
        expect(get(taggingStatuses).get('anime')?.status).toBe('running');
    });

    it('lets a terminal status overwrite running', () => {
        setTaggingStatus(makeStatus({ status: 'running' }));
        setTaggingStatus(makeStatus({ status: 'done', processed: 41 }));
        expect(get(taggingStatuses).get('anime')?.status).toBe('done');
    });

    it('ignores a non-terminal response that races after a terminal SSE for the same job', () => {
        // Simulate the race: SSE ``tag_job_status`` ('done') arrives first,
        // then the HTTP response from ``POST /tagger/start`` lands late
        // carrying the initial 'running' snapshot. The response must NOT
        // yank the UI back to 'running'.
        setTaggingStatus(makeStatus({ status: 'done', processed: 41, job_id: 'job-1' }));
        setTaggingStatus(makeStatus({ status: 'running', processed: 0, job_id: 'job-1' }));
        const after = get(taggingStatuses).get('anime');
        expect(after?.status).toBe('done');
        expect(after?.processed).toBe(41);
    });

    it('ignores a non-terminal response after each terminal kind', () => {
        for (const terminal of ['done', 'error', 'cancelled'] as const) {
            resetTaggingStatus('anime', true);
            setTaggingStatus(makeStatus({ status: terminal, job_id: 'job-1' }));
            setTaggingStatus(makeStatus({ status: 'running', job_id: 'job-1' }));
            expect(get(taggingStatuses).get('anime')?.status).toBe(terminal);
        }
    });

    it('lets a new job (different job_id) overwrite a terminal entry', () => {
        // ``resetTaggingStatus`` only evicts idle/running; a terminal
        // status for job-1 must be overwritten by a fresh running seed
        // for job-2.
        setTaggingStatus(makeStatus({ status: 'done', job_id: 'job-1' }));
        setTaggingStatus(makeStatus({ status: 'running', job_id: 'job-2' }));
        expect(get(taggingStatuses).get('anime')?.status).toBe('running');
        expect(get(taggingStatuses).get('anime')?.job_id).toBe('job-2');
    });

    it('lets the optimistic seed (empty job_id) be overwritten by the real response', () => {
        // The action seeds with job_id='' before the POST returns. The
        // response carries the real job_id. The gate must allow this
        // transition.
        setTaggingStatus(makeStatus({ status: 'running', job_id: '' }));
        setTaggingStatus(makeStatus({ status: 'running', job_id: 'job-real' }));
        expect(get(taggingStatuses).get('anime')?.job_id).toBe('job-real');
    });

    it('lets intermediate progress updates through while a job is still running', () => {
        setTaggingStatus(makeStatus({ status: 'running', processed: 0 }));
        setTaggingStatus(makeStatus({ status: 'running', processed: 10 }));
        setTaggingStatus(makeStatus({ status: 'running', processed: 30 }));
        expect(get(taggingStatuses).get('anime')?.processed).toBe(30);
    });
});
