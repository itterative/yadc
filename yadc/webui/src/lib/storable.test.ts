/** Tests for ``storable`` — resilient localStorage-backed writable. */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { get } from 'svelte/store';
import { z } from 'zod';

import storable from '$lib/storable';

beforeEach(() => {
    localStorage.clear();
});

afterEach(() => {
    vi.restoreAllMocks();
    localStorage.clear();
});

describe('storable — plain versioned usage', () => {
    it('starts at the supplied defaults when storage is empty', () => {
        const store = storable('yadc/test', { $version: 1, count: 0 });
        expect(get(store)).toEqual({ $version: 1, count: 0 });
    });

    it('restores a value previously persisted by an earlier call', () => {
        storable('yadc/test', { $version: 1, count: 0 }).set({ $version: 1, count: 7 });
        const store = storable('yadc/test', { $version: 1, count: 0 });
        expect(get(store)).toEqual({ $version: 1, count: 7 });
    });

    it('shallow-merges stored data with defaults to backfill missing fields', () => {
        // Simulates a schema addition without bumping $version. The old
        // blob is missing the new field; the merge keeps the user's data
        // and backfills ``extra`` from the defaults.
        localStorage.setItem('yadc/test', JSON.stringify({ $version: 2, count: 4 /* no extra */ }));
        const store = storable('yadc/test', { $version: 2, count: 0, extra: 'default' });
        expect(get(store)).toEqual({ $version: 2, count: 4, extra: 'default' });
    });

    it('runs the migrate function when stored $version is behind', () => {
        const migrate = vi.fn((raw: { $version: number; [k: string]: unknown }) => ({
            ...raw,
            $version: 2,
            added: 'migrated'
        }));
        localStorage.setItem('yadc/test', JSON.stringify({ $version: 1, count: 3 }));
        const store = storable('yadc/test', { $version: 2, count: 0, added: '' }, migrate);
        expect(get(store)).toEqual({ $version: 2, count: 3, added: 'migrated' });
        expect(migrate).toHaveBeenCalledOnce();
        expect(migrate).toHaveBeenCalledWith({ $version: 1, count: 3 }, 1);
    });

    it('exposes clear() that wipes the key and resets to defaults', () => {
        const store = storable('yadc/test', { $version: 1, count: 0 });
        store.set({ $version: 1, count: 9 });
        expect(localStorage.getItem('yadc/test')).not.toBeNull();

        store.clear();

        expect(localStorage.getItem('yadc/test')).toBeNull();
        expect(get(store)).toEqual({ $version: 1, count: 0 });
    });
});

describe('storable — failure recovery', () => {
    it('falls back to defaults and quarantines on JSON parse error', () => {
        const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
        localStorage.setItem('yadc/test', '{ not json');

        const store = storable('yadc/test', { $version: 1, count: 99 });

        expect(get(store)).toEqual({ $version: 1, count: 99 });
        expect(warn).toHaveBeenCalledWith(
            'storable[yadc/test]: parse failed, falling back to defaults',
            expect.anything()
        );

        const corruptKeys = Object.keys(localStorage).filter((k) =>
            k.startsWith('yadc/test.corrupt.')
        );
        expect(corruptKeys).toHaveLength(1);
        expect(localStorage.getItem(corruptKeys[0])).toBe('{ not json');
    });

    it('falls back to defaults and quarantines on migration failure', () => {
        const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
        localStorage.setItem('yadc/test', JSON.stringify({ $version: 1, count: 3 }));
        const migrate = vi.fn(() => {
            throw new Error('boom');
        });

        const store = storable('yadc/test', { $version: 2, count: 0 }, migrate);

        expect(get(store)).toEqual({ $version: 2, count: 0 });
        expect(migrate).toHaveBeenCalledOnce();
        expect(warn).toHaveBeenCalledWith(
            'storable[yadc/test]: migration failed, falling back to defaults',
            expect.any(Error)
        );

        const corruptKeys = Object.keys(localStorage).filter((k) =>
            k.startsWith('yadc/test.corrupt.')
        );
        expect(corruptKeys).toHaveLength(1);
    });

    it('falls back to defaults and quarantines on schema validation error', () => {
        const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
        const schema = z.object({
            $version: z.literal(1),
            count: z.number()
        });
        // Stored value missing ``count``.
        localStorage.setItem('yadc/test', JSON.stringify({ $version: 1 }));

        const store = storable('yadc/test', { $version: 1, count: 99 }, null, schema);

        expect(get(store)).toEqual({ $version: 1, count: 99 });
        expect(warn).toHaveBeenCalledWith(
            'storable[yadc/test]: schema validation failed, falling back to defaults',
            expect.anything()
        );

        const corruptKeys = Object.keys(localStorage).filter((k) =>
            k.startsWith('yadc/test.corrupt.')
        );
        expect(corruptKeys).toHaveLength(1);
    });
});

describe('storable — Zod schema', () => {
    it('uses schema as the source of truth and backfills via .default()', () => {
        // This is the exact shape of the bug that bit ``tagHighlights``:
        // an older stored blob is missing a field that the current schema
        // declares with a default. The store should self-heal on load.
        const schema = z.object({
            $version: z.literal(2),
            starred: z.array(z.string()).default([]),
            desired: z.array(z.string()).default([]),
            undesired: z.array(z.string()).default([]),
            categoryOverrides: z.record(z.string(), z.string()).default({})
        });

        // v1 blob: no categoryOverrides.
        localStorage.setItem(
            'yadc/test',
            JSON.stringify({
                $version: 1,
                starred: ['a'],
                desired: [],
                undesired: []
            })
        );

        const defaults: z.infer<typeof schema> = {
            $version: 2,
            starred: [],
            desired: [],
            undesired: [],
            categoryOverrides: {}
        };

        const store = storable('yadc/test', defaults, null, schema);

        // .default() backfilled every missing field, including
        // categoryOverrides: {} — callers can now read/write without
        // hitting undefined.
        expect(get(store)).toEqual({
            $version: 2,
            starred: ['a'],
            desired: [],
            undesired: [],
            categoryOverrides: {}
        });
    });

    it('auto-bumps $version when no migrate is provided, then validates', () => {
        const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
        const schema = z.object({
            $version: z.literal(2),
            count: z.number()
        });
        // Stored blob is at v1 with the same data shape — should heal
        // itself by auto-bumping $version and re-validating.
        localStorage.setItem('yadc/test', JSON.stringify({ $version: 1, count: 5 }));

        const store = storable('yadc/test', { $version: 2, count: 0 }, null, schema);

        expect(get(store)).toEqual({ $version: 2, count: 5 });
        expect(warn).toHaveBeenCalledWith(
            'storable[yadc/test]: auto-bumped $version v1 → v2',
            expect.anything()
        );
    });
});

describe('storable — persistence resilience', () => {
    it('keeps the in-memory store working when setItem throws (quota / private mode)', () => {
        const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
        // jsdom's localStorage doesn't have a quota, so simulate one.
        vi.spyOn(Storage.prototype, 'setItem').mockImplementation(function () {
            throw new DOMException('quota exceeded', 'QuotaExceededError');
        });

        const store = storable('yadc/test', { $version: 1, count: 0 });

        // Updates still take effect in memory even though persistence fails.
        expect(() => store.set({ $version: 1, count: 1 })).not.toThrow();
        expect(get(store)).toEqual({ $version: 1, count: 1 });

        expect(warn).toHaveBeenCalledWith(
            expect.stringContaining('persist failed'),
            expect.any(DOMException)
        );
    });

    it('warns at most once per store when persistence keeps failing', () => {
        const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
        vi.spyOn(Storage.prototype, 'setItem').mockImplementation(function () {
            throw new DOMException('quota exceeded', 'QuotaExceededError');
        });

        const store = storable('yadc/test', { $version: 1, count: 0 });

        store.set({ $version: 1, count: 1 });
        store.set({ $version: 1, count: 2 });
        store.set({ $version: 1, count: 3 });

        // Only the parse-/schema-warn path uses an exact prefix; for the
        // persist path we assert ``stringContaining`` and count >= 1
        // (the initial subscribe firing is also wrapped, but the first
        // set is what surfaces the warn at most once).
        const persistWarns = warn.mock.calls.filter(
            ([msg]) => typeof msg === 'string' && msg.includes('persist failed')
        );
        expect(persistWarns).toHaveLength(1);
    });
});
