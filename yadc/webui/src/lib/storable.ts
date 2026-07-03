/**
 * A writable Svelte store backed by ``localStorage``, with versioning,
 * optional migration, and optional Zod-based shape validation.
 *
 * Read flow on creation:
 *  1. Parse the stored JSON. On parse failure: console-warn, quarantine
 *     the raw blob under ``yadc/<key>.corrupt.<unix-ms>``, fall back to
 *     ``data`` (defaults).
 *  2. If the stored ``$version`` is behind the current default's, run
 *     ``migrate`` to upgrade. On migration failure: warn, quarantine,
 *     fall back to defaults.
 *  3. If a Zod ``schema`` was provided, run ``safeParse``. On failure:
 *     warn (with the structured ``error.issues``), quarantine, fall back
 *     to defaults. On success: ``store.set(result.data)`` — Zod's
 *     ``.default()`` backfills any missing fields, so additive schema
 *     changes self-heal without a migration step.
 *  4. If no schema was provided, shallow-merge the stored value with
 *     ``data`` as a defense-in-depth backfill so additive changes don't
 *     leave callers holding ``undefined`` fields.
 *
 * Persistence is wrapped in a try/catch so ``QuotaExceededError`` (or
 * Safari private mode, or storage disabled) cannot break the in-memory
 * store — the app keeps running, only persistence is lost, and we log a
 * warning once per store.
 */

import { get, writable, type Readable, type Writable } from 'svelte/store';
import { z, type ZodType } from 'zod';

/** Minimum shape for a value passed to ``storable`` — a numeric
 *  ``$version``. Callers' interfaces don't need an index signature; the
 *  library only reads ``$version`` at runtime. */
export interface VersionedValue {
    $version: number;
}

/** The returned store shape, in addition to ``Readable<T>``. */
export interface StorableStore<T> extends Readable<T> {
    set: Writable<T>['set'];
    update: Writable<T>['update'];
    /** Read the current value without subscribing. */
    get: () => T;
    /** Remove the persisted blob and reset the store to the supplied
     *  defaults. Errors from ``removeItem`` are swallowed. */
    clear: () => void;
}

/** Migration callback. Receives the parsed JSON (unknown shape) and the
 *  stored ``$version``; returns the shape that should be passed to the
 *  next stage (Zod validation, or final assignment). Returning a wrong
 *  shape is treated as a migration failure (warn + quarantine).
 *
 *  ``any`` is intentional: callers' migrations are typed against concrete
 *  interface versions, and TypeScript's contravariance on function
 *  parameters would otherwise reject them. Bivariance through ``any``
 *  keeps the assignment direction simple. */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type Migrate = (data: any, version: number) => any;

/** Overload: caller-defined type, no Zod schema. */
export function storable<T extends VersionedValue>(key: string, data: T): StorableStore<T>;
export function storable<T extends VersionedValue>(
    key: string,
    data: T,
    migrate: Migrate | null,
    schema?: undefined
): StorableStore<T>;

/** Overload: Zod schema is the source of truth for the value type. */
export function storable<S extends ZodType<unknown>>(
    key: string,
    data: z.infer<S>,
    migrate: Migrate | null,
    schema: S
): StorableStore<z.infer<S>>;

/** Implementation. */
export function storable(
    key: string,
    data: unknown,
    migrate: Migrate | null = null,
    schema?: ZodType<unknown>
): StorableStore<unknown> {
    const storage = browserStorage();
    const store = writable<unknown>(data);
    const stamp = Date.now();

    const warn = (msg: string, extra?: unknown): void => {
        console.warn(`storable[${key}]: ${msg}`, extra ?? '');
    };

    const quarantine = (raw: string): void => {
        try {
            storage.setItem(`${key}.corrupt.${stamp}`, raw);
        } catch {
            // Ignore — storage may be unavailable; we already logged the root cause.
        }
    };

    const currentVersion = (data as VersionedValue).$version;

    const raw = storage.getItem(key);
    if (raw !== null) {
        let parsed: unknown;
        try {
            parsed = JSON.parse(raw);
        } catch (e) {
            warn('parse failed, falling back to defaults', e);
            quarantine(raw);
            parsed = null;
        }

        if (parsed !== null) {
            // Migration runs before validation so the migrator can return
            // any shape and Zod re-checks it. When no migrate is provided
            // we still auto-bump ``$version`` so the common pure-additive
            // case (new field with ``.default()``) heals without a callback.
            const storedVersion = (parsed as { $version?: unknown }).$version;
            if (typeof storedVersion === 'number' && storedVersion !== currentVersion) {
                if (migrate !== null) {
                    try {
                        parsed = migrate(parsed, storedVersion);
                        warn(`migrated from v${storedVersion} → v${currentVersion}`);
                    } catch (e) {
                        warn('migration failed, falling back to defaults', e);
                        quarantine(raw);
                        parsed = null;
                    }
                } else {
                    parsed = { ...(parsed as object), $version: currentVersion };
                    warn(`auto-bumped $version v${storedVersion} → v${currentVersion}`);
                }
            }

            if (parsed !== null && schema) {
                const result = schema.safeParse(parsed);
                if (result.success) {
                    store.set(result.data);
                } else {
                    warn('schema validation failed, falling back to defaults', result.error.issues);
                    quarantine(raw);
                    // Store stays at defaults.
                }
            } else if (parsed !== null) {
                // No schema: shallow-merge with defaults so additive changes
                // don't leave callers holding ``undefined`` fields. Schema is
                // the recommended path going forward.
                store.set({ ...(data as object), ...(parsed as object) });
            }
        }
    }

    // Persistence with quota/private-mode resilience. Suppress repeat
    // warnings after the first failure so transient errors don't spam.
    // ``suppressPersist`` lets ``clear()`` reset the in-memory store to
    // defaults without re-writing them to storage on the same tick.
    let persistWarned = false;
    let suppressPersist = false;
    store.subscribe((value) => {
        if (suppressPersist) {
            return;
        }
        try {
            storage.setItem(key, JSON.stringify(value));
        } catch (e) {
            if (!persistWarned) {
                warn(
                    'persist failed (storage unavailable or quota exceeded); further failures suppressed',
                    e
                );
                persistWarned = true;
            }
        }
    });

    return {
        subscribe: store.subscribe,
        set: store.set,
        update: store.update,
        get: () => get(store),
        clear: () => {
            try {
                storage.removeItem(key);
            } catch {
                // Ignore — same reasons as the persist path.
            }
            suppressPersist = true;
            store.set(data);
            suppressPersist = false;
        }
    };
}

/** Return ``window.localStorage`` in the browser, an in-memory stub
 *  otherwise. The stub mirrors the Storage interface so SSR doesn't
 *  crash on ``setItem`` / ``getItem`` / etc. */
function browserStorage(): Storage {
    if (typeof window !== 'undefined') {
        return window.localStorage;
    }
    const mem: Record<string, string> = {};
    return {
        get length() {
            return Object.keys(mem).length;
        },
        clear() {
            for (const k of Object.keys(mem)) {
                delete mem[k];
            }
        },
        getItem(k) {
            return Object.prototype.hasOwnProperty.call(mem, k) ? mem[k] : null;
        },
        setItem(k, v) {
            mem[k] = String(v);
        },
        removeItem(k) {
            delete mem[k];
        },
        key(i) {
            return mem[Object.keys(mem)[i]] ?? null;
        }
    };
}

export default storable;
