import { writable, get } from 'svelte/store';

export type StorageType = 'local' | 'session';

function getStorage(type: StorageType): Storage {
    if (typeof window === 'undefined') {
        return {
            get length() {
                return 0;
            },
            clear() {},
            getItem() {
                return null;
            },
            setItem() {},
            removeItem() {},
            key() {
                return null;
            }
        } as Storage;
    }
    return type === 'local' ? window.localStorage : window.sessionStorage;
}

/**
 * A writable Svelte store backed by Web Storage (localStorage or sessionStorage).
 *
 * @template T
 * @param key - Storage key
 * @param initial - Initial value (also used as fallback when storage is empty or corrupted)
 * @param type - 'local' for localStorage, 'session' for sessionStorage
 */
export function storageStore<T>(key: string, initial: T, type: StorageType = 'local') {
    const storage = getStorage(type);
    const store = writable<T>(initial);

    const raw = storage.getItem(key);
    if (raw !== null) {
        try {
            store.set(JSON.parse(raw));
        } catch {
            // ignore parse errors, keep initial
        }
    }

    store.subscribe((value) => {
        storage.setItem(key, JSON.stringify(value));
    });

    return {
        subscribe: store.subscribe,
        set: store.set,
        update: store.update,
        get: () => get(store),
        clear: () => {
            storage.removeItem(key);
            store.set(initial);
        }
    };
}
