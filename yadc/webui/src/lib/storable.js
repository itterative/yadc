import { writable } from 'svelte/store';

/**
 * @returns {Storage}
 */
function localStorage() {
	const isBrowser = typeof window !== 'undefined';

	if (isBrowser) {
		return window.localStorage;
	}

	/** @type {Record<string, string>} */
	let _storage = {};

	return {
		get length() {
			return Object.keys(_storage).length;
		},
		clear() {
			_storage = {};
		},
		getItem(k) {
			return _storage[k];
		},
		setItem(k, v) {
			_storage[k] = v;
		},
		removeItem(k) {
			delete _storage[k];
		},
		key(i) {
			return _storage[Object.keys(_storage)[i]];
		}
	};
}

/**
 * @template T
 * @typedef {T & { $version: number }} VersionedData<T>
 */

/**
 * A writable Svelte store backed by localStorage, with versioning and migration support.
 *
 * @template T
 * @param {string} key
 * @param {VersionedData<T>} data
 * @param {((data: VersionedData<T>, version: number) => VersionedData<T>) | null} migrate
 */
export default function storable(key, data, migrate = null) {
	const storage = localStorage();
	const store = writable(data);

	const storedData = storage.getItem(key);
	if (storedData) {
		let storedDataObj = JSON.parse(storedData);

		try {
			if (data['$version'] === storedDataObj['$version']) {
				store.set(storedDataObj);
			} else if (migrate !== null) {
				storedDataObj = migrate(storedDataObj, storedDataObj['$version']);
			}
		} catch (error) {
			console.error('storable failed to initialize (will use defaults)', { key, error });
		}
	}

	store.subscribe((value) => {
		storage.setItem(key, JSON.stringify(value));
	});

	return store;
}
