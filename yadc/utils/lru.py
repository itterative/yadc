"""Bounded LRU (least-recently-used) dict.

Uses an ``OrderedDict`` for O(1) insertion, lookup, and eviction.
Thread-safe when the caller holds a lock (as in ``CaptioningService``).
"""

from __future__ import annotations

from collections import OrderedDict
from typing import override


class LRU[_K, _V]:
    """A bounded LRU cache that behaves like a read-write dict.

    Setting a key moves it to most-recent.  Getting a key also promotes it
    (mimicking ``functools.lru_cache`` behaviour).  When the capacity is
    exceeded the *least*-recently-used entry is evicted.

    Not thread-safe — callers must serialise access if used from multiple
    threads or asyncio tasks (an ``asyncio.Lock`` is sufficient for
    single-threaded async code).
    """

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError(f"LRU capacity must be >= 1, got {capacity}")
        self._capacity: int = capacity
        self._data: OrderedDict[_K, _V] = OrderedDict()

    @property
    def capacity(self) -> int:
        return self._capacity

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: _K) -> bool:
        return key in self._data

    def get(self, key: _K, default: _V | None = None) -> _V | None:
        """Return the value for *key*, promoting it to most-recently-used.

        Returns *default* (``None``) if the key is absent.
        """
        if key not in self._data:
            return default
        self._data.move_to_end(key)
        return self._data[key]

    def __getitem__(self, key: _K) -> _V:
        """Return the value for *key*, promoting it to most-recently-used.

        Raises ``KeyError`` if the key is absent.
        """
        self._data.move_to_end(key)
        return self._data[key]

    def __setitem__(self, key: _K, value: _V) -> None:
        """Insert or overwrite *key*, promoting it to most-recently-used.

        If the capacity is exceeded after insertion the least-recently-used
        entry is evicted.
        """
        if key in self._data:
            self._data.move_to_end(key)
            self._data[key] = value
        else:
            self._data[key] = value
            if len(self._data) > self._capacity:
                self._data.popitem(last=False)

    def __delitem__(self, key: _K) -> None:
        """Remove *key*.  Raises ``KeyError`` if absent."""
        del self._data[key]

    def clear(self) -> None:
        """Remove all entries."""
        self._data.clear()

    @override
    def __repr__(self) -> str:
        return f"LRU({self._capacity}, {dict(self._data)})"
