"""Bounded LRU (least-recently-used) dict.

Two flavours of bounded LRU live here:

- :class:`LRU` — count-limited (``LRU(capacity=N)``); evicts the
  least-recently-used entry when the dict exceeds *N* items.
- :class:`MemoryLRU` — bytes-limited (``MemoryLRU(max_bytes=M,
  size_fn=fn)``); evicts when the summed size of values (per
  *size_fn*) exceeds *M* bytes.

Both use an :class:`collections.OrderedDict` for O(1) insertion,
lookup, and eviction. Thread-safe when the caller holds a lock (as in
``TaggingService`` / ``CaptioningService``).
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Callable, override


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


class MemoryLRU[_K, _V]:
    """A bounded LRU cache that evicts when summed value sizes exceed a byte budget.

    Each value's size is computed by a caller-supplied
    ``size_fn(value) -> int`` callable. The function is invoked once per
    ``__setitem__`` (insert or overwrite); per-key sizes are cached so
    the *total* size can be checked in O(1) without re-measuring on
    every read.

    Read semantics match :class:`LRU`: setting a key moves it to
    most-recent; getting a key (incl. ``__contains__`` does NOT
    promote, same as ``LRU``) also promotes it. When the budget is
    exceeded the *least*-recently-used entry is evicted until the
    remaining entries fit.

    If a single value's size is larger than ``max_bytes`` it is
    silently skipped (``__setitem__`` is a no-op for that key) so the
    cache never lands in an empty-after-overflow state. The caller
    can detect this via ``__contains__`` if needed.

    Not thread-safe — callers must serialise access if used from
    multiple threads or asyncio tasks.
    """

    def __init__(self, max_bytes: int, size_fn: Callable[[_V], int] | None = None) -> None:
        if max_bytes < 1:
            raise ValueError(f"max_bytes must be >= 1, got {max_bytes}")
        if size_fn is None:
            raise ValueError("size_fn is required for MemoryLRU")
        self._max_bytes: int = max_bytes
        self._size_fn: Callable[[_V], int] = size_fn
        self._data: OrderedDict[_K, _V] = OrderedDict()
        # Per-key measured sizes so an overwrite correctly accounts for
        # the *old* size. ``_total_bytes`` keeps a running sum for cheap
        # budget checks without re-measuring.
        self._sizes: dict[_K, int] = {}
        self._total_bytes: int = 0

    @property
    def max_bytes(self) -> int:
        """The byte budget (summed sizes of values stay ≤ this)."""
        return self._max_bytes

    @property
    def total_bytes(self) -> int:
        """Summed size of values currently in the cache."""
        return self._total_bytes

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

        After insertion, evict LRU entries until ``total_bytes`` is
        within ``max_bytes``. Values whose size alone exceeds the
        budget are skipped (no-op) so the cache never empties itself
        on a single oversized insert.
        """
        new_size = self._size_fn(value)
        if new_size > self._max_bytes:
            # Single value exceeds the budget — skip insertion. The
            # user can bump ``max_bytes`` if they actually want this
            # entry cached; otherwise it's a usage bug.
            return
        if key in self._data:
            old_size = self._sizes.pop(key)
            self._total_bytes -= old_size
            self._data.move_to_end(key)
            self._data[key] = value
            self._sizes[key] = new_size
            self._total_bytes += new_size
        else:
            self._data[key] = value
            self._sizes[key] = new_size
            self._total_bytes += new_size
        self._enforce_budget()

    def __delitem__(self, key: _K) -> None:
        """Remove *key*.  Raises ``KeyError`` if absent.

        Subtracting the size keeps ``_total_bytes`` in sync with
        ``len(self._data)``; ``enforce_budget`` isn't needed (deletion
        cannot exceed the budget).
        """
        size = self._sizes.pop(key, 0)
        self._total_bytes -= size
        del self._data[key]

    def clear(self) -> None:
        """Remove all entries."""
        self._data.clear()
        self._sizes.clear()
        self._total_bytes = 0

    @override
    def __repr__(self) -> str:
        return f"MemoryLRU(max_bytes={self._max_bytes}, total_bytes={self._total_bytes}, {dict(self._data)})"

    # --- internal --------------------------------------------------------

    def _enforce_budget(self) -> None:
        """Pop LRU entries until ``_total_bytes`` is within ``max_bytes``.

        Uses the per-key size cache so evicting doesn't re-measure via
        ``size_fn`` (which the caller might not make idempotent).
        """
        while self._total_bytes > self._max_bytes:
            evicted_key, _ = self._data.popitem(last=False)
            evicted_size = self._sizes.pop(evicted_key, 0)
            self._total_bytes -= evicted_size
