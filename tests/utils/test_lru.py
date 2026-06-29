"""Tests for LRU — bounded least-recently-used dict."""

import pytest

from yadc.utils.lru import LRU, MemoryLRU


class TestLRUBasic:
    def test_set_and_get(self):
        lru = LRU(3)
        lru["a"] = 1
        assert lru["a"] == 1
        assert len(lru) == 1

    def test_get_missing_returns_none(self):
        lru = LRU(3)
        assert lru.get("missing") is None

    def test_get_with_default(self):
        lru = LRU(3)
        assert lru.get("missing", -1) == -1

    def test_contains(self):
        lru = LRU(3)
        lru["a"] = 1
        assert "a" in lru
        assert "b" not in lru

    def test_delete(self):
        lru = LRU(3)
        lru["a"] = 1
        del lru["a"]
        assert "a" not in lru
        assert len(lru) == 0

    def test_clear(self):
        lru = LRU(3)
        lru["a"] = 1
        lru["b"] = 2
        lru.clear()
        assert len(lru) == 0


class TestLRUEviction:
    def test_evicts_lru_when_over_capacity(self):
        lru = LRU(2)
        lru["a"] = 1
        lru["b"] = 2
        lru["c"] = 3  # evicts "a"
        assert "a" not in lru
        assert lru["b"] == 2
        assert lru["c"] == 3

    def test_get_promotes(self):
        lru = LRU(2)
        lru["a"] = 1
        lru["b"] = 2
        lru.get("a")  # promotes "a"
        lru["c"] = 3  # evicts "b" (least recently used)
        assert "a" in lru
        assert "b" not in lru
        assert "c" in lru

    def test_set_existing_promotes(self):
        lru = LRU(2)
        lru["a"] = 1
        lru["b"] = 2
        lru["a"] = 10  # promotes "a"
        lru["c"] = 3  # evicts "b"
        assert lru["a"] == 10
        assert "b" not in lru

    def test_capacity_one(self):
        lru = LRU(1)
        lru["a"] = 1
        assert lru["a"] == 1
        lru["b"] = 2
        assert "a" not in lru
        assert lru["b"] == 2


class TestLRUValidation:
    def test_capacity_zero_raises(self):
        import pytest

        with pytest.raises(ValueError, match="capacity must be >= 1"):
            LRU(0)

    def test_negative_capacity_raises(self):
        import pytest

        with pytest.raises(ValueError, match="capacity must be >= 1"):
            LRU(-5)


# ---------------------------------------------------------------------------
# MemoryLRU — bytes-bounded LRU
# ---------------------------------------------------------------------------


class TestMemoryLRUBasic:
    def test_set_and_get(self):
        lru = MemoryLRU(max_bytes=100, size_fn=lambda v: v)
        lru["a"] = 30
        lru["b"] = 50
        assert lru["a"] == 30  # promotes "a"
        assert lru["b"] == 50
        assert len(lru) == 2

    def test_total_bytes_tracks_values(self):
        lru = MemoryLRU(max_bytes=100, size_fn=lambda v: v)
        lru["a"] = 30
        lru["b"] = 50
        assert lru.total_bytes == 80

    def test_get_with_default(self):
        lru = MemoryLRU(max_bytes=100, size_fn=lambda v: v)
        assert lru.get("missing", -1) == -1

    def test_contains(self):
        lru = MemoryLRU(max_bytes=100, size_fn=lambda v: v)
        lru["a"] = 30
        assert "a" in lru
        assert "b" not in lru

    def test_delete(self):
        lru = MemoryLRU(max_bytes=100, size_fn=lambda v: v)
        lru["a"] = 30
        del lru["a"]
        assert "a" not in lru
        assert lru.total_bytes == 0

    def test_clear(self):
        lru = MemoryLRU(max_bytes=100, size_fn=lambda v: v)
        lru["a"] = 30
        lru["b"] = 50
        lru.clear()
        assert len(lru) == 0
        assert lru.total_bytes == 0


class TestMemoryLRUEviction:
    def test_evicts_lru_when_over_budget(self):
        """Inserting past ``max_bytes`` evicts the LRU entry — and continues until under."""
        lru = MemoryLRU(max_bytes=100, size_fn=lambda v: v)
        lru["a"] = 30  # 30
        lru["b"] = 50  # 80
        lru["c"] = 40  # 120 — over. Evict "a" (30) → 90.
        assert lru.total_bytes == 90
        assert "a" not in lru
        assert "b" in lru
        assert "c" in lru

    def test_evicts_multiple_until_under_budget(self):
        lru = MemoryLRU(max_bytes=100, size_fn=lambda v: v)
        lru["a"] = 30
        lru["b"] = 50  # 80
        lru["c"] = 70  # 150 → evict a (120) → evict b (70)
        assert lru.total_bytes == 70
        assert "a" not in lru
        assert "b" not in lru
        assert "c" in lru

    def test_get_promotes(self):
        """Reading a key promotes it (delays its eviction)."""
        lru = MemoryLRU(max_bytes=100, size_fn=lambda v: v)
        lru["a"] = 30
        lru["b"] = 50  # 80
        lru.get("a")  # promotes "a"; "b" is now the oldest
        lru["c"] = 40  # 120 → evict "b" (70), not "a"
        assert "a" in lru
        assert "b" not in lru
        assert "c" in lru
        assert lru.total_bytes == 70  # 30 + 40

    def test_set_existing_promotes_and_resizes(self):
        """Overwriting with a different size correctly updates ``total_bytes``."""
        lru = MemoryLRU(max_bytes=200, size_fn=lambda v: v)
        lru["a"] = 30
        lru["b"] = 50  # 80
        lru["a"] = 70  # old 30 freed, new 70 added → 120, still within budget
        assert lru["a"] == 70
        assert "b" in lru
        assert lru.total_bytes == 120  # 70 + 50

    def test_oversized_value_is_skipped(self):
        """A single value larger than ``max_bytes`` is silently dropped (no eviction cascade)."""
        lru = MemoryLRU(max_bytes=100, size_fn=lambda v: v)
        lru["a"] = 30
        lru["b"] = 50  # 80
        lru["c"] = 500  # alone exceeds budget; skipped
        assert "c" not in lru
        assert lru.total_bytes == 80  # unchanged
        assert "a" in lru
        assert "b" in lru


class TestMemoryLRUValidation:
    def test_max_bytes_zero_raises(self):
        with pytest.raises(ValueError, match="max_bytes must be >= 1"):
            MemoryLRU(max_bytes=0, size_fn=lambda v: v)

    def test_max_bytes_negative_raises(self):
        with pytest.raises(ValueError, match="max_bytes must be >= 1"):
            MemoryLRU(max_bytes=-5, size_fn=lambda v: v)

    def test_size_fn_required(self):
        with pytest.raises(ValueError, match="size_fn is required"):
            MemoryLRU(max_bytes=100, size_fn=None)  # type: ignore[arg-type]


class TestMemoryLRUSizeFunction:
    def test_size_fn_invoked_per_setitem(self):
        """``size_fn`` is called once per insert / overwrite — not on reads."""
        calls = []

        def size_fn(v):
            calls.append(v)
            return len(v)

        lru = MemoryLRU(max_bytes=100, size_fn=size_fn)
        lru["a"] = "hello"  # 5
        assert calls == ["hello"]
        lru.get("a")
        lru["a"] in lru  # __contains__ no-promote
        assert calls == ["hello"]
        lru["a"] = "world"  # 5, overwrites the old
        assert calls == ["hello", "world"]

    def test_total_bytes_stays_in_sync_after_overwrite(self):
        """Overwriting with a different size correctly updates ``total_bytes``."""
        lru = MemoryLRU(max_bytes=200, size_fn=lambda v: v)
        lru["a"] = 30
        lru["b"] = 50  # 80
        assert lru.total_bytes == 80
        lru["a"] = 70  # 80 - 30 + 70 = 120
        assert lru.total_bytes == 120
        lru["a"] = 0  # 120 - 70 + 0 = 50
        assert lru.total_bytes == 50
