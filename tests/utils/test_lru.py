"""Tests for LRU — bounded least-recently-used dict."""

from yadc.utils.lru import LRU


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
