---
date: 2026-06-29
---
# Memory-aware LRU for the tagger result cache

`tagger_result_buffer_size: int = 500` (count-limited) was
redeemed for `tagger_result_max_memory_bytes: int = 128 MiB`
(bytes-limited) so the cache stays useful as the model / vocab
grows. `MemoryLRU` is a new class in `yadc/utils/lru.py`
alongside the existing count-limited `LRU`; same OrderedDict
backbone, O(1) read/write, eviction order is LRU regardless of
which limit triggered it. Single values larger than the budget are
silently skipped (no eviction cascade).

`tamer_result_size` (in `yadc/taggers/base.py`) is the size
function: recursive `sys.getsizeof` walk with `id()`-memoized
cycle protection over `result.tags` (dict) and `result.categories`
(dict of lists).

**Files:** `yadc/utils/lru.py` (`MemoryLRU` class), `yadc/utils/__init__.py`
(export), `yadc/taggers/base.py` (`_deep_size` + `tamer_result_size`),
`yadc/api/configuration.py` (rename + new default), `yadc/api/services/tagging.py`
(swap `LRU` for `MemoryLRU` with `tamer_result_size`), `tests/utils/test_lru.py`
(`MemoryLRU` TestCases), `tests/taggers/test_service.py` (rewrite
`test_lru_evicts_when_over_capacity` -> `test_lru_evicts_when_over_budget`),
`docs/tagger-architecture.md` (new default + "MemoryLRU" framing).
