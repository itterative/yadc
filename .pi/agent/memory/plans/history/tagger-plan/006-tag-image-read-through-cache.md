---
date: 2026-06-29
---
# `tag_image` is read-through on the LRU result cache + bucketed cache key

`TaggingService._tag_results` was populated but never consulted on
subsequent tag requests. POST `/tag` and the batch job's per-image
loop ran the model unconditionally. Fixed by adding a cache lookup
at the top of `tag_image`: hit returns the cached value (re-applied
at the request's effective thresholds + `replace_underscores`)
with `ImageTaggedEvent` dispatched (`duration_ms=0`), miss falls
through to the model.

Followed up the same day with a looser cache contract:

- `rating_threshold` is hardcoded to `0` in the cache key (never
  pre-filtered at write time; re-applied at retrieval).
- `general_threshold` / `character_threshold` are floored to the
  nearest `0.2` boundary (`bucket_threshold` helper) so nearby
  request thresholds share a slot. The cached value is
  post-thresholded at the bucket floors (a superset of any stricter
  request), then re-filtered at the request's effective thresholds
  on the way out.
- `replace_underscores` is removed from the key — it's a near-free
  string transform, applied post-hoc at retrieval so the slot can
  serve both with- and without-underscores requests.

`tag_image` is now: try cache → re-filter at effective values →
return. Miss → run model → write filtered at bucket floors. One
slot per `(image, model, bucketed_general, bucketed_character)`
instead of one slot per exact threshold tuple.

**Files:** `yadc/api/services/tagging.py` (added `bucket_threshold`
+ `_refilter`; restructured key + `tag_image` + `get_tag_result` +
`evict_tag_result`; dropped `replace_underscores` from
`TaggerResultKey` and from `evict_tag_result`'s signature),
`tests/taggers/test_service.py` (rewrote
`test_threshold_change_misses_cache` and the
`TestTaggerReadThroughCache` cases to use bucket-distinct overrides;
added `TestBucketThreshold`; updated `TestTaggerResultKey` for the
new field set), `docs/tagger-architecture.md`.
