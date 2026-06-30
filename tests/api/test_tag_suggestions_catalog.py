"""Integration-flavored tests for the tag-suggestion matcher.

These tests pull the real BetaDoggo NoobAIXL catalog (~141k tags)
from the local cache and exercise the matcher end-to-end against
production-scale data. Unlike :mod:`test_tag_suggestions` — which
uses a 12-row fixture to lock the matcher's contract — these tests
reveal how the matcher behaves at the scale users actually hit:
multi-word character lookups, abbreviation shorthand, and short
queries against 141k popular tags.

Skipped when the catalog isn't cached locally. First-run users
don't get punished with a network download in their test suite;
anyone who's used the suggest endpoint (or run
:func:`yadc.api.services.tag_catalog.download_catalog` directly)
already has the file and gets these tests for free.
"""

from __future__ import annotations

import asyncio
import logging
import time

import pytest

from yadc.api.services.tag_suggestions import (
    DEFAULT_VARIANT,
    TagCatalog,
    is_cached,
    load_catalog,
    suggest,
)

logger = logging.getLogger(__name__)

# Computed once at import so the skip reason is decided up front
# (not at test runtime, which is what ``skipif`` evaluates).
_REAL_CATALOG_CACHED = is_cached(DEFAULT_VARIANT)


@pytest.fixture(scope="module")
def real_catalog() -> TagCatalog:
    """Load the real NoobAIXL catalog once per module.

    Subsequent tests reuse the in-memory bundle via the module
    scope — no re-parse per test. The fixture drives the async loader
    (:func:`load_catalog`) via ``asyncio.run`` so pytest-asyncio's
    per-test loop management doesn't conflict with the load. The
    resulting :class:`TagCatalog` is plain data (frozen dataclass of
    tuples) and is safe to read from any event loop.
    """
    return asyncio.run(load_catalog(DEFAULT_VARIANT))


async def _timed_suggest(
    query: str,
    catalog: TagCatalog,
    *,
    limit: int = 10,
) -> list[tuple[str, str]]:
    """Run :func:`suggest` and log the elapsed time.

    Surfaces the matcher's perf at production scale so the test
    output doubles as a perf readout. Run with
    ``pytest --log-cli-level=INFO`` to see the lines live.
    """
    t0 = time.perf_counter()
    results = await suggest(query, catalog=catalog, limit=limit)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "suggest query took %.1fms [query=%r, hits=%d, limit=%d]",
        elapsed_ms,
        query,
        len(results),
        limit,
    )
    return results


# ---------- Multi-token queries ----------


@pytest.mark.skipif(not _REAL_CATALOG_CACHED, reason="real NoobAIXL catalog not in local cache")
@pytest.mark.asyncio
async def test_cast_honkai_includes_castorice(real_catalog: TagCatalog) -> None:
    """``cast honkai`` matches the multi-word character tag
    ``castorice_(honkai:_star_rail)`` via subsequence across
    both words — ``cast`` consumes the first word and ``honkai``
    consumes the second.
    """
    results = await _timed_suggest("cast honkai", real_catalog)
    assert "castorice_(honkai:_star_rail)" in [name for name, _ in results]


@pytest.mark.skipif(not _REAL_CATALOG_CACHED, reason="real NoobAIXL catalog not in local cache")
@pytest.mark.asyncio
async def test_fern_frieren_includes_fern_frieren(real_catalog: TagCatalog) -> None:
    """``fern frieren`` exact-matches the canonical after
    normalization (separators collapsed to spaces). It's the
    highest-scoring hit because every query char consumes a
    canonical char contiguously.
    """
    results = await _timed_suggest("fern frieren", real_catalog)
    assert "fern_(frieren)" in [name for name, _ in results]


# ---------- Short queries ----------


@pytest.mark.skipif(not _REAL_CATALOG_CACHED, reason="real NoobAIXL catalog not in local cache")
@pytest.mark.asyncio
async def test_girl_includes_1girl(real_catalog: TagCatalog) -> None:
    """``girl`` should include ``1girl`` in the default top-N.

    ``1girl`` wins the popularity boost — it has 6.1M posts vs
    ~6K for ``girl_on_top`` — so its score dominates even though
    the raw fuzzy score is lower (``g`` at position 1 vs position
    0 for the ``girl_*`` prefix matches). The
    ``fuzzy_score * log(count)`` multiplier is what surfaces
    popular tags above exact-prefix matches in obscure entries.
    """
    results = await _timed_suggest("girl", real_catalog)
    assert "1girl" in [name for name, _ in results]


# ---------- Abbreviation / shorthand ----------


@pytest.mark.skipif(not _REAL_CATALOG_CACHED, reason="real NoobAIXL catalog not in local cache")
@pytest.mark.asyncio
async def test_simple_bg_includes_simple_background(real_catalog: TagCatalog) -> None:
    """``simple bg`` matches ``simple_background`` via subsequence:
    ``simple`` consumes the first word, then ``bg`` jumps to
    ``background`` (``b`` at position 7, ``g`` at position 16).

    ``simple_background`` is also overwhelmingly the only result
    here — it's a unique-target query where the popular synonyms
    (``simple`` on its own, ``bg`` shorthand) all score lower.
    """
    results = await _timed_suggest("simple bg", real_catalog)
    assert "simple_background" in [name for name, _ in results]
