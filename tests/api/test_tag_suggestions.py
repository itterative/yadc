"""Tests for the tag-suggestion matcher.

The matcher is async (it yields to the event loop every
``_YIELD_EVERY`` iterations on real-world 140k+ catalogs). All tests
inject a small explicit fixture catalog so we never hit the loader
or the network.

Two-stage algorithm: cheap per-tag char→positions pre-check (built
once at load time) filters the bulk of the catalog before the
fzf-style fuzzy score in :mod:`yadc.utils.fuzzy` is invoked. The
cap at :data:`tag_suggestions._MAX_CANDIDATES` keeps the candidate
pool bounded; the catalog is sorted by post count desc so the cap
is reached on the highest-popularity hits.

Returning ``(name, category)`` tuples means the frontend can show
category badges (``rating`` / ``general`` / ``character``) without
hitting a separate endpoint.
"""

from __future__ import annotations

import pytest

from yadc.api.services.tag_suggestions import TagCatalog, suggest
from yadc.api.services.tag_suggestions.catalog import _build_catalog

# A small fixture catalog that covers prefix-match, subsequence,
# alias-match, multi-token, and category-coverage cases — enough
# to exercise every branch without dragging in the 140k real catalog.
_FIXTURE_ENTRIES: list[tuple[str, str, tuple[str, ...]]] = [
    # Long hair family — prefix & substring ranking.
    ("long_hair", "general", ("longhair",)),
    ("long_sleeves", "general", ()),
    ("long_dress", "general", ()),
    # Hair family — substring-only hits via canonical.
    ("blonde_hair", "general", ()),
    ("black_hair", "general", ()),
    ("brown_hair", "general", ()),
    # Alias-only path with multiple variants.
    ("highres", "meta", ("high_res", "high_resolution", "hires")),
    # /-prefixed alias (danbooru forum shorthand).
    ("large_breasts", "general", ("big_breasts", "large_boobs", "/lb")),
    # Character category.
    ("hatsune_miku", "character", ()),
    ("rem_(re:zero)", "character", ("rem",)),
    # Multi-word canonical that exercises the fuzzy tokenizer.
    ("castorice_(honkai:_star_rail)", "character", ("castorice",)),
    # Rating category.
    ("sensitive", "rating", ()),
]


@pytest.fixture
def catalog() -> TagCatalog:
    """A real ``TagCatalog`` built from the fixture via the
    catalog service's internal builder.

    Mirrors what ``tag_catalog._do_load`` produces: each canonical
    is paired with its normalized form and per-char position index
    (see :func:`yadc.api.services.tag_catalog._build_catalog`).
    """
    return _build_catalog(_FIXTURE_ENTRIES)


# ---------- Query handling ----------


@pytest.mark.asyncio
async def test_empty_query_returns_empty(catalog: TagCatalog) -> None:
    assert await suggest("", catalog=catalog) == []
    assert await suggest("   ", catalog=catalog) == []
    assert await suggest("\t\n", catalog=catalog) == []


@pytest.mark.asyncio
async def test_query_with_only_separators_returns_empty(catalog: TagCatalog) -> None:
    """A query of pure separators normalizes to ``""``; treat as empty."""
    assert await suggest("___", catalog=catalog) == []


# ---------- Trie walk (single-token queries) ----------


@pytest.mark.asyncio
async def test_prefix_match_returns_canonicals_starting_with_query(catalog: TagCatalog) -> None:
    """Three canonicals start with ``"long"``; alpha-sorted."""
    results = await suggest("long", catalog=catalog)
    assert [name for name, _ in results] == ["long_dress", "long_hair", "long_sleeves"]


@pytest.mark.asyncio
async def test_trie_normalization_makes_underscore_and_space_equivalent(
    catalog: TagCatalog,
) -> None:
    """Trie stores normalized keys; ``long_hair`` (underscore) and
    ``long hair`` (space) hit the same canonical."""
    by_underscore = await suggest("long_hair", catalog=catalog)
    by_space = await suggest("long hair", catalog=catalog)
    # Both forms surface ``long_hair`` first.
    assert by_underscore[0] == ("long_hair", "general")
    assert by_space[0] == ("long_hair", "general")


@pytest.mark.asyncio
async def test_single_subsequence_char_two_returns_all_ha_matches(catalog: TagCatalog) -> None:
    """``ha`` subsequence-matches every canonical that contains
    ``h`` followed by ``a`` in order. The fixture's counts are all
    1 (the default), so the popularity boost is disabled and the
    final ranking is the alpha tiebreak after the raw score — all
    scores are non-negative so we sort by canonical name. ``highres``
    doesn't match because its canonical has no ``a`` at all.
    """
    results = await suggest("ha", catalog=catalog)
    assert [name for name, _ in results] == [
        "black_hair",
        "blonde_hair",
        "brown_hair",
        "castorice_(honkai:_star_rail)",
        "hatsune_miku",
        "long_hair",
    ]


# ---------- Tier-2 fuzzy subsequence behavior ----------


@pytest.mark.asyncio
async def test_tier2_out_of_order_chars_within_token(catalog: TagCatalog) -> None:
    """``lng`` subsequence-matches ``long_hair`` (l, n, g in order)."""
    results = await suggest("lng", catalog=catalog)
    assert ("long_hair", "general") in results


@pytest.mark.asyncio
async def test_tier2_out_of_order_chars_alt_forms(catalog: TagCatalog) -> None:
    """``lhr`` subsequence-matches ``long_hair`` (l, h, r in order)."""
    results = await suggest("lhr", catalog=catalog)
    assert ("long_hair", "general") in results


@pytest.mark.asyncio
async def test_tier2_multi_token_in_order(catalog: TagCatalog) -> None:
    """``ca honkai`` matches the multi-word castorice tag because:
    - ``ca`` subsequence-matches ``castorice``
    - ``honkai`` exact-matches ``honkai`` (after ca's match position)
    Both tokens consume the target's chars left-to-right.
    """
    results = await suggest("ca honkai", catalog=catalog)
    assert ("castorice_(honkai:_star_rail)", "character") in results


@pytest.mark.asyncio
async def test_tier2_multi_token_strict_in_order(catalog: TagCatalog) -> None:
    """``honkai ca`` (tokens reversed) does NOT match. ``honkai``
    matches later in the canonical, leaving no chars for ``ca`` to
    subsequence-match.
    """
    results = await suggest("honkai ca", catalog=catalog)
    assert ("castorice_(honkai:_star_rail)", "character") not in results


@pytest.mark.asyncio
async def test_tier2_substring_still_works(catalog: TagCatalog) -> None:
    """Pure substring matching (single token, all chars in order,
    contiguous in the target) is a special case of subsequence."""
    # ``hair`` subsequence-matches all four hair tags (h, a, i, r).
    results = await suggest("hair", catalog=catalog)
    hair_names = {name for name, _ in results}
    assert {"long_hair", "black_hair", "blonde_hair", "brown_hair"} <= hair_names
    # Unrelated tags don't appear.
    assert ("hatsune_miku", "character") not in results


@pytest.mark.asyncio
async def test_tier2_matches_alias_subsequences(catalog: TagCatalog) -> None:
    """An alias that subsequence-matches the query surfaces the canonical."""
    # ``highres`` has alias ``hires``. Query ``hires`` matches the
    # alias as a subsequence of the canonical key; canonical surfaces.
    results = await suggest("hires", catalog=catalog)
    assert ("highres", "meta") in results


# ---------- Alias matching ----------


@pytest.mark.asyncio
async def test_alias_substring_match_returns_canonical(catalog: TagCatalog) -> None:
    """Typing an alias substring returns the canonical (any tier)."""
    results = await suggest("high_res", catalog=catalog)
    assert ("highres", "meta") in results
    # The alias itself is never surfaced.
    assert all(name != "high_res" for name, _ in results)


@pytest.mark.asyncio
async def test_alias_prefix_match_returns_canonical(catalog: TagCatalog) -> None:
    """``/lb`` (forum shorthand) returns ``large_breasts`` via the alias."""
    results = await suggest("/lb", catalog=catalog)
    assert ("large_breasts", "general") in results


@pytest.mark.asyncio
async def test_alias_match_only_path(catalog: TagCatalog) -> None:
    """A query that only matches via an alias surfaces the canonical."""
    results = await suggest("/lb", catalog=catalog)
    assert results == [("large_breasts", "general")]


# ---------- No match ----------


@pytest.mark.asyncio
async def test_no_match_returns_empty(catalog: TagCatalog) -> None:
    assert await suggest("zzzzzzz", catalog=catalog) == []


# ---------- Limit clamping ----------


@pytest.mark.asyncio
async def test_limit_caps_response(catalog: TagCatalog) -> None:
    results = await suggest("hair", limit=2, catalog=catalog)
    assert len(results) <= 2


@pytest.mark.asyncio
async def test_limit_ceiling_clamped(catalog: TagCatalog) -> None:
    """Asking for more than max returns the full relevant set, not 50."""
    results = await suggest("hair", limit=10_000, catalog=catalog)
    assert len(results) <= len(catalog.entries)


@pytest.mark.asyncio
async def test_limit_floor_clamped(catalog: TagCatalog) -> None:
    """Negative / zero limits don't produce 0 results when matches exist."""
    assert len(await suggest("long", limit=0, catalog=catalog)) >= 1
    assert len(await suggest("long", limit=-5, catalog=catalog)) >= 1


# ---------- Case + separator normalization ----------


@pytest.mark.asyncio
async def test_case_insensitive(catalog: TagCatalog) -> None:
    assert await suggest("LONG", catalog=catalog) == await suggest("long", catalog=catalog)
    assert await suggest("Long", catalog=catalog) == await suggest("long", catalog=catalog)
    # Aliases are case-insensitive too.
    assert await suggest("/LB", catalog=catalog) == await suggest("/lb", catalog=catalog)


# ---------- Category coverage ----------


@pytest.mark.asyncio
async def test_categories_carried_through(catalog: TagCatalog) -> None:
    """Each hit carries its catalog category, not a uniform default."""
    results = await suggest("long", catalog=catalog)
    assert all(category == "general" for _, category in results)


@pytest.mark.asyncio
async def test_different_categories_for_same_query(catalog: TagCatalog) -> None:
    """A query that hits multiple categories keeps each row's own
    category."""
    results = await suggest("rem_(re", catalog=catalog)
    assert ("rem_(re:zero)", "character") in results
