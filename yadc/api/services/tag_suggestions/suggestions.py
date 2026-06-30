"""Tag-suggestion matcher — fast jump-list pre-check + fzf-style scoring.

A single linear pass over the catalog. For each row: an inverted char
index narrows to candidates containing every distinct query char, a
per-canonical jump-list subsequence pre-check rejects the rest cheaply,
then :func:`yadc.utils.fuzzy.fuzzy_score` ranks the survivors. Aliases are
scored alongside the canonical and each row contributes at most one hit
(its best score). A popularity multiplier (``log(count)``) and a
``_MAX_CANDIDATES`` cap (reached early because the catalog is loaded
sorted by count desc) bound the work for popular queries.

The catalog size (140k+) made running the full fzf walk per entry per
keystroke the bottleneck; the two pre-stages drop the majority of entries
before the expensive score pass. The matcher is a pure function
(``suggest(query, catalog, *, limit)``) so tests inject a small fixture
catalog.
"""

from __future__ import annotations

import asyncio
import bisect
import math

from yadc.utils.fuzzy import fuzzy_score
from yadc.utils.sorted_intersect import intersect_sorted_ints

from .catalog import (
    TagCatalog,
    normalize_tag_name,
)

# Per-tag char-index shape — defined here for type clarity, not a
# runtime import. Matches the per-tag value produced by
# :func:`yadc.api.services.tag_suggestions.catalog._normalize_with_char_index`.
_CharIndex = tuple[tuple[str, tuple[int, ...]], ...]


# Cooperatively yield every N iterations so a 141k-entry scan doesn't
# monopolize the event loop. At 5k an iteration is <10ms of pure-Python
# work — invisible to the user but lets other coroutines share time.
_YIELD_EVERY = 5_000

# Response-size ceiling. Endpoint clamps; belt-and-braces guard for direct
# callers (tests, future internal use).
_MAX_LIMIT = 50

# Cap the candidate pool. After this many hits the top-N (typically 10) is
# stable; continuing to scan is unlikely to change the answer. Reached on
# the highest-confidence hits first because the catalog is count-desc.
_MAX_CANDIDATES = 1_000


def _is_subsequence_via_index(query: str, char_index: _CharIndex) -> bool:
    """Greedy subsequence check using a per-tag char position index.

    For each query char, jump to the smallest position > last_pos in the
    matching ``positions`` tuple via ``bisect_right``. The ``char_index`` is
    sorted by char so the walk early-exits when the query char sorts past
    the current entry. Score-less — the matcher only pays for the fuzzy
    score on entries where this returns True.
    """
    pos = -1
    for c in query:
        next_pos: int | None = None
        for char, positions in char_index:
            if char == c:
                idx = bisect.bisect_right(positions, pos)
                if idx < len(positions):
                    next_pos = positions[idx]
                break
            if char > c:
                # Index is sorted by char — c is absent.
                break
        if next_pos is None:
            return False
        pos = next_pos
    return True


async def suggest(
    query: str,
    catalog: TagCatalog,
    *,
    limit: int = 10,
) -> list[tuple[str, str]]:
    """Return up to *limit* ``(tag, category)`` pairs matching *query*.

    Pure matcher: the caller (service / tests) supplies the parsed
    :class:`TagCatalog`. Returns the top *limit* by score, alpha-tiebreak.
    """
    raw = query.strip()
    if not raw:
        return []
    limit = max(1, min(limit, _MAX_LIMIT))

    q_normalized = normalize_tag_name(raw)
    if not q_normalized:
        return []

    # Stage 0: inverted-index candidate filter. A tag lacking any query char
    # (across canonical + aliases) can never subsequence-match. Each char's
    # tuple is appended in post-count-desc order at load, so the intersection
    # is popularity-sorted for free.
    char_index = catalog.char_index
    distinct_chars = set(q_normalized)
    lists: list[tuple[int, ...]] = []
    for c in distinct_chars:
        ids = char_index.get(c)
        if ids is None:
            return []
        lists.append(ids)
    candidates = intersect_sorted_ints(lists)

    # Each row contributes at most one hit — its best score across canonical
    # + aliases. Tracked separately so we sort once at the end.
    scored: list[tuple[str, str, float]] = []
    catalog_entries = catalog.entries
    catalog_normalized = catalog.normalized
    catalog_char_indices = catalog.char_indices
    catalog_counts = catalog.counts
    catalog_normalized_aliases = catalog.normalized_aliases

    n_done = 0
    for i in candidates:
        canonical, category, aliases = catalog_entries[i]

        # Stage 1: cheap subsequence pre-check via the canonical's jump list.
        # Entries with aliases still fall through regardless — the query might
        # match via an alias whose chars aren't in the canonical (e.g. typing
        # ``high_res`` returns the ``highres`` entry via its underscore alias;
        # the canonical's char index has no space, so the pre-check fails).
        canonical_pre = _is_subsequence_via_index(q_normalized, catalog_char_indices[i])
        if not canonical_pre and not aliases:
            n_done += 1
            if n_done % _YIELD_EVERY == 0:
                await asyncio.sleep(0)
            continue

        # Stage 2: fuzzy-score the canonical when the pre-check passed.
        score: float | None = None
        if canonical_pre:
            score = fuzzy_score(q_normalized, catalog_normalized[i])

        # Stage 3: aliases. Pick the best score across canonical + aliases.
        # Aliases are pre-normalized at load, so this is a straight fuzzy pass.
        norm_aliases = catalog_normalized_aliases[i]
        if norm_aliases:
            if score is None:
                for alias in norm_aliases:
                    a_score = fuzzy_score(q_normalized, alias)
                    if a_score is not None:
                        score = a_score
                        break
            else:
                for alias in norm_aliases:
                    a_score = fuzzy_score(q_normalized, alias)
                    if a_score is not None and a_score > score:
                        score = a_score

        if score is not None:
            # Popularity boost: widely-used tags surface above exact
            # substring matches in obscure entries. ``log(max(count, 1))``
            # keeps count 0/1 from crashing and gives them no boost.
            score = score * math.log(max(catalog_counts[i], 1))
            scored.append((canonical, category, score))
            if len(scored) >= _MAX_CANDIDATES:
                break

        n_done += 1
        if n_done % _YIELD_EVERY == 0:
            await asyncio.sleep(0)

    scored.sort(key=lambda triple: (-triple[2], triple[0]))
    return [(canonical, category) for canonical, category, _ in scored[:limit]]


__all__ = ["suggest"]
