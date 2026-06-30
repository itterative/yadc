"""Intersection of ascending-sorted integer sequences — hybrid merge.

Used by :mod:`yadc.api.services.tag_suggestions` to narrow the inverted
char index (char → ascending tag indexes) to candidates that contain
*every* distinct query char. The lists are appended in entry order at
catalog load (post-count desc), so they're already sorted ascending —
and already popularity-sorted, which the matcher relies on for its
early-stop cap.

Two strategies, picked per call from the input shape:

- **Set intersection** (``set.intersection``, C-level) when there are few
  distinct query chars and the rarest one still indexes many tags — the
  per-element Python loop of a galloping merge is the bottleneck there, so
  building sets and intersecting in C wins ~30–50%.
- **Galloping fold** otherwise: pairwise fold, shortest list first, each
  pair merged by ``bisect.bisect_left`` on the smaller list. O(len(small)
  * log(len(large))) per pair, avoids up-front set construction, and
  collapses fast when the rarest char is genuinely rare.

The crossover is well-separated by (char count, smallest list size), so a
cheap heuristic picks the right one without measuring. A plain two-pointer
walk was also measured and rejected: its per-iteration Python overhead
loses to gallop's in-C ``bisect`` whenever the lists are large.
"""

from __future__ import annotations

import bisect
from collections.abc import Sequence

# Switch to set intersection when there are at most this many distinct
# query chars AND the rarest char's list is at least this long. Tuned
# empirically against the 141k-entry NoobAIXL catalog: below these
# thresholds gallop collapses fast (tiny driver), above them the C-level
# set ops beat the per-element Python bisect loop by ~30%.
_SET_MAX_LISTS = 4
_SET_MIN_LIST_LEN = 5_000


def _intersect_gallop(a: Sequence[int], b: Sequence[int]) -> list[int]:
    """Intersect two ascending-sorted int sequences via galloping.

    The smaller sequence drives: each element is binary-searched for in
    the larger one, starting past the previous hit (valid because both
    inputs ascend). Duplicates aren't expected (inputs are position
    sets), so no dedup is needed.
    """
    if len(a) > len(b):
        a, b = b, a
    out: list[int] = []
    lo = 0
    lb = len(b)
    for va in a:
        idx = bisect.bisect_left(b, va, lo)
        if idx < lb and b[idx] == va:
            out.append(va)
        # Advance past va either way: b is ascending, so the next driver
        # element (> va) can't occur before idx.
        lo = idx
    return out


def _intersect_gallop_fold(lists: list[Sequence[int]]) -> list[int]:
    """Pairwise galloping fold, shortest list first.

    Folding shortest-first collapses the running intersection after the
    first merge — the dominant cost is the first (rarest char) pair.
    """
    ordered = sorted(lists, key=len)
    result: list[int] = list(ordered[0])
    for nxt in ordered[1:]:
        if not result:
            return []
        if not nxt:
            return []
        result = _intersect_gallop(result, nxt)
    return result


def _intersect_set(lists: list[Sequence[int]]) -> list[int]:
    """C-level set intersection, result restored to ascending order.

    The matcher relies on popularity-sorted candidates for its early-stop
    cap, so the unordered set result is re-sorted here.
    """
    sets = sorted((set(x) for x in lists), key=len)
    return sorted(sets[0].intersection(*sets[1:]))


def intersect_sorted_ints(lists: Sequence[Sequence[int]]) -> list[int]:
    """Intersect a collection of ascending-sorted integer sequences.

    Returns the values present in **every** input, ascending-sorted. Empty
    input or any empty member yields ``[]``. Strategy is picked from the
    input shape — see the module docstring.
    """
    if not lists:
        return []
    if len(lists) == 1:
        return list(lists[0])
    # Drive set intersection off the smallest list's size: a rare char
    # (e.g. ``1``, ~200 tags) makes gallop collapse almost instantly, so
    # even a 5-char query is better off galloping. The char-count guard
    # caps the total set-construction cost for multi-char queries.
    materialized = list(lists)
    smallest_len = min(len(x) for x in materialized)
    if len(materialized) <= _SET_MAX_LISTS and smallest_len >= _SET_MIN_LIST_LEN:
        return _intersect_set(materialized)
    return _intersect_gallop_fold(materialized)


__all__ = ["intersect_sorted_ints"]
