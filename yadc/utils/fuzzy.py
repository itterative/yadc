"""Fuzzy subsequence matching — fzf-style ranking.

Pure-function utility, no domain knowledge of tags. Used by
:mod:`yadc.api.services.tag_suggestions` for char-subsequence scoring.

The walk matches fzf's published algorithm: walk *target* left-to-right
adding per-match bonuses (base + position/streak) and advancing the query
cursor; non-matches add a gap penalty (``SCORE_GAP_START`` once per
match→gap transition, then ``SCORE_GAP_EXT`` per gap char). If not every
query char matched, the query isn't a subsequence → return ``None``.

Position bonuses reward leading chars (``SCORE_MATCH_START``), matches
after a separator (``SCORE_MATCH_BOUNDARY`` — typing the first char of a
word in a multi-word tag is a strong signal), and CamelCase boundaries.
Streak bonuses (``SCORE_CONSECUTIVE * streak``) push contiguous matches
above scattered-char matches with the same subsequence.

Why this and not a trie: the earlier trie-with-alphabet-pruning spent
~200ms per query at 141k entries; this single linear pass runs in roughly
the same time with a much simpler implementation. The per-keystroke cost
at 141k is unavoidable for a pure-Python generic matcher — the frontend's
250ms debounce + LRU hide it in practice.
"""

from __future__ import annotations

# Per-match base score.
SCORE_MATCH = 16

# Gap penalties: ``SCORE_GAP_START`` fires once when leaving a streak;
# each subsequent gap char adds ``SCORE_GAP_EXT``. Tighter (more negative)
# penalties push contiguous matches higher and de-prioritise scattered
# char matches.
SCORE_GAP_START = -6
SCORE_GAP_EXT = -1

# Position bonuses added to a match. ``SCORE_MATCH_BOUNDARY`` (after a
# non-alphanumeric char) is the most useful for multi-word tags;
# ``SCORE_MATCH_CAMEL`` is mostly inert for the lowercased catalog but
# kept for completeness.
SCORE_MATCH_START = 10
SCORE_MATCH_BOUNDARY = 8
SCORE_MATCH_CAMEL = 6

# Streak bonus: each match adds ``SCORE_CONSECUTIVE * streak`` where
# ``streak`` is the consecutive-match count at and including this one.
SCORE_CONSECUTIVE = 12


def fuzzy_score(query: str, target: str) -> int | None:
    """Compute fzf-style match score for *query* against *target*.

    Case-insensitive (both inputs lowercased). Callers that want
    separator-normalized matching should pre-normalize (via
    :func:`yadc.api.services.tag_suggestions.normalize_tag_name`); this
    function is intentionally separator-agnostic.

    Returns ``0`` for an empty query, the integer score if *query* is a
    char-subsequence of *target* (higher = better), ``None`` if not.
    """
    if not query:
        return 0

    q = query.lower()
    t = target.lower()
    m = len(q)
    n = len(t)

    pi = 0
    ti = 0
    score = 0
    consecutive = 0

    while pi < m and ti < n:
        if t[ti] == q[pi]:
            match_bonus = SCORE_MATCH
            if ti == 0:
                match_bonus += SCORE_MATCH_START
            else:
                prev = t[ti - 1]
                if not prev.isalnum():
                    match_bonus += SCORE_MATCH_BOUNDARY
                elif prev.islower() and t[ti].isupper():
                    match_bonus += SCORE_MATCH_CAMEL
            if consecutive > 0:
                match_bonus += SCORE_CONSECUTIVE * consecutive
            score += match_bonus
            consecutive += 1
            pi += 1
            ti += 1
        else:
            # Gap: SCORE_GAP_START once per streak→gap transition, then
            # SCORE_GAP_EXT for every gap char (including the first).
            if consecutive > 0:
                score += SCORE_GAP_START
                consecutive = 0
            elif score > 0:
                score += SCORE_GAP_EXT
            ti += 1

    if pi < m:
        return None

    return score


def is_fuzzy_match(query: str, target: str) -> bool:
    """True iff *query* is a char-subsequence of *target* (case-insensitive).

    Equivalent to ``fuzzy_score(query, target) is not None`` but reads
    better in callers that only care about membership.
    """
    return fuzzy_score(query, target) is not None


__all__ = [
    "fuzzy_score",
    "is_fuzzy_match",
    "SCORE_MATCH",
    "SCORE_GAP_START",
    "SCORE_GAP_EXT",
    "SCORE_MATCH_START",
    "SCORE_MATCH_BOUNDARY",
    "SCORE_MATCH_CAMEL",
    "SCORE_CONSECUTIVE",
]
