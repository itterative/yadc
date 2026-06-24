"""Tag draft formatters — convert a ``TaggerResult`` to saveable text or a tags dict.

Draft formatters are a pluggable registry keyed by name (like the export
backends), so new text representations (e.g. weighted-confidence, JSON) can
be added without changing the wire format or the save path.

Two are shipped:

- ``comma`` — comma-separated tag list (the sd-scripts training caption; the
  useful default, since a saved ``tags`` draft doubles as a ready-to-train
  caption).
- ``structured`` — semi-structured by category, for feeding a caption
  refinement run.
- ``scored`` — like ``structured`` but each tag carries its confidence
  score, so a refinement LLM can weigh how much to trust each tag.

The extras-targeting helpers (``top_rating`` / ``extras_tags``) produce the
``[tags]`` sub-table shape: ``general``/``character`` as tag lists and
``rating`` as a single top-rating string (rating is categorical, not a set).
"""

from __future__ import annotations

from typing import Protocol

from yadc.taggers.base import TaggerResult


class TagDraftFormatter(Protocol):
    """Convert a (thresholded) :class:`TaggerResult` into draft text."""

    def __call__(self, result: TaggerResult) -> str: ...


_DRAFT_FORMATTERS: dict[str, TagDraftFormatter] = {}

# Canonical output order for category-grouped formatters: rating is
# metadata, character is the subject, general is descriptive detail.
# Character precedes general so the subject leads the description.
_CATEGORY_ORDER = ("rating", "character", "general")


def register_draft_formatter(name: str, fn: TagDraftFormatter) -> TagDraftFormatter:
    """Register a draft formatter under *name*, returning it."""
    _DRAFT_FORMATTERS[name] = fn
    return fn


def get_draft_formatter(name: str) -> TagDraftFormatter:
    """Return the registered formatter, raising ``KeyError`` if unknown."""
    try:
        return _DRAFT_FORMATTERS[name]
    except KeyError:
        raise KeyError(f"Unknown draft format '{name}'. Available: {sorted(_DRAFT_FORMATTERS)}") from None


def available_draft_formats() -> list[str]:
    """Return the sorted names of all registered draft formatters."""
    return sorted(_DRAFT_FORMATTERS)


def format_draft(name: str, result: TaggerResult) -> str:
    """Format *result* with the formatter registered under *name*."""
    return get_draft_formatter(name)(result)


def comma_formatter(result: TaggerResult) -> str:
    """Comma-separated tag list, in canonical category order.

    Rating tags are excluded — rating is categorical metadata, not a
    caption tag. This matches how sd-scripts training captions are
    built.
    """
    rating_tags = _rating_set(result)
    ordered = _ordered_tags(result, exclude=rating_tags)
    return ", ".join(ordered)


def structured_formatter(result: TaggerResult) -> str:
    """Semi-structured text grouped by category, one tag per section.

    Intended as input to a caption refinement run, where the model
    benefits from seeing which tags are rating / general / character.
    Sections appear in canonical order (character before general).
    """
    lines: list[str] = []
    rating = top_rating(result)
    if rating:
        lines.append(f"rating: {rating}")
    for cat in _CATEGORY_ORDER:
        if cat == "rating":
            continue
        cat_tags = result.categories.get(cat, [])
        if cat_tags:
            lines.append(f"{cat}: " + ", ".join(cat_tags))
    return "\n".join(lines)


def scored_formatter(result: TaggerResult) -> str:
    """Category-grouped text with per-tag confidence scores.

    Like :func:`structured_formatter` but each tag carries its score in
    parentheses, e.g. ``1girl (0.95)``. The confidence lets a
    caption-refinement LLM weigh how much to trust each tag. Sections
    appear in canonical order (character before general).
    """
    lines: list[str] = []
    rating = top_rating(result)
    if rating:
        lines.append(f"rating: {rating} ({result.tags.get(rating, 0.0):.2f})")
    for cat in _CATEGORY_ORDER:
        if cat == "rating":
            continue
        cat_tags = result.categories.get(cat, [])
        if cat_tags:
            parts = [f"{t} ({result.tags.get(t, 0.0):.2f})" for t in cat_tags]
            lines.append(f"{cat}: " + ", ".join(parts))
    return "\n".join(lines)


register_draft_formatter("comma", comma_formatter)
register_draft_formatter("structured", structured_formatter)
register_draft_formatter("scored", scored_formatter)


def top_rating(result: TaggerResult) -> str | None:
    """Return the highest-scoring rating tag, or ``None`` if none survived thresholding.

    Rating is categorical (e.g. general / sensitive / questionable /
    explicit), so only the top one is meaningful as stored metadata.
    """
    rating_tags = result.categories.get("rating")
    if not rating_tags:
        return None
    # Pick the highest-scoring; ties break by category order (label-file order).
    best_tag: str | None = None
    best_score = -1.0
    for tag in rating_tags:
        score = result.tags.get(tag, 0.0)
        if score > best_score:
            best_score = score
            best_tag = tag
    return best_tag


def extras_tags(result: TaggerResult) -> dict[str, object]:
    """Build the ``[tags]`` sub-table for the extras TOML sidecar.

    Shape: ``{"general": [...], "character": [...], "rating": "<top>"}``.
    ``rating`` is a single string (or omitted when no rating survived).
    """
    tags: dict[str, object] = {
        "general": list(result.categories.get("general", [])),
        "character": list(result.categories.get("character", [])),
    }
    rating = top_rating(result)
    if rating:
        tags["rating"] = rating
    return tags


# --- internal helpers -----------------------------------------------------


def _rating_set(result: TaggerResult) -> set[str]:
    return set(result.categories.get("rating", []))


def _ordered_tags(result: TaggerResult, *, exclude: set[str]) -> list[str]:
    """All tags in canonical category order, minus any in *exclude*.

    Canonical order is ``rating → character → general``, followed by any
    non-standard categories in their own order. Falls back to
    ``result.tags`` insertion order when the result has no category
    lists (uncategorized model output).
    """
    if result.categories:
        ordered: list[str] = []
        for cat in _CATEGORY_ORDER:
            ordered.extend(result.categories.get(cat, []))
        for cat, cat_tags in result.categories.items():
            if cat not in _CATEGORY_ORDER:
                ordered.extend(cat_tags)
    else:
        ordered = list(result.tags.keys())
    return [t for t in ordered if t not in exclude]
