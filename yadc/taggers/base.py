"""Abstract ``Tagger`` base class and result types.

A tagger maps raw image bytes to a set of detected tags with confidence
scores. Taggers that categorize their output (e.g. SmilingWolf / WD
models with rating / general / character groups) populate
:attr:`TaggerResult.categories`; flat taggers leave it empty.
"""

from __future__ import annotations

import abc
import sys
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TagCustomizations:
    """User edits layered on top of a tag result for one image.

    The model output (``tags`` / ``categories``) is the tagger's truth;
    this carries the interactive selection the user made in the Tags
    tab. ``disabled`` is the subset of model-result tags the user
    turned off (stored rather than ``enabled`` so a cache-bucket change
    — different model / thresholds — naturally defaults the new result
    to all-on, and only the explicitly-turned-off tags carry over).
    ``custom_tags`` mirrors :attr:`TaggerResult.categories` (category →
    names) so each addition remembers where it was placed. Both are
    optional (``None`` for raw model runs) and persist on the cached
    :class:`TaggerResult` so navigating away and back restores the
    selection.
    """

    disabled: list[str]
    custom_tags: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class TaggerResult:
    """Result of a tagging operation.

    Attributes:
        tags: All detected tags and their scores. Every tag in
            ``categories``'s lists must also appear here with its score.
        categories: Map of category name to the ordered list of tag
            names from ``tags`` that belong to it. Empty for taggers
            that don't categorize (flat output).
        customizations: Optional user-edited selection layered on top
            of the model output (kept tags + custom tags). ``None`` for
            raw model runs; set by the interactive Tags tab. Carried
            through the cache so navigation preserves the selection.
    """

    tags: dict[str, float]
    categories: dict[str, list[str]] = field(default_factory=dict)
    customizations: TagCustomizations | None = None


class Tagger(abc.ABC):
    """Abstract base class for image taggers."""

    @abc.abstractmethod
    def load_model(self, model_path: str, **kwargs: Any) -> None:
        """Load the tagging model from *model_path*.

        Args:
            model_path: Path to the model file (e.g. ONNX).
            **kwargs: Implementation-specific options (e.g. provider, session options).
        """

    @abc.abstractmethod
    def unload_model(self) -> None:
        """Unload the model and free resources."""

    @abc.abstractmethod
    def predict(self, image_bytes: bytes) -> TaggerResult:
        """Generate tags for an image.

        Args:
            image_bytes: Raw image bytes (JPEG/PNG).

        Returns:
            Categorized tag scores. Callers should apply any
            caller-defined thresholds; taggers expose raw scores.
        """


def _deep_size(obj: object, seen: set[int]) -> int:
    """Sum ``sys.getsizeof`` for *obj* and all transitively-reachable contents.

    Memoizes visited ``id()``s so a string referenced by both the
    ``tags`` dict (as a key) and a ``categories`` list (as an item) is
    counted once, not twice. Counts each parent's reference overhead
    separately (the dict / list container itself).
    """
    if id(obj) in seen:
        return 0
    seen.add(id(obj))
    size = sys.getsizeof(obj)
    if isinstance(obj, dict):
        for k, v in obj.items():
            size += _deep_size(k, seen) + _deep_size(v, seen)  # pyright: ignore[reportUnknownArgumentType]
    elif isinstance(obj, (list, tuple, set, frozenset)):
        for item in obj:
            size += _deep_size(item, seen)  # pyright: ignore[reportUnknownArgumentType]
    return size


def tamer_result_size(result: TaggerResult) -> int:
    """Approximate memory occupied by a ``TaggerResult`` and all its contents.

    Walks ``result.tags`` (dict of label → score) and
    ``result.categories`` (dict of category → label list) recursively,
    using :func:`sys.getsizeof` at each node. Used as the size function
    for the bytes-bounded LRU cache, so the cache can evict when its
    summed value sizes exceed a byte budget rather than a fixed item
    count.

    Recursion depth is bounded: the structure is two dict layers deep
    at most (tags and categories are dicts; categories' values are
    flat lists of strings). Cycles are broken by ``id()`` memoization.
    """
    seen: set[int] = set()
    total = _deep_size(result, seen) + _deep_size(result.tags, seen) + _deep_size(result.categories, seen)
    if result.customizations is not None:
        total += _deep_size(result.customizations, seen)
        total += _deep_size(result.customizations.disabled, seen)
        total += _deep_size(result.customizations.custom_tags, seen)
    return total
