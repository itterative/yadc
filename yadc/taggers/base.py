"""Abstract ``Tagger`` base class and result types.

A tagger maps raw image bytes to a set of detected tags with confidence
scores. Taggers that categorize their output (e.g. SmilingWolf / WD
models with rating / general / character groups) populate
:attr:`TaggerResult.categories`; flat taggers leave it empty.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TaggerResult:
    """Result of a tagging operation.

    Attributes:
        tags: All detected tags and their scores. Every tag in
            ``categories``'s lists must also appear here with its score.
        categories: Map of category name to the ordered list of tag
            names from ``tags`` that belong to it. Empty for taggers
            that don't categorize (flat output).
    """

    tags: dict[str, float]
    categories: dict[str, list[str]] = field(default_factory=dict)


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
