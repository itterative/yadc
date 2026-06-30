"""Tagger subsystem — image tagging via ONNX Runtime models.

Taggers run in a separate process and communicate via ``multiprocessing.Queue``.
"""

from yadc.taggers.base import Tagger, TaggerResult
from yadc.taggers.formatters import extras_tags, format_draft, top_rating
from yadc.taggers.onnx import OnnxTagger, apply_thresholds
from yadc.taggers.postprocessing import replace_underscore_for_tag, replace_underscores

__all__ = [
    "Tagger",
    "TaggerResult",
    "OnnxTagger",
    "apply_thresholds",
    "replace_underscore_for_tag",
    "replace_underscores",
    "extras_tags",
    "format_draft",
    "top_rating",
]
