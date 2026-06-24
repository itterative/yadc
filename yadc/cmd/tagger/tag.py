"""Pure helpers backing the ``yadc tagger`` CLI.

Keeps label resolution and one-shot spawn/tag/stop out of the click
command layer so the CLI stays thin (see the CLI/cmd split convention).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from yadc.taggers.base import Tagger, TaggerResult
from yadc.taggers.server import TaggerServer


def resolve_label_path(model_path: str, label_path: str | None = None) -> str | None:
    """Pick the label file, auto-discovering ``selected_tags.csv`` next to the model.

    Returns the explicit ``label_path`` if given, else looks for
    ``<model_dir>/selected_tags.csv``. Returns ``None`` if nothing is found.
    """
    if label_path:
        return label_path
    candidate = Path(model_path).parent / "selected_tags.csv"
    return str(candidate) if candidate.exists() else None


def build_tagger_kwargs(label_path: str | None) -> dict[str, Any]:
    """Build :class:`OnnxTagger` kwargs from a (possibly None) label path."""
    if not label_path:
        return {}
    return {"label_path": label_path}


def tag_image(
    tagger_cls: type[Tagger],
    model_path: str,
    image_bytes: bytes,
    tagger_kwargs: dict[str, Any] | None = None,
) -> TaggerResult:
    """Tag a single image using a tagger subprocess (one-shot).

    Spawns the tagger, tags the image, and shuts down. Used by the
    ``yadc tagger tag`` one-shot CLI and short-lived background tasks.
    """
    server = TaggerServer(tagger_cls, model_path, tagger_kwargs or {})
    server.start()
    try:
        return server.tag(image_bytes)
    finally:
        server.stop()
