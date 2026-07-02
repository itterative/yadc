"""Shared utilities for export backends."""

import pathlib

from ..dataset import DatasetImage


def relative_arc_name(path: pathlib.Path, base_dir: pathlib.Path) -> str:
    """Relative path for *path* inside a zip, rooted at *base_dir*.

    Falls back to the bare filename when *path* is not under *base_dir*
    (e.g. external / imported datasets whose files live elsewhere).
    """
    try:
        return str(path.relative_to(base_dir))
    except ValueError:
        return path.name


def read_caption_source(
    image: DatasetImage,
    source: str,
    drafts: tuple[str, ...] = (),
    *,
    delimiter: str = "\n",
) -> str:
    """Read caption text from the chosen source, optionally appending drafts.

    Args:
        image: The dataset image.
        source: ``'caption'`` reads the caption file; ``'draft'`` reads drafts only.
                When source is ``'draft'``, the first element of *drafts* is the
                primary draft (required). Remaining elements are appended.
        drafts: Draft names to include. Order matters — they are joined with
                *delimiter* after the primary source.
        delimiter: String placed between every joined source/draft segment.
                Defaults to ``"\\n"``, matching the prior hardcoded behavior.
                Common presets: ``". "`` (sentences), ``"\\n\\n"`` (paragraphs).

    Returns:
        The combined text, stripped of whitespace.

    Raises:
        ValueError: If source is not ``'caption'`` or ``'draft'``, or if source
                    is ``'draft'`` but *drafts* is empty.
        FileNotFoundError: If a source file does not exist.
    """

    if source == "draft":
        if not drafts:
            raise ValueError('drafts is required when source is "draft"')
        parts: list[str] = []
        for name in drafts:
            path = image.draft_path(name)
            if not path.exists():
                raise FileNotFoundError(f"Draft not found: {path}")
            text = path.read_text().strip()
            if text:
                parts.append(text)
    elif source == "caption":
        text = image.read_caption()
        if not text and not image.caption_path.exists():
            raise FileNotFoundError(f"Caption not found: {image.caption_path}")
        parts = [text] if text else []
        for name in drafts:
            path = image.draft_path(name)
            if not path.exists():
                raise FileNotFoundError(f"Draft not found: {path}")
            draft_text = path.read_text().strip()
            if draft_text:
                parts.append(draft_text)
    else:
        raise ValueError(f'source must be "caption" or "draft", got {source!r}')

    return delimiter.join(parts)
