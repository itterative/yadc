"""Shared utilities for export backends."""

from ..dataset import DatasetImage


def read_caption_source(
    image: DatasetImage,
    source: str,
    draft_name: str = '',
) -> str:
    """Read caption text from the chosen source.

    Args:
        image: The dataset image.
        source: ``'caption'`` reads the caption file; ``'draft'`` reads a named draft.
        draft_name: Required when source=``'draft'``.

    Returns:
        The caption text, stripped of whitespace.

    Raises:
        ValueError: If source is not ``'caption'`` or ``'draft'``, or if source=``'draft'``
                    but draft_name is empty.
        FileNotFoundError: If the source file does not exist.
    """

    if source == 'draft':
        if not draft_name:
            raise ValueError('draft_name is required when source is "draft"')
        path = image.draft_path(draft_name)
        if not path.exists():
            raise FileNotFoundError(f'Draft not found: {path}')
        return path.read_text().strip()

    if source == 'caption':
        text = image.read_caption()
        if not text and not image.caption_path.exists():
            raise FileNotFoundError(f'Caption not found: {image.caption_path}')
        return text

    raise ValueError(f'source must be "caption" or "draft", got {source!r}')
