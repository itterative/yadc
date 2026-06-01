"""Pure validation helpers for dataset uploads.

Functions in this module are stateless — they take inputs, return results,
and (where needed) use an injected logger. They have no dependency on
:class:`DatasetUploadService` and can be unit-tested in isolation.
"""

from logging import Logger
from pathlib import Path
from typing import BinaryIO

import tomlkit
from PIL import Image


def validate_image_stream(stream: BinaryIO, filename: str, logger: Logger | None = None) -> bool:
    """Validate an image stream with PIL. Returns True if valid.

    Falls back to the slower but more reliable ``load()`` check when
    ``verify()`` fails, since ``verify()`` is overly strict with some valid
    images (notably certain animated GIFs and progressive JPEGs).
    """
    try:
        with Image.open(stream) as img:
            img.verify()
    except Exception:
        stream.seek(0)
        try:
            with Image.open(stream) as img:
                img.load()
        except Exception as e:
            if logger is not None:
                logger.debug("Image validation failed: %s — %s", filename, e)
            return False
    return True


def validate_toml_stream(stream: BinaryIO) -> bool:
    """Validate a TOML stream. Returns True if it parses successfully."""
    content = stream.read()
    stream.seek(0)
    try:
        tomlkit.loads(content.decode("utf-8"))
        return True
    except Exception:
        return False


def unique_path(path: Path) -> Path:
    """Return a unique path by appending ``_N`` before the extension.

    If ``path`` doesn't exist, returns it unchanged. Otherwise tries
    ``path.stem + "_1" + path.suffix``, then ``"_2"``, etc.
    """
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    counter = 1
    while True:
        new_path = parent / f"{stem}_{counter}{suffix}"
        if not new_path.exists():
            return new_path
        counter += 1
