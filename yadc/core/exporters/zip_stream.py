"""Streaming zip assembly for export backends, built on ``stream-zip``.

Backends emit their entries as an iterator of :class:`ZipMember`; ``stream_zip_bytes``
encodes them into a streaming zip64 archive. stream-zip's buffered STORED mode reads
each member's bytes itself to compute its CRC32 and size, so memory is bounded to the
largest single member (one image or sidecar) rather than the whole dataset, and every
file is read only once. We never compress: images are incompressible, and STORED keeps
the archive size equal to the sum of its members, which is what the export UI's size
estimate relies on (zip overhead is a few hundred bytes of headers).

The file-reading seam (``_read_file_chunks``) is isolated here so that a future move to
async I/O only requires swapping these helpers and the single ``stream_zip`` call for
stream-zip's ``async_stream_zip``.
"""

import pathlib
import stat
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime

from stream_zip import NO_COMPRESSION_64, Method, stream_zip

# Match stream-zip's own default chunk size when reading files off disk.
_CHUNK_SIZE = 1 << 16  # 64 KiB

# Regular file, rw-r--r-- — used for generated members. On-disk members keep the
# file's real mode. stream-zip shifts this into the zip entry's external attributes.
_REGULAR_644 = stat.S_IFREG | 0o644


@dataclass(frozen=True)
class ZipMember:
    """A single entry to add to the streaming zip.

    ``content`` is an iterable of byte chunks consumed lazily as the zip is written;
    ``method`` is always buffered zip64 STORED (``NO_COMPRESSION_64``), so stream-zip
    derives each entry's CRC32 and size from the content itself.
    """

    name: str
    mtime: datetime
    mode: int
    method: Method
    content: Iterable[bytes]


def file_member(name: str, path: pathlib.Path, mtime: datetime, mode: int) -> ZipMember:
    """A STORED member streaming an on-disk file in chunks."""
    return ZipMember(name, mtime, mode, NO_COMPRESSION_64, _read_file_chunks(path))


def bytes_member(name: str, data: bytes, mtime: datetime) -> ZipMember:
    """A STORED member for an in-memory byte string (generated text, small sidecars)."""
    return ZipMember(name, mtime, _REGULAR_644, NO_COMPRESSION_64, _single_chunk(data))


def _read_file_chunks(path: pathlib.Path) -> Iterator[bytes]:
    with open(path, "rb") as f:
        while chunk := f.read(_CHUNK_SIZE):
            yield chunk


def _single_chunk(data: bytes) -> Iterator[bytes]:
    yield data


def stream_zip_bytes(members: Iterable[ZipMember]) -> Iterator[bytes]:
    """Encode *members* into a streaming zip64 archive, yielding bytes lazily."""
    files = ((m.name, m.mtime, m.mode, m.method, m.content) for m in members)
    yield from stream_zip(files)
