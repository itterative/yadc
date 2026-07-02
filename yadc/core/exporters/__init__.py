"""Export backend registry.

Each backend is a module under ``yadc.core.exporters`` that exposes:

- ``BACKEND`` — a dataclass (frozen) with ``name``, ``description``, ``formats``.
- ``run(images, *, fmt, source, drafts, output, append, caption_extension) -> int``
- ``iter_zip_members(images, *, fmt, source, drafts, caption_extension, include_images, base_dir) -> tuple[Iterator[ZipMember], int]``

To register a new backend, import its module and add it to ``_BACKENDS`` below.
"""

import pathlib
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol

from ..dataset import DatasetImage
from . import sd_scripts as _sd_scripts
from . import yadc as _yadc
from .zip_stream import ZipMember, stream_zip_bytes


class _RunFn(Protocol):
    def __call__(
        self,
        images: list[DatasetImage],
        *,
        fmt: str,
        source: str,
        drafts: tuple[str, ...],
        output: pathlib.Path | None,
        append: bool,
        caption_extension: str,
        delimiter: str,
    ) -> int: ...


class _IterZipMembersFn(Protocol):
    def __call__(
        self,
        images: list[DatasetImage],
        *,
        fmt: str,
        source: str,
        drafts: tuple[str, ...],
        caption_extension: str,
        include_images: bool,
        base_dir: pathlib.Path | None,
        delimiter: str,
    ) -> tuple[Iterator[ZipMember], int]: ...


@dataclass(frozen=True)
class _BackendDescriptor:
    run: _RunFn
    iter_zip_members: _IterZipMembersFn
    name: str
    description: str
    formats: tuple[str, ...]
    zip_only: bool = False


_BACKENDS: dict[str, _BackendDescriptor] = {
    "sd-scripts": _BackendDescriptor(
        run=_sd_scripts.run,
        iter_zip_members=_sd_scripts.iter_zip_members,
        name=_sd_scripts.BACKEND.name,
        description=_sd_scripts.BACKEND.description,
        formats=_sd_scripts.BACKEND.formats,
        zip_only=_sd_scripts.BACKEND.zip_only,
    ),
    "yadc": _BackendDescriptor(
        run=_yadc.run,
        iter_zip_members=_yadc.iter_zip_members,
        name=_yadc.BACKEND.name,
        description=_yadc.BACKEND.description,
        formats=_yadc.BACKEND.formats,
        zip_only=_yadc.BACKEND.zip_only,
    ),
}


def list_backends() -> dict[str, _BackendDescriptor]:
    """Return a dict of all registered backends."""
    return dict(_BACKENDS)


def get_backend(name: str) -> _BackendDescriptor:
    """Look up a backend by name.

    Raises:
        ValueError: If no backend with that name exists.
    """
    backend = _BACKENDS.get(name)
    if backend is None:
        raise ValueError(f"Unknown backend: {name!r}. Available: {', '.join(_BACKENDS)}")
    return backend


def run_export(
    backend_name: str,
    images: list[DatasetImage],
    *,
    fmt: str,
    source: str,
    drafts: tuple[str, ...] = (),
    output: pathlib.Path | None = None,
    append: bool = False,
    caption_extension: str = ".txt",
    delimiter: str = "\n",
) -> int:
    """Dispatch to the named backend's ``run()`` function."""
    descriptor = get_backend(backend_name)
    if fmt not in descriptor.formats:
        raise ValueError(f"Backend {backend_name!r} does not support format {fmt!r}. Available: {', '.join(descriptor.formats)}")

    return descriptor.run(
        images,
        fmt=fmt,
        source=source,
        drafts=drafts,
        output=output,
        append=append,
        caption_extension=caption_extension,
        delimiter=delimiter,
    )


def stream_export_zip(
    backend_name: str,
    images: list[DatasetImage],
    *,
    fmt: str,
    source: str,
    drafts: tuple[str, ...] = (),
    caption_extension: str = ".txt",
    include_images: bool = False,
    base_dir: pathlib.Path | None = None,
    delimiter: str = "\n",
) -> tuple[Iterator[bytes], int]:
    """Stream a dataset export as a zip64 archive.

    Runs the backend's pre-flight validation eagerly (so a missing file raises
    before any bytes are produced), then returns a lazy byte iterator plus the
    count of exported entries. The iterator yields the encoded zip on demand;
    hand it directly to a streaming HTTP response.
    """
    descriptor = get_backend(backend_name)
    if fmt not in descriptor.formats:
        raise ValueError(f"Backend {backend_name!r} does not support format {fmt!r}. Available: {', '.join(descriptor.formats)}")

    members, count = descriptor.iter_zip_members(
        images,
        fmt=fmt,
        source=source,
        drafts=drafts,
        caption_extension=caption_extension,
        include_images=include_images,
        base_dir=base_dir,
        delimiter=delimiter,
    )
    return stream_zip_bytes(members), count
