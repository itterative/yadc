"""Export backend registry.

Each backend is a module under ``yadc.core.exporters`` that exposes:

- ``BACKEND`` — a dataclass (frozen) with ``name``, ``description``, ``formats``.
- ``run(images, *, fmt, source, drafts, output, append, caption_extension) -> int``
- ``run_zip(images, *, fmt, source, drafts, caption_extension, include_images, base_dir) -> tuple[BytesIO, int]``

To register a new backend, import its module and add it to ``_BACKENDS`` below.
"""

import io
import pathlib
from dataclasses import dataclass
from typing import Protocol

from ..dataset import DatasetImage
from . import sd_scripts as _sd_scripts


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
    ) -> int: ...


class _RunZipFn(Protocol):
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
    ) -> tuple[io.BytesIO, int]: ...


@dataclass(frozen=True)
class _BackendDescriptor:
    run: _RunFn
    run_zip: _RunZipFn
    name: str
    description: str
    formats: tuple[str, ...]


_BACKENDS: dict[str, _BackendDescriptor] = {
    "sd-scripts": _BackendDescriptor(
        run=_sd_scripts.run,
        run_zip=_sd_scripts.run_zip,
        name=_sd_scripts.BACKEND.name,
        description=_sd_scripts.BACKEND.description,
        formats=_sd_scripts.BACKEND.formats,
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
    )


def run_export_zip(
    backend_name: str,
    images: list[DatasetImage],
    *,
    fmt: str,
    source: str,
    drafts: tuple[str, ...] = (),
    caption_extension: str = ".txt",
    include_images: bool = False,
    base_dir: pathlib.Path | None = None,
) -> tuple[io.BytesIO, int]:
    """Dispatch to the named backend's ``run_zip()`` function."""
    descriptor = get_backend(backend_name)
    if fmt not in descriptor.formats:
        raise ValueError(f"Backend {backend_name!r} does not support format {fmt!r}. Available: {', '.join(descriptor.formats)}")

    return descriptor.run_zip(
        images,
        fmt=fmt,
        source=source,
        drafts=drafts,
        caption_extension=caption_extension,
        include_images=include_images,
        base_dir=base_dir,
    )
