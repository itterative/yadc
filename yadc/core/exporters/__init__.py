"""Export backend registry.

Each backend is a module under ``yadc.core.exporters`` that exposes:

- ``BACKEND`` — a dataclass (frozen) with ``name``, ``description``, ``formats``.
- ``run(images, *, fmt, source, draft_name, output, append, caption_extension) -> int``

To register a new backend, import its module and add it to ``_BACKENDS`` below.
"""

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
        draft_name: str,
        output: pathlib.Path | None,
        append: bool,
        caption_extension: str,
    ) -> int: ...


@dataclass(frozen=True)
class _BackendDescriptor:
    run: _RunFn
    name: str
    description: str
    formats: tuple[str, ...]


_BACKENDS: dict[str, _BackendDescriptor] = {
    "sd-scripts": _BackendDescriptor(
        run=_sd_scripts.run,
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
    draft_name: str,
    output: pathlib.Path | None,
    append: bool,
    caption_extension: str,
) -> int:
    """Dispatch to the named backend's ``run()`` function."""
    descriptor = get_backend(backend_name)
    if fmt not in descriptor.formats:
        raise ValueError(f"Backend {backend_name!r} does not support format {fmt!r}. Available: {', '.join(descriptor.formats)}")

    return descriptor.run(
        images,
        fmt=fmt,
        source=source,
        draft_name=draft_name,
        output=output,
        append=append,
        caption_extension=caption_extension,
    )
