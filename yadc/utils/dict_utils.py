"""Dict utilities — deep merge and related helpers."""

from __future__ import annotations

import copy
from typing import IO, Any, cast

import tomlkit

type TomlValue = str | int | float | bool | None | list["TomlValue"] | dict[str, "TomlValue"]
"""Recursive type representing any value that can appear in a TOML document."""


def load_toml(content: str | bytes, *, plain: bool = True) -> dict[str, TomlValue]:
    """Typed wrapper around :func:`tomlkit.loads` that returns a plain ``dict``.

    tomlkit's ``loads()`` returns a ``TOMLDocument`` whose ``.items()`` and
    ``.get()`` are typed as ``Unknown``, which cascades into dozens of
    ``reportUnknownVariableType`` / ``reportUnknownMemberType`` warnings at
    every call site.  This wrapper widens the return type to
    ``dict[str, TomlValue]`` so downstream code is fully typed.

    When *plain* is ``True`` (the default), the result is converted to plain
    Python types via :func:`toml_to_plain` so that Pydantic's Rust-level
    validation can process it.  Pass ``plain=False`` when you need the raw
    ``TOMLDocument`` for round-tripping through :func:`tomlkit.dumps` or
    :func:`toml_merge`.
    """
    doc = cast("dict[str, TomlValue]", tomlkit.loads(content))
    return toml_to_plain(doc) if plain else doc


def load_toml_file(fp: IO[str] | IO[bytes], *, plain: bool = True) -> dict[str, TomlValue]:
    """Typed wrapper around :func:`tomlkit.load` that returns a plain ``dict``.

    See :func:`load_toml` for why this wrapper exists and what *plain* does.
    """
    doc = cast("dict[str, TomlValue]", tomlkit.load(fp))
    return toml_to_plain(doc) if plain else doc


def deep_merge(
    base: dict[str, TomlValue],
    override: dict[str, TomlValue],
    *,
    remove_none: bool = False,
) -> dict[str, TomlValue]:
    """Recursively merge *override* into *base*, returning a new dict.

    Neither input is mutated.

    Args:
        base: The original dict (e.g. parsed TOML config).
        override: The partial dict to merge in (e.g. user overrides or API patch).
        remove_none: When ``True``, keys whose override value is ``None`` are
            omitted from the result entirely (useful for "reset to default"
            semantics in user config overlays).

    Rules:
        - Dict values are merged recursively.
        - All other types (including lists) are replaced by the override value.
        - Keys present in *base* but not in *override* are kept unchanged.
        - Keys present in *override* but not in *base* are added.
    """
    result: dict[str, TomlValue] = copy.deepcopy(base)

    for key, val in override.items():
        if remove_none and val is None:
            result.pop(key, None)
            continue

        existing = result.get(key)
        if isinstance(existing, dict) and isinstance(val, dict):
            result[key] = deep_merge(existing, val, remove_none=remove_none)
        else:
            result[key] = copy.deepcopy(val)

    return result


def toml_merge(base: dict[str, TomlValue], override: dict[str, TomlValue]) -> dict[str, TomlValue]:
    """Recursively merge *override* into a tomlkit *base* document.

    Returns a new dict — neither input is mutated.  When *base* is a
    ``TOMLDocument``, comments, whitespace, and formatting are preserved
    for untouched keys.  Overridden scalar/list values replace the
    originals.

    Rules (same as :func:`deep_merge`):
        - Dict values are merged recursively.
        - All other types (including lists) are replaced by the override value.
        - Keys in *base* but not in *override* are kept unchanged.
        - Keys in *override* but not in *base* are added.
    """
    result: dict[str, TomlValue] = copy.deepcopy(base)
    _merge_into(result, override)
    return result


def _merge_into(target: Any, override: Any) -> None:
    """Merge *override* into *target* in place, preserving tomlkit containers."""
    for key, val in override.items():
        existing = target.get(key)
        if isinstance(existing, dict) and isinstance(val, dict):
            _merge_into(existing, val)
        elif isinstance(val, (str, int, float, bool)):
            target[key] = val
        else:
            target[key] = copy.deepcopy(val)


def toml_to_plain(doc: dict[str, TomlValue]) -> dict[str, TomlValue]:
    """Recursively convert a tomlkit container to a plain Python dict.

    tomlkit wraps values in its own types (``Integer``, ``String``,
    ``Table``) which Pydantic's Rust-level validation rejects.  This
    helper strips the wrappers by converting scalars to builtins.
    """
    result: dict[str, TomlValue] = {}
    for key, val in doc.items():
        if isinstance(val, dict):
            result[key] = toml_to_plain(val)
        elif isinstance(val, list):
            result[key] = _list_to_plain(val)
        elif isinstance(val, bool):
            result[key] = bool(val)
        elif isinstance(val, int):
            result[key] = int(val)
        elif isinstance(val, float):
            result[key] = float(val)
        elif isinstance(val, str):
            result[key] = str(val)
        else:
            result[key] = val
    return result


def _list_to_plain(items: list[TomlValue]) -> list[TomlValue]:
    out: list[TomlValue] = []
    for item in items:
        if isinstance(item, dict):
            out.append(toml_to_plain(item))
        elif isinstance(item, list):
            out.append(_list_to_plain(item))
        elif isinstance(item, bool):
            out.append(bool(item))
        elif isinstance(item, int):
            out.append(int(item))
        elif isinstance(item, float):
            out.append(float(item))
        elif isinstance(item, str):
            out.append(str(item))
        else:
            out.append(item)
    return out
