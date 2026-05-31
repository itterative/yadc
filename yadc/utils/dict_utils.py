"""Dict utilities — deep merge and related helpers."""

from __future__ import annotations

import copy

from tomlkit.toml_document import TOMLDocument

type TomlValue = str | int | float | bool | None | list["TomlValue"] | dict[str, "TomlValue"]
"""Recursive type representing any value that can appear in a TOML document."""


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


def toml_merge(base: "TOMLDocument", override: dict) -> "TOMLDocument":
    """Recursively merge *override* into a tomlkit *base* document.

    Returns a new ``TOMLDocument`` — neither input is mutated.  Comments,
    whitespace, and formatting from *base* are preserved for untouched
    keys.  Overridden scalar/list values replace the originals.

    Rules (same as :func:`deep_merge`):
        - Dict values are merged recursively.
        - All other types (including lists) are replaced by the override value.
        - Keys in *base* but not *override* are kept unchanged.
        - Keys in *override* but not in *base* are added.
    """
    result = copy.deepcopy(base)
    _merge_into(result, override)
    return result


def _merge_into(target: dict, override: dict) -> None:
    """Merge *override* into *target* in place, preserving tomlkit containers."""
    for key, val in override.items():
        existing = target.get(key)
        if isinstance(existing, dict) and isinstance(val, dict):
            _merge_into(existing, val)
        elif isinstance(val, (str, int, float, bool)):
            target[key] = val
        else:
            target[key] = copy.deepcopy(val)


def toml_to_plain(doc: dict) -> dict:
    """Recursively convert a tomlkit container to a plain Python dict.

    tomlkit wraps values in its own types (``Integer``, ``String``,
    ``Table``) which Pydantic's Rust-level validation rejects.  This
    helper strips the wrappers by converting scalars to builtins.
    """
    result: dict = {}
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


def _list_to_plain(items: list) -> list:
    out: list = []
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
