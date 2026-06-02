"""Dict utilities — deep merge and related helpers."""

import copy

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
