"""``size_units`` — pretty-print a byte count in B/KiB/MiB."""

_units = ["B", "KiB", "MiB"]


def size_units(size: int) -> str:
    _size = float(size)

    unit = _units[0]
    for unit in _units:
        if _size >= 1024:
            _size /= 1024
            continue

        break

    return f"{_size:.2f} {unit}"
