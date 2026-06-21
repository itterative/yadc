"""Restricted-exec sandbox for stored Python migration hooks.

The canonical migration hook source lives in the :mod:`yadc.api.migrations`
package and is fully trusted. :mod:`yadc.api.modules.db_migrations` also stores
that source in the DB (``migration_downgrades_py.down_py_source``) so rollbacks
keep working when the package source is absent — frozen executable, package
downgrade, etc. On rollback the stored source is ``exec``d.

This module narrows the blast radius if a stored row is ever corrupted or
tampered: the source runs with a reduced ``__builtins__`` and a
whitelist-only ``__import__``. It is defense-in-depth, **not** a security
boundary — see the ``db_migrations`` module docstring for the threat model
(an attacker with write access to the local DB has write access to the
package too).
"""

from __future__ import annotations

import builtins
import sqlite3
from collections.abc import Callable
from types import ModuleType
from typing import cast

# Standard-library modules a data-migration hook may legitimately need.
# Deliberately omits anything that escapes the process (``os``, ``sys``,
# ``subprocess``, ``shutil``, ``socket``, ``ctypes`` …). Keep this list
# **additive only**: a stored row captured by an older yadc version is
# checked against the *current* whitelist at rollback time, so removing an
# entry could brick a rollback on an existing DB.
_ALLOWED_MODULES: frozenset[str] = frozenset(
    {
        "__future__",  # ``from __future__ import annotations`` is routed through __import__
        "base64",
        "binascii",
        "collections",
        "datetime",
        "itertools",
        "json",
        "math",
        "re",
        "sqlite3",
    }
)

# Builtins removed from the sandbox namespace — file/process access and
# dynamic exec. ``__import__`` is replaced by a whitelist-checked version
# (see :func:`_restricted_import`), not removed outright, so ``import``
# statements in the hook still work for allowed modules.
_DANGEROUS_BUILTINS: frozenset[str] = frozenset({"open", "compile", "eval", "exec", "input", "breakpoint", "exit", "quit", "__import__"})


class RestrictedImportError(ImportError):
    """Raised when stored hook source imports a module not on the allowlist."""


class StoredHookError(Exception):
    """Raised when stored hook source does not define a callable ``migrate``."""


def _restricted_import(
    name: str,
    globals: dict[str, object] | None = None,
    locals: dict[str, object] | None = None,
    fromlist: tuple[str, ...] = (),
    level: int = 0,
) -> ModuleType:
    root = name.split(".", 1)[0]
    if root not in _ALLOWED_MODULES:
        raise RestrictedImportError(f"Stored migration hooks may not import {name!r}. Allowed: {sorted(_ALLOWED_MODULES)}.")
    return builtins.__import__(name, globals, locals, fromlist, level)


def _safe_builtins() -> dict[str, object]:
    safe: dict[str, object] = {k: v for k, v in vars(builtins).items() if k not in _DANGEROUS_BUILTINS}
    safe["__import__"] = _restricted_import
    return safe


def exec_hook(source: str, module_name: str) -> Callable[[sqlite3.Connection], None]:
    """Exec ``source`` in a restricted namespace and return its ``migrate`` callable.

    The source is the whole hook module (docstring + ``__future__`` import +
    ``import`` statements + ``def migrate(conn)``), as captured by
    :func:`yadc.api.modules.db_migrations._capture_hook`. It must define a
    ``migrate(conn: sqlite3.Connection) -> None`` function.
    """
    namespace: dict[str, object] = {"__name__": module_name, "__builtins__": _safe_builtins()}
    exec(compile(source, f"<stored:{module_name}>", "exec"), namespace)
    migrate = namespace.get("migrate")
    if migrate is None or not callable(migrate):
        raise StoredHookError(f"Stored Python source for migration {module_name!r} does not define a callable ``migrate(conn)``.")
    return cast(Callable[[sqlite3.Connection], None], migrate)
