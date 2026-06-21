"""Permanent fixture for the Python-hook capture path — NOT a migration step.

``_discover_migrations`` ignores this file: its name matches neither
``_MIGRATIONS_PATTERN`` nor ``_PY_HOOKS_PATTERN`` (no leading ``<step>``… prefix),
so it is never treated as a migration. It exists solely so
``DBMigrations._capture_hook`` has a realistic, on-disk module to exercise
whole-module source capture (``__future__`` + imports + ``def migrate``).

Without this guard, a regression to function-only capture
(``inspect.getsource(module.migrate)`` instead of ``inspect.getsource(module)``)
would ship silently — the failure only surfaces at rollback time on a real
data migration, of which the package currently has none. See
``tests/api/test_db_migrations.py::TestHookCapture``.
"""

from __future__ import annotations

import sqlite3


def migrate(conn: sqlite3.Connection) -> None:
    """Fixture body — touches the connection so the signature is realistic.

    Only the source *shape* (``__future__`` + imports + ``def migrate``)
    matters to the capture test; the body is irrelevant, but a real hook
    always uses ``conn``, so this one does a trivial read.
    """
    conn.execute("SELECT 1")
