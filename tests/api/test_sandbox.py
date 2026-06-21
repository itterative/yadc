"""Tests for the stored-hook execution sandbox."""

import sqlite3

import pytest

from yadc.api.modules.sandbox import RestrictedImportError, StoredHookError, exec_hook


class TestAllowedAccess:
    def test_defines_and_returns_callable_migrate(self):
        source = "def migrate(conn):\n    conn.execute('SELECT 1')\n"
        migrate = exec_hook(source, "test_mod")
        assert callable(migrate)

    def test_allowed_stdlib_imports_resolve(self):
        source = (
            "import base64\n"
            "import json\n"
            "import re\n"
            "import sqlite3\n"
            "\n"
            "def migrate(conn):\n"
            "    assert base64.b64encode(b'x') is not None\n"
            "    assert json.dumps({}) is not None\n"
            "    assert re.match('a', 'a') is not None\n"
        )
        exec_hook(source, "test_mod")(sqlite3.connect(":memory:"))

    def test_future_annotations_allowed_without_sqlite3_resolved(self):
        """``sqlite3.Connection`` is never evaluated thanks to ``__future__`` annotations."""
        source = "from __future__ import annotations\n\ndef migrate(conn: sqlite3.Connection) -> None:\n    pass\n"
        exec_hook(source, "test_mod")(sqlite3.connect(":memory:"))


class TestBlockedAccess:
    def test_disallowed_import_raises(self):
        source = "import os\n\ndef migrate(conn):\n    pass\n"
        with pytest.raises(RestrictedImportError, match=r"'os'"):
            exec_hook(source, "test_mod")

    def test_dotted_disallowed_root_raises(self):
        source = "import os.path\n\ndef migrate(conn):\n    pass\n"
        with pytest.raises(RestrictedImportError, match=r"'os.path'"):
            exec_hook(source, "test_mod")

    def test_open_not_accessible(self):
        source = "def migrate(conn):\n    open('/etc/passwd')\n"
        migrate = exec_hook(source, "test_mod")
        with pytest.raises(NameError):
            migrate(sqlite3.connect(":memory:"))

    def test_exec_builtin_not_accessible(self):
        source = "def migrate(conn):\n    exec('1')\n"
        migrate = exec_hook(source, "test_mod")
        with pytest.raises(NameError):
            migrate(sqlite3.connect(":memory:"))


class TestStoredHookShape:
    def test_missing_migrate_raises(self):
        source = "x = 1\n"
        with pytest.raises(StoredHookError, match="does not define a callable"):
            exec_hook(source, "test_mod")

    def test_non_callable_migrate_raises(self):
        source = "migrate = 42\n"
        with pytest.raises(StoredHookError, match="does not define a callable"):
            exec_hook(source, "test_mod")

    def test_realistic_base64_round_trip_runs(self):
        """A representative data-migration hook (base64 decode into a BLOB) runs end to end."""
        source = (
            "import base64\n"
            "\n"
            "def migrate(conn):\n"
            "    conn.execute('CREATE TABLE t (b BLOB)')\n"
            "    conn.execute('INSERT INTO t (b) VALUES (?)', (base64.b64decode(base64.b64encode(b'hi')),))\n"
        )
        conn = sqlite3.connect(":memory:")
        exec_hook(source, "test_mod")(conn)
        assert conn.execute("SELECT b FROM t").fetchone()[0] == b"hi"
