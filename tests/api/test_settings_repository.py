"""Unit tests for SettingsRepository.

Strategy:

- Use the shared :class:`DBConnectionFactory` fixture (with migrations
  applied to a temp-file DB) so the schema is created by the existing
  migrations.
- Build up state by calling the repository's public methods.
- Assert on the repository's public methods (no raw SQL in tests).
"""

from __future__ import annotations

import pytest

from yadc.api.modules.db_connection_factory import DBConnectionFactory
from yadc.api.modules.logging_factory import LoggingFactory
from yadc.api.services.settings_repository import SettingsRepository


@pytest.fixture
def repo(
    db_connection_factory: DBConnectionFactory,
    logging_factory: LoggingFactory,
) -> SettingsRepository:
    return SettingsRepository(db=db_connection_factory, logging=logging_factory)


class TestGet:
    def test_returns_none_for_missing_key(self, repo):
        assert repo.get("missing") is None

    def test_returns_raw_value(self, repo):
        repo.upsert("foo", '"bar"')
        assert repo.get("foo") == '"bar"'


class TestUpsert:
    def test_inserts_new_key(self, repo):
        repo.upsert("key1", '"value1"')
        assert repo.get("key1") == '"value1"'

    def test_updates_existing_key(self, repo):
        repo.upsert("key1", '"old"')
        repo.upsert("key1", '"new"')
        assert repo.get("key1") == '"new"'


class TestDelete:
    def test_returns_one_for_existing_key(self, repo):
        repo.upsert("foo", '"x"')
        assert repo.delete("foo") == 1

    def test_returns_zero_for_missing_key(self, repo):
        assert repo.delete("missing") == 0

    def test_actually_deletes(self, repo):
        repo.upsert("foo", '"x"')
        assert repo.delete("foo") == 1
        assert repo.get("foo") is None


class TestListAll:
    def test_empty_table(self, repo):
        assert repo.list_all() == []

    def test_returns_all_rows_in_key_order(self, repo):
        repo.upsert("b", '"2"')
        repo.upsert("a", '"1"')
        repo.upsert("c", '"3"')
        assert repo.list_all() == [("a", '"1"'), ("b", '"2"'), ("c", '"3"')]
