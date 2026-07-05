"""Unit tests for :class:`DatasetSettingsRepository`.

Mirrors the structure of ``tests/api/test_settings_repository.py``:
real DB + migrations on a temp file, then exercise the repo's public
surface. JSON encoding / decoding is the service layer's job — the
repo just stores the raw bytes.
"""

from __future__ import annotations

import pytest

from yadc.api.modules.db_connection_factory import DBConnectionFactory
from yadc.api.modules.logging_factory import LoggingFactory
from yadc.api.services.dataset_settings_repository import DatasetSettingsRepository


@pytest.fixture
def repo(
    db_connection_factory: DBConnectionFactory,
    logging_factory: LoggingFactory,
) -> DatasetSettingsRepository:
    return DatasetSettingsRepository(db_connection_factory, logging_factory)


@pytest.fixture
def dataset_id(db_connection_factory: DBConnectionFactory) -> int:
    """Insert one dataset row so the FK target exists; return its id."""
    with db_connection_factory.connection() as conn:
        conn.execute(
            "INSERT INTO datasets (name, source, config_path) VALUES (?, ?, ?)",
            ("ds1", "import", None),
        )
        row = conn.execute("SELECT id FROM datasets WHERE name = ?", ("ds1",)).fetchone()
        assert row is not None
        return row[0]


class TestGet:
    def test_returns_none_for_missing_key(self, repo, dataset_id: int):
        assert repo.get(dataset_id, "missing") is None

    def test_returns_raw_value_for_existing_key(self, repo, dataset_id: int):
        repo.upsert(dataset_id, "policy_always_add", '["a", "b"]')
        assert repo.get(dataset_id, "policy_always_add") == '["a", "b"]'

    def test_different_keys_are_independent(self, repo, dataset_id: int):
        repo.upsert(dataset_id, "policy_always_add", '["a"]')
        repo.upsert(dataset_id, "policy_banned", '["b"]')
        assert repo.get(dataset_id, "policy_always_add") == '["a"]'
        assert repo.get(dataset_id, "policy_banned") == '["b"]'


class TestUpsert:
    def test_inserts_new_row(self, repo, dataset_id: int):
        repo.upsert(dataset_id, "policy_always_add", '"x"')
        assert repo.get(dataset_id, "policy_always_add") == '"x"'

    def test_overwrites_existing_row(self, repo, dataset_id: int):
        repo.upsert(dataset_id, "policy_always_add", '"old"')
        repo.upsert(dataset_id, "policy_always_add", '"new"')
        assert repo.get(dataset_id, "policy_always_add") == '"new"'


class TestDelete:
    def test_returns_one_for_existing_row(self, repo, dataset_id: int):
        repo.upsert(dataset_id, "policy_always_add", '"x"')
        assert repo.delete(dataset_id, "policy_always_add") == 1

    def test_returns_zero_for_missing_row(self, repo, dataset_id: int):
        assert repo.delete(dataset_id, "missing") == 0

    def test_deleting_one_key_keeps_others(self, repo, dataset_id: int):
        repo.upsert(dataset_id, "policy_always_add", '"x"')
        repo.upsert(dataset_id, "policy_banned", '"y"')
        assert repo.delete(dataset_id, "policy_always_add") == 1
        assert repo.get(dataset_id, "policy_always_add") is None
        assert repo.get(dataset_id, "policy_banned") == '"y"'


class TestListForDataset:
    def test_empty(self, repo, dataset_id: int):
        assert repo.list_for_dataset(dataset_id) == []

    def test_returns_rows_in_key_order(self, repo, dataset_id: int):
        repo.upsert(dataset_id, "policy_banned", '"2"')
        repo.upsert(dataset_id, "policy_always_add", '"1"')
        assert repo.list_for_dataset(dataset_id) == [
            ("policy_always_add", '"1"'),
            ("policy_banned", '"2"'),
        ]


class TestForeignKeyCascade:
    """Deleting a dataset should remove its settings rows.

    The ``ON DELETE CASCADE`` on the FK is the dataset cleanup story;
    these tests pin the behaviour so a future migration that drops the
    constraint by mistake gets caught.
    """

    def test_deleting_dataset_cascades_to_settings(self, db_connection_factory, dataset_id: int):
        with db_connection_factory.connection() as conn:
            conn.execute(
                "INSERT INTO dataset_settings (dataset_id, key, value) VALUES (?, ?, ?)",
                (dataset_id, "policy_always_add", '"x"'),
            )
            conn.execute("DELETE FROM datasets WHERE id = ?", (dataset_id,))
            row = conn.execute("SELECT 1 FROM dataset_settings WHERE dataset_id = ?", (dataset_id,)).fetchone()
            assert row is None
