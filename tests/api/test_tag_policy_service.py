"""Unit tests for :class:`TagPolicyService`.

Persists and reads always-add / banned lists keyed by dataset name.
A dataset that has never been configured decodes to an empty
:class:`StoredPolicy`; an unregistered dataset returns the empty
default without raising (callers upstream gate on registration).
"""

from __future__ import annotations

import pytest

from yadc.api.modules.db_connection_factory import DBConnectionFactory
from yadc.api.modules.logging_factory import LoggingFactory
from yadc.api.services.dataset_repository import DatasetRepository
from yadc.api.services.dataset_settings_repository import DatasetSettingsRepository
from yadc.api.services.tag_highlights_service import TaggedEntry
from yadc.api.services.tag_policy_service import StoredPolicy, TagPolicyService
from yadc.taggers.postprocessing import TagPolicy


def _e(name: str, canonical_form: bool = True) -> TaggedEntry:
    return TaggedEntry(name=name, canonical_form=canonical_form)


def _policy(always_add: list[str] | None = None, banned: list[str] | None = None) -> StoredPolicy:
    return StoredPolicy(
        always_add=[_e(n) for n in (always_add or [])],
        banned=[_e(n) for n in (banned or [])],
    )


@pytest.fixture
def dataset_repo(
    db_connection_factory: DBConnectionFactory,
    logging_factory: LoggingFactory,
) -> DatasetRepository:
    return DatasetRepository(db_connection_factory, logging_factory)


@pytest.fixture
def settings_repo(
    db_connection_factory: DBConnectionFactory,
    logging_factory: LoggingFactory,
) -> DatasetSettingsRepository:
    return DatasetSettingsRepository(db_connection_factory, logging_factory)


@pytest.fixture
def service(
    db_connection_factory: DBConnectionFactory,
    logging_factory: LoggingFactory,
    settings_repo: DatasetSettingsRepository,
    dataset_repo: DatasetRepository,
) -> TagPolicyService:
    return TagPolicyService(db_connection_factory, logging_factory, settings_repo, dataset_repo)


@pytest.fixture
def register_dataset(dataset_repo: DatasetRepository):
    """Register a dataset and return its name.

    The service resolves ``dataset_id`` via :meth:`DatasetRepository.get_dataset_row`;
    so each test needs a row in ``datasets`` for the policy to attach to.
    """

    def _register(name: str) -> str:
        dataset_repo.upsert_dataset(name=name, config_path="", source="import")
        return name

    return _register


class TestGet:
    def test_returns_empty_policy_for_unregistered_dataset(self, service: TagPolicyService):
        """An unknown name decodes to the empty default, no exception."""
        assert service.get("never-registered") == _policy()

    def test_returns_empty_policy_for_registered_dataset_with_no_settings(self, service, register_dataset):
        """A row in ``datasets`` with no settings rows is the legitimate 'unconfigured' state."""
        register_dataset("ds1")
        assert service.get("ds1") == _policy()

    def test_round_trips_stored_policy(self, service, register_dataset):
        register_dataset("ds1")
        stored = _policy(always_add=["masterpiece", "1girl"], banned=["nsfw"])
        assert service.set("ds1", stored) is True
        assert service.get("ds1") == stored

    def test_canonical_form_flag_round_trips(self, service, register_dataset):
        """``canonical_form: False`` (free-text) is persisted alongside the
        name; the identity is stored verbatim and rendered as-is by the
        frontend."""
        register_dataset("ds1")
        stored = StoredPolicy(always_add=[_e("custom_tag", canonical_form=False)], banned=[])
        service.set("ds1", stored)
        assert service.get("ds1") == stored

    def test_kaomoji_canonical_form_forced_to_false(self, service, register_dataset):
        """A kaomoji name sent with ``canonical_form: True`` is flipped to
        ``False`` by the Pydantic validator on the way to storage."""
        register_dataset("ds1")
        stored = StoredPolicy(always_add=[_e("0_0", canonical_form=True)], banned=[])
        service.set("ds1", stored)
        result = service.get("ds1")
        assert result.always_add == [TaggedEntry(name="0_0", canonical_form=False)]

    def test_per_dataset_isolation(self, service, register_dataset):
        register_dataset("ds1")
        register_dataset("ds2")
        service.set("ds1", _policy(always_add=["a"]))
        service.set("ds2", _policy(always_add=["b"]))
        assert service.get("ds1") == _policy(always_add=["a"])
        assert service.get("ds2") == _policy(always_add=["b"])

    def test_unparseable_json_falls_back_to_empty(self, service, settings_repo, dataset_repo, register_dataset):
        """A row with garbage ``value`` is treated as empty + logged."""
        name = register_dataset("ds1")
        settings_repo.upsert(
            dataset_repo.get_dataset_row(name)[0],  # type: ignore[index]
            "policy_always_add",
            "this is not json",
        )
        assert service.get("ds1") == _policy()

    def test_non_list_json_falls_back_to_empty(self, service, settings_repo, dataset_repo, register_dataset):
        """A JSON object instead of an array is treated as empty + logged."""
        name = register_dataset("ds1")
        settings_repo.upsert(
            dataset_repo.get_dataset_row(name)[0],  # type: ignore[index]
            "policy_always_add",
            '{"oops": "object"}',
        )
        assert service.get("ds1") == _policy()


class TestSet:
    def test_returns_false_for_unregistered_dataset(self, service: TagPolicyService):
        assert service.set("never-registered", _policy(always_add=["x"])) is False

    def test_overwrites_existing_policy(self, service, register_dataset):
        register_dataset("ds1")
        assert service.set("ds1", _policy(always_add=["a"])) is True
        assert service.set("ds1", _policy(always_add=["b"])) is True
        assert service.get("ds1") == _policy(always_add=["b"])

    def test_empty_lists_are_persisted(self, service, register_dataset):
        """Clearing a list is a real edit — the row stays so the next read sees ``[]``."""
        register_dataset("ds1")
        service.set("ds1", _policy(always_add=["a"], banned=["b"]))
        assert service.set("ds1", _policy()) is True
        result = service.get("ds1")
        assert result.always_add == []
        assert result.banned == []

    def test_each_key_is_stored_independently(self, service, settings_repo, dataset_repo, register_dataset):
        """Editing one list doesn't blank the other."""
        name = register_dataset("ds1")
        service.set(name, _policy(always_add=["a"], banned=["b"]))
        service.set(name, _policy(always_add=["a"], banned=[]))
        assert service.get(name) == _policy(always_add=["a"], banned=[])
        dataset_id = dataset_repo.get_dataset_row(name)[0]  # type: ignore[index]
        assert settings_repo.get(dataset_id, "policy_banned") is not None  # ``[]`` persisted as JSON text


class TestNamesForPolicy:
    def test_drops_per_entry_metadata_into_legacy_names(self, service: TagPolicyService):
        """The matching layer (:func:`apply_policy`) operates on bare
        names — this helper flattens the curated-tier storage into the
        legacy ``TagPolicy`` shape."""
        policy = StoredPolicy(
            always_add=[_e("a"), _e("b", canonical_form=False)],
            banned=[_e("c")],
        )
        flat = service.names_for_policy(policy)
        assert flat == TagPolicy(always_add=["a", "b"], banned=["c"])
