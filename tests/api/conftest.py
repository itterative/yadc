"""Shared fixtures for the API test suite.

The configuration and database factory fixtures live here so individual
test files don't have to rebuild them with the same test-specific
defaults.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from yadc.api.configuration import Configuration
from yadc.api.modules.db_connection_factory import DBConnectionFactory
from yadc.api.modules.db_migrations import DBMigrations
from yadc.api.modules.logging_factory import LoggingFactory
from yadc.api.services.dataset_jobs import DatasetJobService
from yadc.api.services.settings import SettingsService
from yadc.api.services.settings_repository import SettingsRepository


@pytest.fixture
def test_configuration(tmp_path: Path) -> Configuration:
    """A Configuration populated with paths under ``tmp_path``.

    Database and state/cache/config directories all point at
    ``tmp_path`` so tests don't touch real user data and the DB is
    recreated fresh for each test. Every tagger field is set
    explicitly so local edits to ``yadc/api/configuration.py``
    (e.g. tmp commits pointing at the user's animetimm path) can't
    silently configure the tagger for tests that didn't ask for one.
    """
    return Configuration(
        db_path=str(tmp_path / "test.db"),
        state_path=str(tmp_path / "state"),
        config_path=str(tmp_path / "config.toml"),
        cache_path=str(tmp_path / "cache"),
        # Tagger model identity — explicitly empty.
        tagger_model_path="",
        tagger_label_path="",
        tagger_repo_id="",
        tagger_repo_model_filename="model.onnx",
        tagger_repo_label_filename="selected_tags.csv",
        tagger_preproc_profile="wd-tagger",
        tagger_default_input_size=0,
        # Thresholds.
        tagger_rating_threshold=0.0,
        tagger_general_threshold=0.35,
        tagger_character_threshold=0.85,
        tagger_replace_underscores=False,
        # Subprocess liveness.
        tagger_heartbeat_interval_seconds=0.5,
        tagger_response_timeout_seconds=5.0,
        tagger_liveness_poll_seconds=0.1,
        tagger_cancel_grace_seconds=0.5,
        tagger_idle_timeout_seconds=60.0,
        tagger_expected_changes_grace_seconds=0.0,
        tagger_result_max_memory_bytes=1024 * 1024,
    )


@pytest.fixture
def logging_factory(test_configuration: Configuration) -> LoggingFactory:
    """A LoggingFactory wired to the test configuration."""
    return LoggingFactory(test_configuration)


@pytest.fixture
def dataset_jobs(logging_factory: LoggingFactory) -> DatasetJobService:
    """A real coordinator shared by service-level tests."""
    return DatasetJobService(logging=logging_factory)


@pytest.fixture
def settings_service(
    db_connection_factory: DBConnectionFactory,
    logging_factory: LoggingFactory,
) -> SettingsService:
    """A real SettingsService wired to a tmp SQLite DB."""
    repo = SettingsRepository(db_connection_factory, logging_factory)
    return SettingsService(db_connection_factory, logging_factory, repo)


@pytest.fixture
def db_connection_factory(
    test_configuration: Configuration,
    logging_factory: LoggingFactory,
) -> Iterator[DBConnectionFactory]:
    """A real DBConnectionFactory with a temp-file DB. Migrations run synchronously."""
    migrations = DBMigrations(logging_factory)
    factory = DBConnectionFactory(test_configuration, logging_factory, migrations)
    yield factory
    factory.close()
