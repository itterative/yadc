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


@pytest.fixture
def test_configuration(tmp_path: Path) -> Configuration:
    """A Configuration populated with paths under ``tmp_path``.

    Database and state/cache/config directories all point at
    ``tmp_path`` so tests don't touch real user data and the DB is
    recreated fresh for each test.
    """
    return Configuration(
        db_path=str(tmp_path / "test.db"),
        state_path=str(tmp_path / "state"),
        config_path=str(tmp_path / "config.toml"),
        cache_path=str(tmp_path / "cache"),
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
def db_connection_factory(
    test_configuration: Configuration,
    logging_factory: LoggingFactory,
) -> Iterator[DBConnectionFactory]:
    """A real DBConnectionFactory with a temp-file DB. Migrations run synchronously."""
    migrations = DBMigrations(logging_factory)
    factory = DBConnectionFactory(test_configuration, logging_factory, migrations)
    yield factory
    factory.close()
