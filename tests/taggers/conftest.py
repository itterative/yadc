"""Shared fixtures for the tagger test suite.

Re-declares the small set of fixtures used by ``test_service.py`` /
``test_events.py`` so the tagger tests are self-contained (the larger
fixtures in ``tests/api/conftest.py`` are only visible inside
``tests/api/``).
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from yadc.api.configuration import Configuration
from yadc.api.modules import EventDispatcher, JobScheduler
from yadc.api.modules.db_connection_factory import DBConnectionFactory
from yadc.api.modules.db_migrations import DBMigrations
from yadc.api.modules.logging_factory import LoggingFactory
from yadc.api.services.dataset_jobs import DatasetJobService
from yadc.api.services.dataset_repository import ImageInfo
from yadc.api.services.datasets import DatasetService
from yadc.api.services.settings import SettingsService
from yadc.api.services.settings_repository import SettingsRepository
from yadc.api.services.tagging import TaggingService
from yadc.taggers.base import TaggerResult

# ---------------------------------------------------------------------------
# Configuration / logger / dataset / event plumbing
# ---------------------------------------------------------------------------


@pytest.fixture
def test_configuration(tmp_path: Path) -> Configuration:
    """A Configuration with paths under ``tmp_path`` so tests don't touch user data.

    Every tagger field is set explicitly so local edits to
    ``yadc/api/configuration.py`` (e.g. tmp commits pointing at the
    user's animetimm path) can't silently affect which model the
    tagger thinks it's running. Tests that need a specific tagger
    shape should build a fresh ``Configuration(...)`` inline (or
    override individual fields) rather than rely on the fixture.
    """
    return Configuration(
        # Filesystem paths under tmp_path.
        db_path=str(tmp_path / "test.db"),
        state_path=str(tmp_path / "state"),
        config_path=str(tmp_path / "config.toml"),
        cache_path=str(tmp_path / "cache"),
        # Tagger model identity \u2014 explicitly empty so no implicit
        # default configures the tagger. Tests that want a model
        # configure it inline.
        tagger_model_path="",
        tagger_label_path="",
        tagger_repo_id="",
        tagger_repo_model_filename="model.onnx",
        tagger_repo_label_filename="selected_tags.csv",
        tagger_preproc_profile="wd-tagger",
        tagger_default_input_size=0,
        # Tagger thresholds \u2014 standard wd-tagger canonical values.
        tagger_rating_threshold=0.0,
        tagger_general_threshold=0.35,
        tagger_character_threshold=0.85,
        tagger_replace_underscores=False,
        # Subprocess liveness knobs \u2014 shrunk so timing-sensitive
        # tests detect a dead/unresponsive worker quickly.
        tagger_heartbeat_interval_seconds=0.5,
        tagger_response_timeout_seconds=5.0,
        tagger_liveness_poll_seconds=0.1,
        tagger_cancel_grace_seconds=0.5,
        tagger_idle_timeout_seconds=60.0,
        # Immediate deferred expected-changes clear so tests don't
        # wait out the production grace window.
        tagger_expected_changes_grace_seconds=0.0,
        # Memory budget for the result cache.
        tagger_result_max_memory_bytes=1024 * 1024,
    )


@pytest.fixture
def logging_factory(test_configuration: Configuration) -> LoggingFactory:
    return LoggingFactory(test_configuration)


@pytest.fixture
def event_dispatcher(logging_factory: LoggingFactory) -> EventDispatcher:
    return EventDispatcher(logging_factory)


@pytest.fixture
def dataset_jobs(logging_factory: LoggingFactory) -> DatasetJobService:
    """A real coordinator so service-level tests exercise the claim path."""
    return DatasetJobService(logging=logging_factory)


@pytest.fixture
def db_connection_factory(
    test_configuration: Configuration,
    logging_factory: LoggingFactory,
) -> Iterator[DBConnectionFactory]:
    """A real ``DBConnectionFactory`` with migrations applied to a tmp SQLite DB.

    Settings-persistence tests need an actual store (not a mock) so the
    JSON round-trip and the ``SettingsService`` error paths are
    exercised. The factory is closed after the test to release the
    keep-alive connection.
    """
    migrations = DBMigrations(logging_factory)
    factory = DBConnectionFactory(test_configuration, logging_factory, migrations)
    yield factory
    factory.close()


@pytest.fixture
def settings_service(
    logging_factory: LoggingFactory,
    db_connection_factory: DBConnectionFactory,
) -> SettingsService:
    """A real ``SettingsService`` wired to a tmp SQLite DB.

    First real consumer of the existing settings KV store in this test
    suite — used by tagger selection persistence (``tagger.active_model``).
    """
    repo = SettingsRepository(db_connection_factory, logging_factory)
    return SettingsService(db_connection_factory, logging_factory, repo)


@pytest.fixture
def job_scheduler() -> MagicMock:
    """A ``MagicMock(spec=JobScheduler)`` — the idle check isn't exercised in these tests."""
    return MagicMock(spec=JobScheduler)


@pytest.fixture
def service(
    test_configuration,
    logging_factory: LoggingFactory,
    event_dispatcher: EventDispatcher,
    dataset_service: MagicMock,
    dataset_watcher: MagicMock,
    job_scheduler: MagicMock,
    dataset_jobs: DatasetJobService,
    settings_service: SettingsService,
) -> TaggingService:
    """A ``TaggingService`` configured for a local-path tagger model with a 60s idle timeout.

    Shared across the tagger test suite so individual files don't
    redefine the same fixture. Tests that need a different
    configuration (or that pre-seed ``settings_service``) should
    build a service via the ``service_factory`` fixture below.
    """
    test_configuration.tagger_model_path = "/fake/model.onnx"
    test_configuration.tagger_idle_timeout_seconds = 60.0
    return TaggingService(
        configuration=test_configuration,
        logging=logging_factory,
        event_dispatcher=event_dispatcher,
        dataset_service=dataset_service,
        dataset_watcher=dataset_watcher,
        job_scheduler=job_scheduler,
        dataset_jobs=dataset_jobs,
        settings_service=settings_service,
    )


@pytest.fixture
def service_factory(
    test_configuration,
    logging_factory: LoggingFactory,
    event_dispatcher: EventDispatcher,
    dataset_service: MagicMock,
    dataset_watcher: MagicMock,
    job_scheduler: MagicMock,
    dataset_jobs: DatasetJobService,
):
    """Build a ``TaggingService`` against a caller-provided ``SettingsService``.

    Useful for tests that pre-seed the settings table before
    constructing the service (so the hydration path sees the seeded
    row). Mirrors ``service`` but takes the ``SettingsService`` as an
    argument so the test controls which store the constructor reads.
    """
    from yadc.api.services.settings import SettingsService as _SettingsService

    def _factory(svc: _SettingsService) -> TaggingService:
        return TaggingService(
            configuration=test_configuration,
            logging=logging_factory,
            event_dispatcher=event_dispatcher,
            dataset_service=dataset_service,
            dataset_watcher=dataset_watcher,
            job_scheduler=job_scheduler,
            dataset_jobs=dataset_jobs,
            settings_service=svc,
        )

    return _factory


@pytest.fixture
def dataset_service() -> MagicMock:
    """A ``MagicMock(spec=DatasetService)`` for service-level tagger tests.

    Service tests mock the subprocess and exercise the service state
    machine; the dataset service is only touched by the batch/save paths,
    which have their own targeted tests that configure its return values.
    """
    return MagicMock(spec=DatasetService)


@pytest.fixture
def dataset_watcher() -> MagicMock:
    """A ``MagicMock(spec=DatasetWatcherService)`` — expect_changes/clear are no-ops in tests."""
    from yadc.api.modules.dataset_watcher import DatasetWatcherService

    return MagicMock(spec=DatasetWatcherService)


# ---------------------------------------------------------------------------
# Image fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_image(tmp_path: Path) -> Path:
    """Create a tiny PNG on disk and return the path."""
    from PIL import Image

    img_path = tmp_path / "img.png"
    Image.new("RGB", (1, 1), color="red").save(img_path, format="PNG")
    return img_path


@pytest.fixture
def image_info(fake_image: Path) -> ImageInfo:
    """A valid ``ImageInfo`` for the fake image (id=1)."""
    return ImageInfo(id=1, file_name=fake_image.name, path=str(fake_image), width=1, height=1)


@pytest.fixture
def make_image_info(tmp_path: Path):
    """Factory for distinct ``ImageInfo`` objects, each backed by its own file.

    Tests that issue multiple tag requests should call this multiple
    times so each request targets a different image (different id,
    different path) — matching real-world usage where two tag calls
    almost never target the same image.
    """
    from PIL import Image

    counter = 0

    def factory(*, image_id: int | None = None) -> ImageInfo:
        nonlocal counter
        counter += 1
        file_name = f"img{counter}.png"
        path = tmp_path / file_name
        Image.new("RGB", (1, 1), color="red").save(path, format="PNG")
        return ImageInfo(id=image_id if image_id is not None else counter, file_name=file_name, path=str(path), width=1, height=1)

    return factory


# ---------------------------------------------------------------------------
# Subprocess client mock + patcher
# ---------------------------------------------------------------------------


def make_client_mock(*, alive: bool = True, tag_result: TaggerResult | None = None) -> MagicMock:
    """Build a mock standing in for ``TaggerClient``.

    ``start`` / ``stop`` are awaitable, ``tag`` returns *tag_result* (or a
    fixed 1girl/safe result when omitted), and ``is_alive`` is a real
    bool so the service's liveness checks work.
    """
    client = MagicMock()
    client.is_alive = alive
    client.start = AsyncMock()
    client.stop = AsyncMock()
    client.tag = AsyncMock(
        return_value=tag_result
        or TaggerResult(
            tags={"1girl": 0.99, "safe": 0.99},
            categories={"general": ["1girl"], "rating": ["safe"]},
        )
    )
    # _server is what the sync stop path reaches into; give it a sync
    # stop() so _stop_locked_sync is exercisable.
    client._server = MagicMock()
    client._server.stop = MagicMock()
    return client


@contextmanager
def patch_client_factory(client: Any):
    """Monkeypatches ``TaggingService``'s subprocess spawn to return *client*.

    The service constructs a ``TaggerClient(OnnxTagger, ...)`` then awaits
    ``client.start()``; patching ``TaggerClient`` in the service module
    short-circuits the construction so every spawn returns our mock.

    Usage::

        with patch_client_factory(client):
            await service.tag_image(...)
    """
    from yadc.api.services import tagging as tagging_module

    with patch.object(tagging_module, "TaggerClient", return_value=client):
        yield client
