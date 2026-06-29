"""Tests for the cross-service dataset job coordinator.

The coordinator is the single source of truth for "who holds a dataset":
at most one sidecar-writing job (captioning or tagging) per dataset at a
time. These tests cover the coordinator in isolation and the cross-service
mutual exclusion it enforces.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from yadc.api.configuration import Configuration
from yadc.api.modules.db_connection_factory import DBConnectionFactory
from yadc.api.modules.logging_factory import LoggingFactory
from yadc.api.services.dataset_jobs import (
    DatasetBusyError,
    DatasetJobService,
)


@pytest.fixture
def test_configuration(tmp_path: Path) -> Configuration:
    return Configuration(
        db_path=str(tmp_path / "test.db"),
        state_path=str(tmp_path / "state"),
        config_path=str(tmp_path / "config.toml"),
        cache_path=str(tmp_path / "cache"),
    )


@pytest.fixture
def logging_factory(test_configuration: Configuration) -> LoggingFactory:
    return LoggingFactory(test_configuration)


@pytest.fixture
def jobs(logging_factory: LoggingFactory) -> DatasetJobService:
    return DatasetJobService(logging=logging_factory)


# ---------------------------------------------------------------------------
# Coordinator primitives
# ---------------------------------------------------------------------------


class TestAcquireRelease:
    def test_acquire_then_busy(self, jobs: DatasetJobService) -> None:
        asyncio.run(jobs.try_acquire("ds", "captioning", "job-1"))
        assert jobs.is_busy("ds")
        assert jobs.current("ds").kind == "captioning"  # pyright: ignore[reportOptionalMemberAccess]

    def test_release_frees_dataset(self, jobs: DatasetJobService) -> None:
        claim = asyncio.run(jobs.try_acquire("ds", "tagging", "job-1"))
        asyncio.run(jobs.release(claim))
        assert not jobs.is_busy("ds")
        assert jobs.current("ds") is None

    def test_reacquire_after_release(self, jobs: DatasetJobService) -> None:
        c1 = asyncio.run(jobs.try_acquire("ds", "captioning", "a"))
        asyncio.run(jobs.release(c1))
        # A different kind can now take it.
        c2 = asyncio.run(jobs.try_acquire("ds", "tagging", "b"))
        assert c2.kind == "tagging"

    def test_separate_datasets_independent(self, jobs: DatasetJobService) -> None:
        asyncio.run(jobs.try_acquire("ds-a", "captioning", "a"))
        # A different dataset is free even for the same kind.
        asyncio.run(jobs.try_acquire("ds-b", "captioning", "b"))
        assert jobs.is_busy("ds-a") and jobs.is_busy("ds-b")


class TestBusyError:
    def test_cross_kind_conflict(self, jobs: DatasetJobService) -> None:
        """A tagging claim is blocked by a captioning claim (the whole point)."""
        asyncio.run(jobs.try_acquire("ds", "captioning", "cap-1"))
        with pytest.raises(DatasetBusyError) as exc_info:
            asyncio.run(jobs.try_acquire("ds", "tagging", "tag-1"))
        err = exc_info.value
        assert err.held_by.kind == "captioning"
        assert err.held_by.job_id == "cap-1"
        assert "captioning" in str(err)
        assert "ds" in str(err)

    def test_same_kind_conflict(self, jobs: DatasetJobService) -> None:
        asyncio.run(jobs.try_acquire("ds", "tagging", "tag-1"))
        with pytest.raises(DatasetBusyError):
            asyncio.run(jobs.try_acquire("ds", "tagging", "tag-2"))

    def test_busy_error_is_value_error(self, jobs: DatasetJobService) -> None:
        """Controllers map ValueError → 409; the subclass must satisfy that."""
        asyncio.run(jobs.try_acquire("ds", "captioning", "a"))
        with pytest.raises(ValueError):
            asyncio.run(jobs.try_acquire("ds", "tagging", "b"))


class TestStaleRelease:
    def test_wrong_claim_is_noop(self, jobs: DatasetJobService) -> None:
        """A late cleanup releasing a stale handle can't evict a newer owner."""
        old = asyncio.run(jobs.try_acquire("ds", "captioning", "old"))
        # New owner takes over (simulating old job already released + reacquired).
        asyncio.run(jobs.release(old))
        new = asyncio.run(jobs.try_acquire("ds", "tagging", "new"))
        # The old handle's release must not drop the new owner.
        asyncio.run(jobs.release(old))
        assert jobs.current("ds") is new

    def test_release_unknown_dataset_is_noop(self, jobs: DatasetJobService) -> None:
        from yadc.api.services.dataset_jobs import JobClaim

        asyncio.run(jobs.release(JobClaim(dataset_name="never", kind="captioning", job_id="x")))


# ---------------------------------------------------------------------------
# Mutual exclusion under concurrency
# ---------------------------------------------------------------------------


class TestConcurrency:
    def test_two_concurrent_starts_one_wins(self, jobs: DatasetJobService) -> None:
        """Two cross-kind acquires racing on the same dataset: exactly one wins."""

        async def race() -> tuple[bool, bool]:
            async def attempt(kind: str):
                try:
                    await jobs.try_acquire("ds", kind, f"{kind}-1")
                    return True
                except DatasetBusyError:
                    return False

            cap_ok, tag_ok = await asyncio.gather(attempt("captioning"), attempt("tagging"))
            return cap_ok, tag_ok

        cap_ok, tag_ok = asyncio.run(race())
        assert cap_ok != tag_ok, "exactly one of the two racers must win"
        assert jobs.is_busy("ds")


# ---------------------------------------------------------------------------
# Cross-service integration (real services, shared coordinator)
# ---------------------------------------------------------------------------


@pytest.fixture
def shared_jobs(logging_factory: LoggingFactory) -> DatasetJobService:
    return DatasetJobService(logging=logging_factory)


def _make_tagging(
    shared_jobs: DatasetJobService,
    test_configuration: Configuration,
    db_connection_factory: DBConnectionFactory,
    logging_factory: LoggingFactory,
) -> object:
    from yadc.api.services.settings import SettingsService
    from yadc.api.services.settings_repository import SettingsRepository
    from yadc.api.services.tagging import TaggingService

    test_configuration.tagger_model_path = "/fake/model.onnx"
    repo = SettingsRepository(db_connection_factory, logging_factory)
    settings_service = SettingsService(db_connection_factory, logging_factory, repo)
    return TaggingService(
        configuration=test_configuration,
        logging=MagicMock(),
        event_dispatcher=MagicMock(),
        dataset_service=MagicMock(),
        dataset_watcher=MagicMock(),
        dataset_jobs=shared_jobs,
        job_scheduler=None,
        settings_service=settings_service,
    )


class TestCrossServiceExclusion:
    def test_captioning_claim_blocks_tagging(
        self,
        shared_jobs: DatasetJobService,
        test_configuration: Configuration,
        db_connection_factory: DBConnectionFactory,
        logging_factory: LoggingFactory,
    ) -> None:
        """If the coordinator already holds a captioning claim, a tagging job start raises."""
        tagging = _make_tagging(shared_jobs, test_configuration, db_connection_factory, logging_factory)

        async def run() -> None:
            await shared_jobs.try_acquire("ds", "captioning", "cap-1")
            with pytest.raises(DatasetBusyError):
                # Directly exercising the claim gate the service uses.
                await tagging._dataset_jobs.try_acquire("ds", "tagging", "tag-1")  # pyright: ignore[reportAttributeAccessIssue]

        asyncio.run(run())

    def test_tagging_holding_blocks_captioning_claim(self, shared_jobs: DatasetJobService, test_configuration: Configuration) -> None:
        asyncio.run(shared_jobs.try_acquire("ds", "tagging", "tag-1"))
        with pytest.raises(DatasetBusyError):
            asyncio.run(shared_jobs.try_acquire("ds", "captioning", "cap-1"))

    def test_single_tag_releases_claim_on_success(
        self,
        shared_jobs: DatasetJobService,
        test_configuration: Configuration,
        db_connection_factory: DBConnectionFactory,
        logging_factory: LoggingFactory,
    ) -> None:
        """The single-image tag wrapper releases its transient claim once done."""
        tagging = _make_tagging(shared_jobs, test_configuration, db_connection_factory, logging_factory)
        from yadc.api.services.dataset_repository import ImageInfo
        from yadc.taggers.base import TaggerResult

        async def fake_tag(*a, **k):
            return TaggerResult(tags={}, categories={})

        tagging.tag_image = fake_tag  # type: ignore[method-assign]
        info = ImageInfo(id=1, file_name="x.png", path="/x.png")

        async def run() -> None:
            await tagging.tag_single_image_async("ds", info)
            assert not shared_jobs.is_busy("ds")

        asyncio.run(run())

    def test_single_tag_409_when_batch_holds(
        self,
        shared_jobs: DatasetJobService,
        test_configuration: Configuration,
        db_connection_factory: DBConnectionFactory,
        logging_factory: LoggingFactory,
    ) -> None:
        """A held claim makes the single-image wrapper raise DatasetBusyError."""
        tagging = _make_tagging(shared_jobs, test_configuration, db_connection_factory, logging_factory)
        asyncio.run(shared_jobs.try_acquire("ds", "captioning", "cap-1"))

        async def run() -> None:
            from yadc.api.services.dataset_repository import ImageInfo

            info = ImageInfo(id=1, file_name="x.png", path="/x.png")
            with pytest.raises(DatasetBusyError):
                await tagging.tag_single_image_async("ds", info)

        asyncio.run(run())
