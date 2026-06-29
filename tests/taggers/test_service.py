"""Tests for ``TaggingService`` — lazy spawn, idle timeout, and lifecycle.

The subprocess layer (``TaggerClient`` / ``TaggerServer``) is mocked so
these tests exercise only the service's state machine and lock
interactions. The ``TaggerServer`` subprocess is covered by
``test_server.py``.

The service trusts its inputs (an ``ImageInfo`` that's already been
validated by the caller), so the "missing image" / "image not found"
checks are tested at the controller level.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from yadc.api.modules import EventDispatcher, JobScheduler, LoggingFactory
from yadc.api.services import tagging as tagging_module
from yadc.api.services.dataset_jobs import DatasetJobService
from yadc.api.services.dataset_repository import ImageInfo
from yadc.api.services.tagging import (
    TaggerResultKey,
    TaggingService,
    TaggingThresholds,
    TagJobOptions,
    TagSaveOptions,
)
from yadc.taggers.base import TaggerResult

from .conftest import make_client_mock, patch_client_factory

# ---------------------------------------------------------------------------
# Fixtures (job_scheduler + service variants). Image fixtures and the
# subprocess client / patcher come from the directory's conftest.py.
# ---------------------------------------------------------------------------


@pytest.fixture
def job_scheduler() -> MagicMock:
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
) -> TaggingService:
    """A ``TaggingService`` configured for a local-path tagger model with a 60s idle timeout."""
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
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def configure_dataset_service(dataset_service: MagicMock, images: list[ImageInfo]) -> None:
    """Wire ``dataset_service`` so job resolution finds *images*.

    Sets ``get_image`` to look up by id and ``list_images`` to return a
    single page containing *images*. ``_has_tags_table`` falls through
    to the default (MagicMock returns ``False`` for ``"tags" in mock``),
    which is what batch tests want unless they override it explicitly.
    """
    dataset_service.get_image = MagicMock(side_effect=lambda ds, i: next((im for im in images if im.id == i), None))

    class _Page:
        def __init__(self, imgs: list[ImageInfo]) -> None:
            self.images = imgs
            self.next_token = None

    dataset_service.list_images = MagicMock(return_value=_Page(images))


# ---------------------------------------------------------------------------
# Configuration / property behavior
# ---------------------------------------------------------------------------


class TestConfiguration:
    def test_is_configured_reflects_model_path(self, test_configuration, service):  # noqa: ANN001
        """``is_configured`` is True iff a non-empty model path or repo_id is set."""
        test_configuration.tagger_model_path = ""
        test_configuration.tagger_repo_id = ""
        assert service.is_configured is False

        test_configuration.tagger_model_path = "  "
        test_configuration.tagger_repo_id = ""
        assert service.is_configured is False

        test_configuration.tagger_model_path = "/some/model.onnx"
        test_configuration.tagger_repo_id = ""
        assert service.is_configured is True

        test_configuration.tagger_model_path = ""
        test_configuration.tagger_repo_id = "SmilingWolf/wd-v1-4-vit-tagger-v2"
        assert service.is_configured is True

    def test_is_available_false_until_spawned(self, service: TaggingService) -> None:
        """``is_available`` is False when the subprocess hasn't been spawned yet."""
        assert service.is_available is False


# ---------------------------------------------------------------------------
# Startup wiring
# ---------------------------------------------------------------------------


class TestStartup:
    def test_startup_schedules_idle_check(self, test_configuration, service, job_scheduler):  # noqa: ANN001
        """``on_startup`` schedules the idle check job when a JobScheduler is provided."""
        test_configuration.tagger_model_path = "/fake/model.onnx"

        asyncio.run(service.on_startup(None))

        job_scheduler.new_scheduled_job.assert_called_once()
        args, _ = job_scheduler.new_scheduled_job.call_args
        assert args[1] == service._idle_check_tick

    def test_startup_without_scheduler_does_not_crash(self, test_configuration, service):  # noqa: ANN001
        """``on_startup`` is a no-op for the schedule side when no JobScheduler is injected."""
        test_configuration.tagger_model_path = "/fake/model.onnx"
        asyncio.run(service.on_startup(None))  # should not raise


# ---------------------------------------------------------------------------
# Lazy spawn on first request
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_first_request_spawns_subprocess(self, service: TaggingService, image_info: ImageInfo) -> None:
        """The first tagging request triggers a lazy spawn; ``is_available`` flips on."""
        client = make_client_mock(alive=True)
        with patch_client_factory(client):
            asyncio.run(service.tag_image("ds", image_info))

        client.start.assert_awaited_once()
        client.tag.assert_awaited_once()
        assert service.is_available is True

    def test_subsequent_requests_reuse_subprocess(self, service: TaggingService, make_image_info) -> None:
        """Two requests in a row reuse the same subprocess; only one ``start()``."""
        client = make_client_mock(alive=True)
        with patch_client_factory(client):
            asyncio.run(service.tag_image("ds", make_image_info()))
            asyncio.run(service.tag_image("ds", make_image_info()))

        client.start.assert_awaited_once()
        assert client.tag.await_count == 2

    def test_request_with_unconfigured_model_raises(self, test_configuration, service, image_info: ImageInfo):  # noqa: ANN001
        """Requests raise ``RuntimeError`` when no model path or repo is configured."""
        test_configuration.tagger_model_path = ""
        test_configuration.tagger_repo_id = ""
        with pytest.raises(RuntimeError, match="not configured"):
            asyncio.run(service.tag_image("ds", image_info))


# ---------------------------------------------------------------------------
# Underscore replacement (post-threshold postprocessing)
# ---------------------------------------------------------------------------


class TestReplaceUnderscores:
    def test_disabled_by_default_preserves_raw_output(self, service: TaggingService, image_info: ImageInfo) -> None:
        """With the config flag off (the default), tags pass through untouched."""
        client = make_client_mock(alive=True)
        client.tag = AsyncMock(
            return_value=TaggerResult(
                tags={"long_hair": 0.9, "1girl": 0.99},
                categories={"general": ["long_hair", "1girl"]},
            )
        )
        with patch_client_factory(client):
            out = asyncio.run(service.tag_image("ds", image_info))
        assert out.tags == {"long_hair": 0.9, "1girl": 0.99}

    def test_enabled_via_param_converts_underscores(self, service: TaggingService, image_info: ImageInfo) -> None:
        """``replace_underscores=True`` converts underscores to spaces in both
        tags and category lists, preserving kaomojis."""
        client = make_client_mock(alive=True)
        client.tag = AsyncMock(
            return_value=TaggerResult(
                tags={"long_hair": 0.9, "1girl": 0.99, "^_^": 0.7},
                categories={"general": ["long_hair", "1girl", "^_^"]},
            )
        )
        with patch_client_factory(client):
            out = asyncio.run(service.tag_image("ds", image_info, replace_underscores=True))
        assert out.tags == {"long hair": 0.9, "1girl": 0.99, "^_^": 0.7}
        assert out.categories == {"general": ["long hair", "1girl", "^_^"]}

    def test_param_false_overrides_config_true(self, test_configuration, service: TaggingService, image_info: ImageInfo) -> None:
        """An explicit ``False`` wins over a ``True`` server config (per-request
        override semantics, mirroring thresholds)."""
        test_configuration.tagger_replace_underscores = True
        client = make_client_mock(alive=True)
        client.tag = AsyncMock(return_value=TaggerResult(tags={"long_hair": 0.9}, categories={"general": ["long_hair"]}))
        with patch_client_factory(client):
            out = asyncio.run(service.tag_image("ds", image_info, replace_underscores=False))
        assert out.tags == {"long_hair": 0.9}

    def test_config_true_applies_when_param_unset(self, test_configuration, service: TaggingService, image_info: ImageInfo) -> None:
        """A ``True`` config applies when no per-request override is given."""
        test_configuration.tagger_replace_underscores = True
        client = make_client_mock(alive=True)
        client.tag = AsyncMock(return_value=TaggerResult(tags={"long_hair": 0.9}, categories={"general": ["long_hair"]}))
        with patch_client_factory(client):
            out = asyncio.run(service.tag_image("ds", image_info))
        assert out.tags == {"long hair": 0.9}


# ---------------------------------------------------------------------------
# Re-spawn after the subprocess dies
# ---------------------------------------------------------------------------


class TestRespawn:
    def test_respawn_when_subprocess_dies(self, service: TaggingService, make_image_info) -> None:
        """If the subprocess is no longer alive, the next request spawns a fresh one."""
        # First request: client is alive.
        client1 = make_client_mock(alive=True)
        with patch_client_factory(client1):
            asyncio.run(service.tag_image("ds", make_image_info()))

        # Simulate the subprocess dying between requests.
        client1.is_alive = False

        # Next request: spawn a fresh client.
        client2 = make_client_mock(alive=True)
        with patch_client_factory(client2):
            asyncio.run(service.tag_image("ds", make_image_info()))

        client1.start.assert_awaited_once()
        client2.start.assert_awaited_once()


# ---------------------------------------------------------------------------
# Idle timeout
# ---------------------------------------------------------------------------


class TestIdleTimeout:
    def test_idle_check_stops_when_idle(
        self,
        service: TaggingService,
        image_info: ImageInfo,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Idle check stops the subprocess when ``last_used`` is older than the timeout."""
        client = make_client_mock(alive=True)
        with patch_client_factory(client):
            asyncio.run(service.tag_image("ds", image_info))
            last_used = service._last_used_t
            assert last_used is not None

            # Fake time advancing past the timeout.
            monkeypatch.setattr(time, "monotonic", lambda: last_used + service._configuration.tagger_idle_timeout_seconds + 1)
            service._idle_check_tick()

        client._server.stop.assert_called_once()
        assert service._tagger_client is None
        assert service._last_used_t is None

    def test_idle_check_skips_when_recently_used(
        self,
        service: TaggingService,
        image_info: ImageInfo,
    ) -> None:
        """Idle check is a no-op when the server was used recently."""
        client = make_client_mock(alive=True)
        with patch_client_factory(client):
            asyncio.run(service.tag_image("ds", image_info))
            service._idle_check_tick()

        client._server.stop.assert_not_called()
        assert service.is_available is True

    def test_idle_check_disabled_with_zero_timeout(
        self,
        test_configuration,
        service: TaggingService,
        image_info: ImageInfo,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``tagger_idle_timeout_seconds <= 0`` disables teardown entirely."""
        test_configuration.tagger_model_path = "/fake/model.onnx"
        test_configuration.tagger_idle_timeout_seconds = 0.0

        client = make_client_mock(alive=True)
        with patch_client_factory(client):
            asyncio.run(service.tag_image("ds", image_info))
            # Pretend a huge amount of time has passed.
            monkeypatch.setattr(time, "monotonic", lambda: time.monotonic() + 10_000_000)
            service._idle_check_tick()

        client._server.stop.assert_not_called()
        assert service.is_available is True

    def test_idle_check_cleans_up_dead_client_without_alive_flag(
        self,
        service: TaggingService,
        image_info: ImageInfo,
    ) -> None:
        """If the subprocess died but the client object lingers, the idle tick clears state."""
        client = make_client_mock(alive=False)  # already dead
        with patch_client_factory(client):
            asyncio.run(service.tag_image("ds", image_info))
            # First request respawns via _ensure_running_locked, so let's
            # fake that the second spawn also died. Simulate by reaching
            # into the service directly:
            service._last_used_t = time.monotonic() - 10_000  # long ago
            service._idle_check_tick()

        # State is cleared; client.stop was not called (the client was
        # already dead, so we just drop the reference).
        assert service._tagger_client is None
        assert service._last_used_t is None
        client._server.stop.assert_not_called()

    def test_respawn_after_idle_stop(self, service: TaggingService, make_image_info) -> None:
        """A request after idle teardown spawns a fresh subprocess."""
        client1 = make_client_mock(alive=True)
        with patch_client_factory(client1):
            asyncio.run(service.tag_image("ds", make_image_info()))
            # Simulate idle teardown.
            service._last_used_t = time.monotonic() - service._configuration.tagger_idle_timeout_seconds - 1
            service._idle_check_tick()
            assert service._tagger_client is None

            client2 = make_client_mock(alive=True)
            with patch_client_factory(client2):
                asyncio.run(service.tag_image("ds", make_image_info()))

        client1.start.assert_awaited_once()
        client2.start.assert_awaited_once()


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------


class TestShutdown:
    def test_shutdown_stops_subprocess(self, service: TaggingService, image_info: ImageInfo) -> None:
        """``on_shutdown`` tears down a running subprocess."""
        client = make_client_mock(alive=True)
        with patch_client_factory(client):
            asyncio.run(service.tag_image("ds", image_info))
            asyncio.run(service.on_shutdown(None))

        client.stop.assert_awaited_once()
        assert service._tagger_client is None

    def test_shutdown_when_not_running_is_noop(self, service: TaggingService) -> None:
        """``on_shutdown`` is a no-op when no subprocess has been spawned."""
        asyncio.run(service.on_shutdown(None))  # should not raise
        assert service._tagger_client is None


# ---------------------------------------------------------------------------
# HuggingFace repo_id passthrough
# ---------------------------------------------------------------------------


class TestHF:
    def test_repo_id_passed_through_to_tagger(
        self,
        test_configuration,
        service: TaggingService,
        image_info: ImageInfo,
    ) -> None:
        """When ``tagger_repo_id`` is set, the service passes ``repo_id`` + filenames in ``TaggerClient`` kwargs."""
        test_configuration.tagger_model_path = ""
        test_configuration.tagger_repo_id = "SmilingWolf/wd-v1-4-vit-tagger-v2"
        test_configuration.tagger_repo_model_filename = "model.onnx"
        test_configuration.tagger_repo_label_filename = "selected_tags.csv"

        client = make_client_mock(alive=True)
        with patch.object(tagging_module, "TaggerClient", return_value=client) as mock_cls:
            asyncio.run(service.tag_image("ds", image_info))

        # Inspect the kwargs the service passed to ``TaggerClient`` — the
        # third positional arg is the tagger_kwargs dict that the worker
        # forwards to the ``OnnxTagger`` constructor.
        assert mock_cls.call_count == 1
        tagger_kwargs = mock_cls.call_args.args[2] or {}
        assert tagger_kwargs.get("repo_id") == "SmilingWolf/wd-v1-4-vit-tagger-v2"
        assert tagger_kwargs.get("repo_model_filename") == "model.onnx"
        assert tagger_kwargs.get("repo_label_filename") == "selected_tags.csv"
        # Local label path is not used in the repo_id flow.
        assert "label_path" not in tagger_kwargs

    def test_local_path_used_when_repo_id_empty(self, service: TaggingService, image_info: ImageInfo) -> None:
        """When ``tagger_repo_id`` is empty, the service uses local-path flow (label auto-discovery)."""
        # Force empty repo + non-empty model path so the local-path branch is taken.
        service._configuration.tagger_repo_id = ""
        service._configuration.tagger_model_path = "/fake/model.onnx"
        assert service._configuration.tagger_repo_id == ""

        client = make_client_mock(alive=True)
        with patch_client_factory(client):
            asyncio.run(service.tag_image("ds", image_info))

        client.start.assert_awaited_once()
        assert service.is_available is True


# ---------------------------------------------------------------------------
# Concurrent requests
# ---------------------------------------------------------------------------


class TestConcurrent:
    def test_concurrent_requests_share_single_spawn(self, service: TaggingService, make_image_info) -> None:
        """Concurrent requests serialize through the lock; only one spawn happens."""
        client = make_client_mock(alive=True)
        with patch_client_factory(client):
            # ``tag()`` blocks the lock briefly while "running"; if the
            # lock isn't held across the await, the second request would
            # see a non-alive client and respawn. So this also verifies
            # the lock-held-during-await behavior.
            async def tag_twice() -> None:
                await asyncio.gather(
                    service.tag_image("ds", make_image_info()),
                    service.tag_image("ds", make_image_info()),
                    service.tag_image("ds", make_image_info()),
                )

            asyncio.run(tag_twice())

        client.start.assert_awaited_once()
        assert client.tag.await_count == 3


# ---------------------------------------------------------------------------
# Batch tagging jobs
# ---------------------------------------------------------------------------


class TestBatchJob:
    """Tests for the batch tagging job (start / status / stop / save)."""

    def test_start_job_tags_all_images_and_saves_draft(
        self,
        service: TaggingService,
        make_image_info,
        dataset_service: MagicMock,
    ) -> None:
        """A job with no ``image_ids`` tags every image and writes a draft per image."""
        images = [make_image_info(), make_image_info()]
        configure_dataset_service(dataset_service, images)
        dataset_service.write_draft = MagicMock(return_value=True)

        client = make_client_mock(alive=True)

        opts = TagJobOptions(save=TagSaveOptions(mode="draft", draft_name="tags", draft_format="comma"))

        async def runner() -> None:
            with patch_client_factory(client):
                info = await service.start_tag_job_async("ds", opts)
                await service._tag_jobs["ds"].task  # type: ignore[arg-type]

            assert info.total == 2
            assert client.tag.await_count == 2
            # Each image got a draft write with comma-formatted text.
            assert dataset_service.write_draft.call_count == 2
            first_call = dataset_service.write_draft.call_args_list[0]
            assert first_call.args[2] == "tags"  # draft_name
            # Comma formatter excludes rating tags → "1girl" only (the mock result).
            assert "1girl" in first_call.args[3]

            final = await service.get_tag_job_status_async("ds")
            assert final.status == "done"
            assert final.processed == 2
            assert final.errors == 0

        asyncio.run(runner())

    def test_start_job_with_image_ids_tags_only_those(
        self,
        service: TaggingService,
        make_image_info,
        dataset_service: MagicMock,
    ) -> None:
        """``image_ids`` restricts the job to the listed images."""
        images = [make_image_info(), make_image_info(), make_image_info()]
        configure_dataset_service(dataset_service, images)

        client = make_client_mock(alive=True)

        opts = TagJobOptions(image_ids=[images[0].id, images[2].id])

        async def runner() -> None:
            with patch_client_factory(client):
                info = await service.start_tag_job_async("ds", opts)
                await service._tag_jobs["ds"].task  # type: ignore[arg-type]

            assert info.total == 2
            assert client.tag.await_count == 2

        asyncio.run(runner())

    def test_start_job_extras_mode_calls_merge_extras_tags(
        self,
        service: TaggingService,
        make_image_info,
        dataset_service: MagicMock,
    ) -> None:
        """``save.mode='extras'`` writes the ``[tags]`` sub-table via the dataset service."""
        images = [make_image_info()]
        configure_dataset_service(dataset_service, images)
        service._merge_extras_tags = MagicMock(return_value=True)

        client = make_client_mock(alive=True)

        opts = TagJobOptions(save=TagSaveOptions(mode="extras"))

        async def runner() -> None:
            with patch_client_factory(client):
                await service.start_tag_job_async("ds", opts)
                await service._tag_jobs["ds"].task  # type: ignore[arg-type]

            service._merge_extras_tags.assert_called_once()
            tags_arg = service._merge_extras_tags.call_args.args[2]
            assert tags_arg["general"] == ["1girl"]
            assert tags_arg["rating"] == "safe"

        asyncio.run(runner())

    def test_overwrite_false_skips_images_with_existing_draft(
        self,
        service: TaggingService,
        make_image_info,
        dataset_service: MagicMock,
    ) -> None:
        """``overwrite=False`` skips images already carrying the target draft — no tag, no save."""
        has_draft = make_image_info()
        has_draft.draft_names = ["tags"]
        fresh = make_image_info()
        images = [has_draft, fresh]
        configure_dataset_service(dataset_service, images)
        dataset_service.write_draft = MagicMock(return_value=True)

        client = make_client_mock(alive=True)

        opts = TagJobOptions(save=TagSaveOptions(mode="draft", draft_name="tags"))

        async def runner() -> None:
            with patch_client_factory(client):
                await service.start_tag_job_async("ds", opts)
                await service._tag_jobs["ds"].task  # type: ignore[arg-type]

            # Only the fresh image reached the tagger + write path.
            assert client.tag.await_count == 1
            assert dataset_service.write_draft.call_count == 1
            assert dataset_service.write_draft.call_args.args[2] == "tags"
            # Both count as processed so the progress bar still completes.
            final = await service.get_tag_job_status_async("ds")
            assert final.status == "done"
            assert final.processed == 2
            assert final.errors == 0

        asyncio.run(runner())

    def test_overwrite_false_all_skipped_returns_done_without_task(
        self,
        service: TaggingService,
        make_image_info,
        dataset_service: MagicMock,
    ) -> None:
        """When every image is skip-eligible, start returns ``done`` with no background task.

        This is the race fix: the HTTP response itself carries ``done``, so
        no SSE event can arrive before the response and clobber it back to
        ``running``. No task is spawned, no state is stored, and the claim is
        released immediately.
        """
        has_draft = make_image_info()
        has_draft.draft_names = ["tags"]
        has_draft2 = make_image_info()
        has_draft2.draft_names = ["tags"]
        configure_dataset_service(dataset_service, [has_draft, has_draft2])

        client = make_client_mock(alive=True)

        opts = TagJobOptions(save=TagSaveOptions(mode="draft", draft_name="tags"))

        async def runner() -> None:
            with patch_client_factory(client):
                info = await service.start_tag_job_async("ds", opts)

            # The response itself is ``done`` — no background task.
            assert info.status == "done"
            assert info.total == 2
            assert info.processed == 2
            assert info.errors == 0
            assert "ds" not in service._tag_jobs
            assert client.tag.await_count == 0

        asyncio.run(runner())

    def test_overwrite_true_tags_all_despite_existing_draft(
        self,
        service: TaggingService,
        make_image_info,
        dataset_service: MagicMock,
    ) -> None:
        """``overwrite=True`` bypasses the skip: every image is tagged and rewritten."""
        has_draft = make_image_info()
        has_draft.draft_names = ["tags"]
        fresh = make_image_info()
        images = [has_draft, fresh]
        configure_dataset_service(dataset_service, images)
        dataset_service.write_draft = MagicMock(return_value=True)

        client = make_client_mock(alive=True)

        opts = TagJobOptions(save=TagSaveOptions(mode="draft", draft_name="tags", overwrite=True))

        async def runner() -> None:
            with patch_client_factory(client):
                await service.start_tag_job_async("ds", opts)
                await service._tag_jobs["ds"].task  # type: ignore[arg-type]

            assert client.tag.await_count == 2
            assert dataset_service.write_draft.call_count == 2

        asyncio.run(runner())

    def test_overwrite_false_skips_extras_when_tags_table_present(
        self,
        service: TaggingService,
        make_image_info,
        dataset_service: MagicMock,
    ) -> None:
        """Extras mode skips images whose TOML already has a ``[tags]`` sub-table."""
        images = [make_image_info(), make_image_info()]
        configure_dataset_service(dataset_service, images)
        # First image already has [tags]; second doesn't.
        service._has_tags_table = MagicMock(side_effect=lambda _ds, i: i == images[0].id)
        service._merge_extras_tags = MagicMock(return_value=True)

        client = make_client_mock(alive=True)

        opts = TagJobOptions(save=TagSaveOptions(mode="extras"))

        async def runner() -> None:
            with patch_client_factory(client):
                await service.start_tag_job_async("ds", opts)
                await service._tag_jobs["ds"].task  # type: ignore[arg-type]

            assert client.tag.await_count == 1
            assert service._merge_extras_tags.call_count == 1
            assert service._merge_extras_tags.call_args.args[1] == images[1].id

        asyncio.run(runner())

    def test_second_job_rejected_while_one_running(
        self,
        service: TaggingService,
        make_image_info,
        dataset_service: MagicMock,
    ) -> None:
        """Starting a job while one is running raises ValueError (→ 409)."""
        images = [make_image_info(), make_image_info()]
        configure_dataset_service(dataset_service, images)

        # Make tag() block so the first job stays running.
        slow_event = asyncio.Event()

        async def _slow_tag(_bytes: bytes) -> TaggerResult:
            await slow_event.wait()
            return TaggerResult(tags={"1girl": 0.99}, categories={"general": ["1girl"]})

        client = make_client_mock(alive=True)
        client.tag = _slow_tag

        async def runner() -> None:
            with patch_client_factory(client):
                await service.start_tag_job_async("ds", TagJobOptions())
                with pytest.raises(ValueError, match="already running"):
                    await service.start_tag_job_async("ds", TagJobOptions())
                # Unblock and finish.
                slow_event.set()
                await service._tag_jobs["ds"].task  # type: ignore[arg-type]

        asyncio.run(runner())

    def test_stop_job_cancels_run(
        self,
        service: TaggingService,
        make_image_info,
        dataset_service: MagicMock,
    ) -> None:
        """``stop_tag_job_async`` sets the stop flag; the job ends as 'cancelled'."""
        images = [make_image_info(), make_image_info()]
        configure_dataset_service(dataset_service, images)

        started = asyncio.Event()

        async def _blocking_tag(_bytes: bytes) -> TaggerResult:
            started.set()
            # Block until cancelled. The stop path doesn't await the task,
            # so we rely on the task being cancelled — but our loop checks
            # ``stop_event`` between images, so we yield once and return.
            await asyncio.sleep(0)
            return TaggerResult(tags={"1girl": 0.99}, categories={"general": ["1girl"]})

        client = make_client_mock(alive=True)
        client.tag = _blocking_tag

        async def runner() -> None:
            with patch_client_factory(client):
                await service.start_tag_job_async("ds", TagJobOptions())
                await started.wait()
                stopped = await service.stop_tag_job_async("ds")
                assert stopped is True
                await service._tag_jobs["ds"].task  # type: ignore[arg-type]

        asyncio.run(runner())
        final = asyncio.run(service.get_tag_job_status_async("ds"))
        assert final.status == "cancelled"

    def test_save_tags_none_mode_writes_nothing(
        self,
        service: TaggingService,
        make_image_info,
        dataset_service: MagicMock,
    ) -> None:
        """``save.mode='none'`` tags but performs no write."""
        images = [make_image_info()]
        configure_dataset_service(dataset_service, images)
        dataset_service.write_draft = MagicMock()
        service._merge_extras_tags = MagicMock()

        client = make_client_mock(alive=True)

        opts = TagJobOptions(save=TagSaveOptions(mode="none"))

        async def runner() -> None:
            with patch_client_factory(client):
                await service.start_tag_job_async("ds", opts)
                await service._tag_jobs["ds"].task  # type: ignore[arg-type]

            dataset_service.write_draft.assert_not_called()
            service._merge_extras_tags.assert_not_called()

        asyncio.run(runner())

    def test_tag_failure_counts_as_error_but_run_continues(
        self,
        service: TaggingService,
        make_image_info,
        dataset_service: MagicMock,
    ) -> None:
        """A failing image is counted as an error; the rest of the run proceeds."""
        images = [make_image_info(), make_image_info()]
        configure_dataset_service(dataset_service, images)

        call_count = 0

        async def _flaky_tag(_bytes: bytes) -> TaggerResult:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("boom")
            return TaggerResult(tags={"1girl": 0.99}, categories={"general": ["1girl"]})

        client = make_client_mock(alive=True)
        client.tag = _flaky_tag

        async def runner() -> None:
            with patch_client_factory(client):
                await service.start_tag_job_async("ds", TagJobOptions())
                await service._tag_jobs["ds"].task  # type: ignore[arg-type]

            final = await service.get_tag_job_status_async("ds")
            assert final.status == "done"
            assert final.errors == 1
            assert final.processed == 1

        asyncio.run(runner())


# ---------------------------------------------------------------------------
# preview_save_tags
# ---------------------------------------------------------------------------


class TestPreviewSaveTags:
    """``preview_save_tags`` formats without touching disk."""

    def test_preview_none_returns_empty(self, service: TaggingService) -> None:

        result = TaggerResult(
            tags={"1girl": 0.99, "safe": 0.99},
            categories={"general": ["1girl"], "rating": ["safe"]},
        )
        out = service.preview_save_tags(result, TagSaveOptions(mode="none"))
        assert out == ""

    def test_preview_draft_comma_excludes_rating(self, service: TaggingService) -> None:

        result = TaggerResult(
            tags={"1girl": 0.99, "safe": 0.99, "solo": 0.8},
            categories={"general": ["1girl", "solo"], "rating": ["safe"]},
        )
        out = service.preview_save_tags(result, TagSaveOptions(mode="draft", draft_format="comma"))
        # Rating is excluded from the training caption.
        assert "1girl" in out and "solo" in out
        assert "safe" not in out

    def test_preview_draft_structured_includes_categories(self, service: TaggingService) -> None:

        result = TaggerResult(
            tags={"1girl": 0.99, "safe": 0.9, "hatsune miku": 0.7},
            categories={"general": ["1girl"], "character": ["hatsune miku"], "rating": ["safe"]},
        )
        out = service.preview_save_tags(result, TagSaveOptions(mode="draft", draft_format="structured"))
        assert "rating: safe" in out
        assert "general: 1girl" in out
        assert "character: hatsune miku" in out

    def test_preview_extras_renders_toml_subtable(self, service: TaggingService) -> None:

        result = TaggerResult(
            tags={"1girl": 0.99, "safe": 0.9, "hatsune miku": 0.7},
            categories={"general": ["1girl"], "character": ["hatsune miku"], "rating": ["safe"]},
        )
        out = service.preview_save_tags(result, TagSaveOptions(mode="extras"))
        # Single top-rating string, not a list.
        assert "[tags]" in out
        assert 'rating = "safe"' in out
        assert "1girl" in out

    def test_preview_extras_omits_rating_when_none_survived(self, service: TaggingService) -> None:

        result = TaggerResult(
            tags={"1girl": 0.99},
            categories={"general": ["1girl"]},
        )
        out = service.preview_save_tags(result, TagSaveOptions(mode="extras"))
        assert "[tags]" in out
        assert "rating" not in out


# ---------------------------------------------------------------------------
# Cancel (graceful-first, kill-as-fallback)
# ---------------------------------------------------------------------------


class TestCancel:
    """``cancel_async`` — graceful stop_event, stale-job guard, kill escalation."""

    def test_cancel_with_nothing_running(self, service: TaggingService) -> None:
        """No job, no in-flight tag (lock free) → nothing_running, no kill."""
        result = asyncio.run(service.cancel_async())
        assert result.outcome == "nothing_running"
        assert result.killed is False

    def test_cancel_stale_job_id(self, service: TaggingService) -> None:
        """An unknown job_id → stale_job (refuse to cancel something unknown)."""
        result = asyncio.run(service.cancel_async("does-not-exist"))
        assert result.outcome == "stale_job"

    def test_cancel_running_job_sets_stop_event(
        self,
        service: TaggingService,
        make_image_info,
        dataset_service: MagicMock,
    ) -> None:
        """A matching job_id with the lock free → stopped (graceful stop_event)."""

        images = [make_image_info(), make_image_info(), make_image_info()]
        configure_dataset_service(dataset_service, images)

        started = asyncio.Event()

        async def _yielding_tag(_bytes: bytes) -> TaggerResult:
            started.set()
            await asyncio.sleep(0)
            return TaggerResult(tags={"1girl": 0.99}, categories={"general": ["1girl"]})

        client = make_client_mock(alive=True)
        client.tag = _yielding_tag

        async def runner() -> None:
            with patch_client_factory(client):
                await service.start_tag_job_async("ds", TagJobOptions())
                await started.wait()
                # The job is running but between images (lock free here).
                job_id = service._tag_jobs["ds"].job_id
                result = await service.cancel_async(job_id)
                assert result.outcome == "stopped"
                assert result.killed is False
                assert result.job_id == job_id
                # stop_event was set.
                assert service._tag_jobs["ds"].stop_event.is_set() is True
                await service._tag_jobs["ds"].task  # type: ignore[arg-type]

        asyncio.run(runner())
        assert asyncio.run(service.get_tag_job_status_async("ds")).status == "cancelled"

    def test_cancel_kills_when_lock_held(self, service: TaggingService) -> None:
        """An in-flight tag (lock held) with no graceful completion → force-kill."""
        client = make_client_mock(alive=True)
        client.kill = AsyncMock()
        service._tagger_client = client
        # Grace window = 0 so cancel proceeds to kill without sleeping.
        service._configuration.tagger_cancel_grace_seconds = 0.0

        # Simulate an in-flight tag holding the lifecycle lock. The killed
        # worker's unwind would release it — model that by releasing on kill.
        assert service._lifecycle_lock.acquire(blocking=False) is True

        async def _release_on_kill() -> None:
            service._lifecycle_lock.release()

        client.kill.side_effect = _release_on_kill

        result = asyncio.run(service.cancel_async())
        assert result.outcome == "killed"
        assert result.killed is True
        client.kill.assert_awaited_once()
        # The dead client ref was cleared under the lock after the unwind.
        assert service._tagger_client is None

    def test_cancel_does_not_clobber_respawned_client(self, service: TaggingService) -> None:
        """If another request respawned a client during the kill window, cancel
        leaves it in place (doesn't clear the fresh ref)."""
        old_client = make_client_mock(alive=True)
        old_client.kill = AsyncMock()
        service._tagger_client = old_client
        service._configuration.tagger_cancel_grace_seconds = 0.0
        assert service._lifecycle_lock.acquire(blocking=False) is True

        new_client = make_client_mock(alive=True)

        async def _kill_then_respawn() -> None:
            # The killed worker's unwind releases the lock…
            service._lifecycle_lock.release()
            # …and a request arriving in the window respawns a fresh client.
            service._tagger_client = new_client

        old_client.kill.side_effect = _kill_then_respawn

        result = asyncio.run(service.cancel_async())
        assert result.outcome == "killed"
        # The fresh client is left intact (not cleared back to None).
        assert service._tagger_client is new_client


# ---------------------------------------------------------------------------
# Extras save path — orchestration over DatasetService
# ---------------------------------------------------------------------------


class TestMergeExtrasTags:
    """``TaggingService._merge_extras_tags`` — wires the [tags] sub-table through ``DatasetService``.

    Exercises the orchestration in isolation: ``DatasetService.get_image`` /
    ``load_extras`` / ``update_extras`` are mocked so the tagger's merge
    logic is the only thing under test. The on-disk sidecar behavior
    (round-trip, comments preservation, watcher suppression) lives in
    ``DatasetService.update_extras`` and is covered by
    ``tests/api/test_datasets_service.py::TestUpdateExtrasHistoryRoundTrip``.
    """

    def _stub_dataset_service(
        self,
        dataset_service: MagicMock,
        *,
        image_exists: bool,
        extras_doc,
        update_extras_returns: bool = True,
    ) -> MagicMock:
        """Wire the dataset service mocks to drive ``_merge_extras_tags``."""
        from yadc.api.services.dataset_repository import ImageInfo

        if image_exists:
            dataset_service.get_image = MagicMock(return_value=ImageInfo(id=1, file_name="img.jpg", path="/tmp/img.jpg", width=1, height=1))
        else:
            dataset_service.get_image = MagicMock(return_value=None)
        dataset_service.load_extras = MagicMock(return_value=extras_doc)
        dataset_service.update_extras = MagicMock(return_value=update_extras_returns)
        return dataset_service

    def test_merges_into_existing_extras_preserving_other_keys(
        self,
        service: TaggingService,
        dataset_service: MagicMock,
    ) -> None:
        """An existing extras doc with other keys survives the merge round-trip."""
        import tomlkit

        doc = tomlkit.loads('artist = "Monet"\nstyle = "impressionism"\n')
        self._stub_dataset_service(dataset_service, image_exists=True, extras_doc=doc)

        result = service._merge_extras_tags("ds", 1, {"general": ["1girl"], "rating": "general"})
        assert result is True

        # ``update_extras`` was called with the merged TOML text.
        dataset_service.update_extras.assert_called_once()
        merged_raw = dataset_service.update_extras.call_args.args[2]
        parsed = tomlkit.loads(merged_raw)
        # Other keys preserved.
        assert parsed["artist"] == "Monet"
        assert parsed["style"] == "impressionism"
        # ``[tags]`` sub-table written with the given shape.
        assert parsed["tags"]["general"] == ["1girl"]
        assert parsed["tags"]["rating"] == "general"

    def test_creates_sidecar_when_none_exists(
        self,
        service: TaggingService,
        dataset_service: MagicMock,
    ) -> None:
        """No sidecar → ``load_extras`` returns ``None`` → start from empty doc, write tags."""
        import tomlkit

        self._stub_dataset_service(dataset_service, image_exists=True, extras_doc=None)

        result = service._merge_extras_tags("ds", 1, {"general": ["cat"], "character": []})
        assert result is True

        dataset_service.update_extras.assert_called_once()
        parsed = tomlkit.loads(dataset_service.update_extras.call_args.args[2])
        assert parsed["tags"]["general"] == ["cat"]
        assert parsed["tags"]["character"] == []
        # No rating key when not provided.
        assert "rating" not in parsed["tags"]

    def test_overwrites_existing_tags_subtable(
        self,
        service: TaggingService,
        dataset_service: MagicMock,
    ) -> None:
        """An existing ``[tags]`` is fully replaced — not merged within."""
        import tomlkit

        doc = tomlkit.loads('artist = "Monet"\n[tags]\ngeneral = ["old"]\nrating = "explicit"\n')
        self._stub_dataset_service(dataset_service, image_exists=True, extras_doc=doc)

        service._merge_extras_tags("ds", 1, {"general": ["new"], "character": [], "rating": "general"})

        parsed = tomlkit.loads(dataset_service.update_extras.call_args.args[2])
        assert parsed["artist"] == "Monet"
        # ``tags`` is authoritative — fully replaced.
        assert parsed["tags"]["general"] == ["new"]
        assert parsed["tags"]["rating"] == "general"

    def test_returns_false_for_missing_image(
        self,
        service: TaggingService,
        dataset_service: MagicMock,
    ) -> None:
        """No image → returns ``False`` and never touches ``load_extras`` / ``update_extras``."""
        self._stub_dataset_service(dataset_service, image_exists=False, extras_doc=None)

        assert service._merge_extras_tags("ds", 99999, {"general": []}) is False
        dataset_service.load_extras.assert_not_called()
        dataset_service.update_extras.assert_not_called()


class TestHasTagsTable:
    """``TaggingService._has_tags_table`` — skip detection for the extras path."""

    def test_returns_true_when_tags_key_present(
        self,
        service: TaggingService,
        dataset_service: MagicMock,
    ) -> None:
        import tomlkit

        dataset_service.load_extras = MagicMock(return_value=tomlkit.loads('artist = "X"\n[tags]\ngeneral = ["1girl"]\n'))
        assert service._has_tags_table("ds", 1) is True

    def test_returns_false_when_tags_key_absent(
        self,
        service: TaggingService,
        dataset_service: MagicMock,
    ) -> None:
        import tomlkit

        dataset_service.load_extras = MagicMock(return_value=tomlkit.loads('artist = "X"\n'))
        assert service._has_tags_table("ds", 1) is False

    def test_returns_false_when_load_extras_returns_none(
        self,
        service: TaggingService,
        dataset_service: MagicMock,
    ) -> None:
        """Missing image, missing sidecar, or unparseable sidecar → ``False``."""
        dataset_service.load_extras = MagicMock(return_value=None)
        assert service._has_tags_table("ds", 1) is False


# ---------------------------------------------------------------------------
# Tagger result LRU cache
# ---------------------------------------------------------------------------


class TestTaggerResultCache:
    """``TaggingService._tag_results`` — LRU keyed by :class:`TaggerResultKey`.

    Verifies the cache write at the end of ``tag_image()`` and the
    read / evict paths. The subprocess client is mocked via the
    project's ``patch_client_factory`` helper so the assertion is on
    what actually gets cached, not on a synthesized key. ``asyncio.run``
    matches the surrounding test style (this project doesn't pull in
    pytest-asyncio).
    """

    def test_tag_image_populates_cache(
        self,
        service: TaggingService,
        image_info: ImageInfo,
    ) -> None:
        """A successful ``tag_image()`` writes the thresholded result to the LRU."""
        client = make_client_mock(alive=True)
        with patch_client_factory(client):
            result = asyncio.run(service.tag_image("ds", image_info))

        cached = asyncio.run(service.get_tag_result("ds", 1))
        assert cached is not None
        assert cached.tags == result.tags
        assert cached.categories == result.categories

    def test_get_tag_result_returns_none_when_absent(self, service: TaggingService) -> None:
        """An image never tagged (or evicted) → ``None``."""
        assert asyncio.run(service.get_tag_result("ds", 99999)) is None

    def test_evict_tag_result_removes_entry(
        self,
        service: TaggingService,
        image_info: ImageInfo,
    ) -> None:
        """``evict_tag_result`` returns ``True`` and the next ``get_tag_result`` is ``None``."""
        client = make_client_mock(alive=True)
        with patch_client_factory(client):
            asyncio.run(service.tag_image("ds", image_info))
        assert asyncio.run(service.get_tag_result("ds", 1)) is not None

        assert asyncio.run(service.evict_tag_result("ds", 1)) is True
        assert asyncio.run(service.get_tag_result("ds", 1)) is None

    def test_evict_tag_result_returns_false_when_absent(self, service: TaggingService) -> None:
        """No matching entry → ``False`` (no error)."""
        assert asyncio.run(service.evict_tag_result("ds", 99999)) is False

    def test_threshold_change_misses_cache(
        self,
        service: TaggingService,
        test_configuration,
        image_info: ImageInfo,
    ) -> None:
        """A request with a different *bucketed* fingerprint reads a different cache slot.

        ``tag_image`` is called once under config-default thresholds
        (bucket ``general=0.2``, ``character=0.8``). A subsequent
        ``get_tag_result`` with a ``general_threshold`` that floors to
        a different bucket (``0.6`` here, instead of ``0.2``) reads
        the empty ``general=0.6`` slot.

        ``rating_threshold`` alone never changes the bucket (it's
        always ``0`` in the key); bumping only ``rating`` is a cache
        hit on the same slot — which matches the design (cached
        rating is always 0, rating is reapplied at retrieval).
        """
        test_configuration.tagger_rating_threshold = 0.0
        test_configuration.tagger_general_threshold = 0.35
        test_configuration.tagger_character_threshold = 0.85

        client = make_client_mock(alive=True)
        with patch_client_factory(client):
            asyncio.run(service.tag_image("ds", image_info))

        # Cache was written under (0.0, 0.2, 0.8); this reads the
        # (0.0, 0.6, 0.8) bucket, which is empty.
        assert (
            asyncio.run(
                service.get_tag_result(
                    "ds",
                    1,
                    thresholds=TaggingThresholds(rating=0.0, general=0.7, character=0.85),
                )
            )
            is None
        )

    def test_rating_threshold_change_does_not_change_bucket(
        self,
        service: TaggingService,
        test_configuration,
        image_info: ImageInfo,
    ) -> None:
        """Bumping only ``rating_threshold`` keeps the same cache bucket (rating is hardcoded to 0).

        Reading the cache with a higher rating threshold re-filters the
        cached set on the way out — not a cache miss, just a stricter
        filter pass at retrieval.
        """
        test_configuration.tagger_rating_threshold = 0.0
        test_configuration.tagger_general_threshold = 0.35
        test_configuration.tagger_character_threshold = 0.85

        from yadc.taggers.base import TaggerResult

        client = make_client_mock(
            alive=True,
            tag_result=TaggerResult(
                tags={"general": 0.6, "safe": 0.5, "explicit": 0.99},
                categories={"rating": ["general", "safe", "explicit"]},
            ),
        )
        with patch_client_factory(client):
            asyncio.run(service.tag_image("ds", image_info))

        # Bumping rating to 0.9 keeps the same (rating=0, general=0.2,
        # character=0.8) slot. Re-filter at request values drops "general"
        # (score 0.6 < 0.9) and "safe" (score 0.5 < 0.9); only
        # "explicit" survives.
        out = asyncio.run(service.get_tag_result("ds", 1, thresholds=TaggingThresholds(rating=0.9)))
        assert out is not None
        assert out.tags == {"explicit": 0.99}

    def test_model_swap_misses_cache(
        self,
        service: TaggingService,
        test_configuration,
        image_info: ImageInfo,
    ) -> None:
        """Swapping ``tagger_repo_id`` mid-session gives the next reader an empty slot.

        The cache key includes the model id, so an entry cached under
        the old model isn't returned when the config now points at a
        different one. The old entry ages out via LRU eviction.
        """
        test_configuration.tagger_repo_id = "SmilingWolf/wd-eva02-large-tagger-v3"
        client = make_client_mock(alive=True)
        with patch_client_factory(client):
            asyncio.run(service.tag_image("ds", image_info))
        assert asyncio.run(service.get_tag_result("ds", 1)) is not None

        # Swap the configured model.
        test_configuration.tagger_repo_id = "SmilingWolf/wd-vit-large-tagger-v3"
        assert asyncio.run(service.get_tag_result("ds", 1)) is None

    def test_lru_evicts_when_over_budget(
        self,
        service: TaggingService,
        test_configuration,
        make_image_info,
    ) -> None:
        """Capacity from ``tagger_result_max_memory_bytes`` — entries beyond it are evicted LRU-first.

        With a controlled size function (each entry costs 30 bytes)
        and a 60-byte budget, only 2 entries fit. The third write
        pushes the total over budget and the LRU entry (image 1) is
        evicted to bring it back to 60 bytes. A controlled size
        function sidesteps the mock's natural TaggerResult footprint
        so the budget math is deterministic.
        """
        from yadc.utils import MemoryLRU

        test_configuration.tagger_result_max_memory_bytes = 60

        def _size_fn(_value):  # noqa: ANN001
            return 30  # each (mock) TaggerResult occupies 30 bytes

        service._tag_results = MemoryLRU(  # noqa: SLF001
            test_configuration.tagger_result_max_memory_bytes,
            size_fn=_size_fn,
        )

        client = make_client_mock(alive=True)
        with patch_client_factory(client):
            for info in (make_image_info(image_id=1), make_image_info(image_id=2), make_image_info(image_id=3)):
                asyncio.run(service.tag_image("ds", info))

        # Image 1 was the least-recently-used after writing 2 and 3 —
        # the budget eviction on the third write kicked it out first.
        assert asyncio.run(service.get_tag_result("ds", 1)) is None
        assert asyncio.run(service.get_tag_result("ds", 2)) is not None
        assert asyncio.run(service.get_tag_result("ds", 3)) is not None

    def test_budget_uses_tamer_result_size(
        self,
        service: TaggingService,
        make_image_info,
    ) -> None:
        """The service installs ``tamer_result_size`` as the bytes-aware size function."""
        from yadc.taggers.base import tamer_result_size
        from yadc.utils import MemoryLRU

        assert isinstance(service._tag_results, MemoryLRU)  # noqa: SLF001
        # Smoke-test the size function on a representative result;
        # exact numbers depend on Python / dict layout, so we just
        # assert it returns a positive value.
        result = TaggerResult(
            tags={"1girl": 0.95, "long_hair": 0.8},
            categories={"general": ["1girl", "long_hair"]},
        )
        assert tamer_result_size(result) > 0


class TestTaggerReadThroughCache:
    """``TaggingService.tag_image`` is read-through on the LRU cache.

    Cache hits must (a) return the cached value verbatim, (b) skip the
    subprocess spawn, (c) skip the byte read on the image path, and
    (d) still dispatch ``ImageTaggedEvent`` so SSE clients see the
    same success signal a real model run would emit. Cache misses
    must still go through the full path so the entry is then
    available for the next read.

    These tests reuse a single ``make_client_mock`` across the cold
    call and any warm-calls-that-should-still-run-the-model: the
    service-level ``_tagger_client`` is kept alive across the cold
    call, and a fresh ``patch_client_factory`` would re-spawn on the
    second ``_ensure_running_locked`` only if the existing client
    wasn't alive. So the same mock's ``tag.await_count`` is the
    right surface to assert against for "did the model run".
    """

    def test_cache_hit_returns_cached_value_without_calling_model(
        self,
        service: TaggingService,
        image_info: ImageInfo,
    ) -> None:
        """A second ``tag_image`` under the same fingerprint hits the LRU and skips the subprocess."""
        client = make_client_mock(alive=True)
        with patch_client_factory(client):
            cold = asyncio.run(service.tag_image("ds", image_info))
            assert client.tag.await_count == 1

            warm = asyncio.run(service.tag_image("ds", image_info))
            assert client.tag.await_count == 1  # unchanged — short-circuited

        # Cold-path result is the source of truth; warm should equal it
        # by value (the post-threshold / post-replace form).
        assert warm.tags == cold.tags
        assert warm.categories == cold.categories

    def test_cache_hit_does_not_touch_idle_timer(
        self,
        service: TaggingService,
        image_info: ImageInfo,
    ) -> None:
        """A cache hit must not reset ``_last_used_t`` (subprocess wasn't started)."""
        import time

        client = make_client_mock(alive=True)
        with patch_client_factory(client):
            asyncio.run(service.tag_image("ds", image_info))

        baseline_idle_t = service._last_used_t  # noqa: SLF001 — set by the cold call
        # Wait a tick so monotonic moves; the hit must leave it alone.
        before = time.monotonic()
        time.sleep(0.05)
        with patch_client_factory(client):
            asyncio.run(service.tag_image("ds", image_info))
        assert service._last_used_t == baseline_idle_t  # noqa: SLF001
        assert time.monotonic() - before >= 0.05

    def test_cache_hit_dispatches_image_tagged_event(
        self,
        service: TaggingService,
        image_info: ImageInfo,
    ) -> None:
        """``ImageTaggedEvent`` is still emitted on a hit so SSE clients see success."""
        from yadc.api.events import ImageTaggedEvent

        client = make_client_mock(alive=True)
        with patch_client_factory(client):
            # Cold call to populate the cache and prime the event log.
            asyncio.run(service.tag_image("ds", image_info))

            # Spy on dispatch for the warm call.
            dispatch_spy = MagicMock()
            original_dispatch = service._event_dispatcher.dispatch  # noqa: SLF001
            service._event_dispatcher.dispatch = dispatch_spy  # type: ignore[method-assign]  # noqa: SLF001
            try:
                asyncio.run(service.tag_image("ds", image_info))
            finally:
                service._event_dispatcher.dispatch = original_dispatch  # type: ignore[method-assign]  # noqa: SLF001

        tagged = [c.args[0] for c in dispatch_spy.call_args_list if isinstance(c.args[0], ImageTaggedEvent)]
        assert len(tagged) == 1
        assert tagged[0].dataset_name == "ds"
        assert tagged[0].image_id == image_info.id
        # Cache hits are instantaneous — ``duration_ms=0`` marks "no
        # real model run" for downstream consumers (e.g. timing stats).
        assert tagged[0].duration_ms == 0

    def test_cache_miss_under_different_thresholds_runs_the_model(
        self,
        service: TaggingService,
        image_info: ImageInfo,
    ) -> None:
        """Different *bucketed* thresholds hash to a different slot and miss the cache.

        ``rating_threshold`` changes keep the same bucket (rating is
        always 0 in the key), so to exercise a real bucket-distinct
        miss we bump ``general_threshold`` enough to land in a
        different bucket.
        """
        client = make_client_mock(alive=True)
        with patch_client_factory(client):
            asyncio.run(service.tag_image("ds", image_info))
            # Cold write under default thresholds (bucket g=0.2) called
            # the model once.
            assert client.tag.await_count == 1

            # Override with general=0.7 (bucket g=0.6) — different slot,
            # the read-through path must fall through to the model again.
            asyncio.run(
                service.tag_image(
                    "ds",
                    image_info,
                    thresholds=TaggingThresholds(rating=0.0, general=0.7, character=0.85),
                )
            )
            assert client.tag.await_count == 2

    def test_warm_cache_writes_under_independent_fingerprint(
        self,
        service: TaggingService,
        image_info: ImageInfo,
    ) -> None:
        """Buckets stay isolated: a bucket-distinct write doesn't pollute the existing slot.

        Calls under two different bucket floors + a repeat of the
        first: 3 calls, 2 model runs. The override write created its
        own bucket; the third call reuses the original slot.
        """
        client = make_client_mock(alive=True)
        with patch_client_factory(client):
            asyncio.run(service.tag_image("ds", image_info))  # bucket g=0.2 (default)
            asyncio.run(
                service.tag_image(
                    "ds",
                    image_info,
                    thresholds=TaggingThresholds(rating=0.0, general=0.7, character=0.85),
                )
            )  # bucket g=0.6 (override) — miss
            asyncio.run(service.tag_image("ds", image_info))  # bucket g=0.2 (default) — hit
            assert client.tag.await_count == 2


class TestTaggerResultKey:
    """``TaggerResultKey`` — hashable + ``__eq__`` contract used as an LRU key."""

    def _make_key(self, **overrides: Any) -> TaggerResultKey:
        defaults: dict[str, Any] = dict(
            dataset_name="ds",
            image_id=1,
            model_id="hf:foo",
            rating_threshold=0.0,
            general_threshold=0.35,
            character_threshold=0.85,
        )
        defaults.update(overrides)
        return TaggerResultKey(**defaults)

    def test_equal_keys_hash_to_same_bucket(self) -> None:
        """Two keys with identical fields compare equal and hash equal."""
        a = self._make_key()
        b = self._make_key()
        assert a == b
        assert hash(a) == hash(b)

    def test_distinct_field_makes_keys_unequal(self) -> None:
        """Each field participates in equality — flipping any one breaks the match.

        ``replace_underscores`` is no longer in the key (applied
        post-hoc), so it's intentionally omitted from the
        equality-breaking list.
        """
        base = self._make_key()
        for overrides in (
            {"dataset_name": "other"},
            {"image_id": 2},
            {"model_id": "hf:other"},
            {"rating_threshold": 0.1},
            {"general_threshold": 0.5},
            {"character_threshold": 0.9},
        ):
            modified = self._make_key(**overrides)
            assert modified != base, f"expected {overrides} to break equality"

    def test_key_is_hashable_in_a_dict(self) -> None:
        """The class satisfies the dict-key contract the LRU relies on."""
        key = self._make_key()
        d: dict[TaggerResultKey, int] = {key: 42}
        assert d[key] == 42

    def test_key_is_immutable(self) -> None:
        """Frozen dataclass — field assignment raises ``FrozenInstanceError``."""
        import dataclasses

        key = self._make_key()
        with pytest.raises(dataclasses.FrozenInstanceError):
            key.image_id = 999  # type: ignore[misc]


class TestBucketThreshold:
    """``bucket_threshold`` — rounds a tagger threshold DOWN to the nearest 0.2 boundary."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (0.0, 0.0),
            (0.05, 0.0),
            (0.1, 0.0),
            (0.2, 0.2),
            (0.35, 0.2),  # canonical default general threshold
            (0.4, 0.4),
            (0.5, 0.4),
            (0.6, 0.6),
            (0.85, 0.8),  # canonical default character threshold
            (1.0, 1.0),
            (1.5, 1.4),
            (2.0, 2.0),
        ],
    )
    def test_floor_to_step(self, value: float, expected: float) -> None:
        from yadc.api.services.tagging import bucket_threshold

        assert bucket_threshold(value) == pytest.approx(expected)

    def test_negative_returns_zero(self) -> None:
        """Negative thresholds aren't useful in the cache — floor to 0."""
        from yadc.api.services.tagging import bucket_threshold

        assert bucket_threshold(-0.1) == 0.0
