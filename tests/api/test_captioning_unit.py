"""Tests for AsyncCaptionJob — Protocol implementation, runner integration, and CaptioningService lifecycle."""

import asyncio
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from yadc.api.events import (
    CaptioningStatusEvent,
    ImageCaptionedEvent,
    ImageCaptionErrorEvent,
    ImageCaptionStartedEvent,
    ImageRefinedEvent,
    StartupEvent,
)
from yadc.api.modules.event_dispatcher import EventDispatcher
from yadc.api.services.captioning import AsyncCaptionJob, AsyncCaptionJobRunner, CaptioningService
from yadc.core.captioning import CaptioningCallbacks, CaptionJobOptions

# Patch target paths. Centralized so renames only need to be updated
# in one place.
_PATCH_CAPTIONING_RUNNER = "yadc.api.services.captioning.job_runner.CaptioningRunner"
_PATCH_LOAD_DATASET_CONFIG = "yadc.api.services.captioning.job_runner.load_dataset_config"


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_runner():
    """Create an AsyncCaptionJobRunner with mocked dependencies."""
    return MagicMock(spec=AsyncCaptionJobRunner)


@pytest.fixture
def runner(test_configuration):
    """Create a real AsyncCaptionJobRunner with mocked DI dependencies.

    Use this when the test needs to exercise the runner's own methods
    (preflight, emit_image_*, expect_file_changes, build_runner). For
    tests that just want to verify the job delegates to the runner,
    use ``mock_runner`` (a MagicMock spec) instead.
    """
    return AsyncCaptionJobRunner(
        configuration=test_configuration,
        dataset_service=MagicMock(),
        dataset_watcher=MagicMock(),
        event_dispatcher=MagicMock(),
        logger=MagicMock(),
    )


@pytest.fixture
def job(mock_runner):
    """Create an AsyncCaptionJob with a mocked runner."""
    opts = CaptionJobOptions(max_concurrent=1)

    return AsyncCaptionJob(
        dataset_name="test_ds",
        options=opts,
        job_id="abc123",
        on_done=lambda: None,
        runner=mock_runner,
    )


# ---------------------------------------------------------------------------
# Protocol implementation + callbacks
# ---------------------------------------------------------------------------


class TestCaptioningCallbacks:
    """AsyncCaptionJob implements the CaptioningCallbacks Protocol."""

    def test_satisfies_protocol(self, job):
        """The job instance must satisfy the CaptioningCallbacks Protocol."""
        assert isinstance(job, CaptioningCallbacks)

    @pytest.mark.asyncio
    async def test_on_token_is_noop(self, job):
        """on_token does not raise (no observable side effect)."""
        await job.on_token("hello")
        await job.on_token(" world")

    @pytest.mark.asyncio
    async def test_on_image_started_delegates_to_runner(self, job, mock_runner):
        """on_image_started delegates to runner.emit_image_started."""
        image = MagicMock(path="/img.jpg")
        await job.on_image_started(image)

        mock_runner.emit_image_started.assert_called_once_with("test_ds", "abc123", image)

    @pytest.mark.asyncio
    async def test_on_image_captioned_increments_processed(self, job, mock_runner):
        """on_image_captioned increments the processed counter."""
        await job.on_image_captioned(MagicMock(path="/img.jpg", caption="hi"), duration_ms=100)

        assert job._processed == 1

    @pytest.mark.asyncio
    async def test_on_image_captioned_delegates_to_runner(self, job, mock_runner):
        """on_image_captioned calls runner.emit_image_captioned and runner.emit_status."""
        image = MagicMock(path="/img.jpg", caption="hello")
        await job.on_image_captioned(image, duration_ms=120)

        mock_runner.emit_image_captioned.assert_called_once_with("test_ds", "abc123", image, "", "", 120)
        mock_runner.emit_status.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_on_image_error_increments_and_records(self, job, mock_runner):
        """on_image_error increments errors and records the message."""
        await job.on_image_error(MagicMock(path="/img.jpg"), "boom", duration_ms=50)

        assert job._errors == 1
        assert job._error_messages == ["boom"]

    @pytest.mark.asyncio
    async def test_on_image_error_delegates_to_runner(self, job, mock_runner):
        """on_image_error calls runner.emit_image_error and runner.emit_status."""
        image = MagicMock(path="/unknown.jpg")
        await job.on_image_error(image, "boom", duration_ms=50)

        mock_runner.emit_image_error.assert_called_once_with("test_ds", "abc123", "", "", image, "boom", 50)
        mock_runner.emit_status.assert_awaited_once()


# ---------------------------------------------------------------------------
# _ado_run integration with CaptioningRunner
# ---------------------------------------------------------------------------


class TestAdoRunWithRunner:
    """_ado_run — uses runner.preflight + CaptioningRunner + processes images."""

    @pytest.fixture
    def ado_run_env(self, tmp_path, test_configuration):
        """Set up the environment for _ado_run tests.

        Yields ``(mock_instance, MockRunner, mock_load, mock_ds)`` where
        the patches target the job_runner module.
        """
        config_path = tmp_path / "config.toml"
        config_path.write_text('[api]\nurl = "x"\nmodel_name = "m"\n[prompt]\ntemplate = "t"\n')

        mock_ds = MagicMock()
        mock_ds.get_dataset.return_value = MagicMock(config_path=str(config_path))
        mock_ds.get_image_paths_in_desc_order.return_value = []

        mock_watcher = MagicMock()
        mock_dispatcher = MagicMock()
        mock_logger = MagicMock()

        runner = AsyncCaptionJobRunner(
            configuration=test_configuration,
            dataset_service=mock_ds,
            dataset_watcher=mock_watcher,
            event_dispatcher=mock_dispatcher,
            logger=mock_logger,
        )

        with patch(_PATCH_CAPTIONING_RUNNER) as MockRunner, patch(_PATCH_LOAD_DATASET_CONFIG) as mock_load:
            mock_instance = MagicMock()
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_instance.caption_images = AsyncMock()
            MockRunner.return_value = mock_instance
            yield mock_instance, MockRunner, mock_load, runner, mock_ds

    @pytest.mark.asyncio
    async def test_uses_captioning_runner(self, ado_run_env):
        """_ado_run builds a CaptioningRunner with the parsed config."""
        mock_instance, MockRunner, mock_load, runner, mock_ds = ado_run_env
        mock_config = MagicMock()
        mock_load.return_value = (mock_config, [MagicMock()])

        opts = CaptionJobOptions(max_concurrent=1)
        job = AsyncCaptionJob(dataset_name="test_ds", options=opts, job_id="abc123", on_done=lambda: None, runner=runner)
        await job._ado_run()

        MockRunner.assert_called_once()
        args, kwargs = MockRunner.call_args
        assert args[0] is mock_config
        assert args[1] is opts

    @pytest.mark.asyncio
    async def test_caption_images_called_with_all_images(self, ado_run_env):
        """_ado_run calls caption_images with the to-do list and self as callbacks."""
        mock_instance, _, mock_load, runner, _ = ado_run_env
        images = [MagicMock(), MagicMock(), MagicMock()]
        mock_load.return_value = (MagicMock(), images)

        opts = CaptionJobOptions(max_concurrent=1)
        job = AsyncCaptionJob(dataset_name="test_ds", options=opts, job_id="abc123", on_done=lambda: None, runner=runner)
        await job._ado_run()

        mock_instance.caption_images.assert_awaited_once()
        args, kwargs = mock_instance.caption_images.call_args
        assert args[0] == images
        assert kwargs.get("max_concurrent") == 1  # default

    @pytest.mark.asyncio
    async def test_caption_images_passes_max_concurrent(self, ado_run_env):
        """_ado_run forwards the CaptionJobOptions.max_concurrent to the runner."""
        from yadc.core.captioning.options import CaptionJobOptions

        mock_instance, _, mock_load, runner, _ = ado_run_env
        mock_load.return_value = (MagicMock(), [MagicMock()])

        opts = CaptionJobOptions(max_concurrent=4)
        job = AsyncCaptionJob(dataset_name="test_ds", options=opts, job_id="abc123", on_done=lambda: None, runner=runner)
        await job._ado_run()

        assert mock_instance.caption_images.call_args.kwargs.get("max_concurrent") == 4

    @pytest.mark.asyncio
    async def test_caption_images_receives_desc_sorted_to_do(self, ado_run_env):
        """``_ado_run`` passes the SQL id-DESC-sorted list to
        ``caption_images`` so concurrent captioning starts with the
        newest images first. Reorder is driven by a single SQL
        query (``get_image_paths_in_desc_order``), not N+1 callbacks.
        """
        from yadc.core.dataset import DatasetImage

        mock_instance, _, mock_load, runner, mock_ds = ado_run_env
        config_path = Path(mock_ds.get_dataset.return_value.config_path)

        # Real loader call returns images in filesystem order; the
        # API runner then reorders via the SQL DESC query.
        img_a = DatasetImage(path=str(config_path.parent / "a.jpg"))
        img_b = DatasetImage(path=str(config_path.parent / "b.jpg"))
        img_c = DatasetImage(path=str(config_path.parent / "c.jpg"))

        # Path list returned by the SQL query (newest first). IDs
        # are intentionally non-monotonic with file names so we
        # can prove the sort comes from SQL, not path.
        mock_ds.get_image_paths_in_desc_order.return_value = [
            str(img_b.path),
            str(img_c.path),
            str(img_a.path),
        ]

        mock_load.return_value = (MagicMock(), [img_a, img_b, img_c])

        opts = CaptionJobOptions(max_concurrent=1)
        job = AsyncCaptionJob(dataset_name="test_ds", options=opts, job_id="abc123", on_done=lambda: None, runner=runner)
        await job._ado_run()

        mock_instance.caption_images.assert_awaited_once()
        args, _ = mock_instance.caption_images.call_args
        assert args[0] == [img_b, img_c, img_a]
        # The reorder uses the SQL helper, not N+1 get_image_by_path calls.
        mock_ds.get_image_paths_in_desc_order.assert_called_once_with("test_ds")

    @pytest.mark.asyncio
    async def test_done_status_on_no_images(self, ado_run_env):
        """_ado_run early-exits with done status when there are no images."""
        _, _, mock_load, runner, _ = ado_run_env
        mock_load.return_value = (MagicMock(), [])

        opts = CaptionJobOptions(max_concurrent=1)
        job = AsyncCaptionJob(dataset_name="test_ds", options=opts, job_id="abc123", on_done=lambda: None, runner=runner)
        await job._ado_run()

        snap = await job.snapshot()
        assert snap.status == "done"
        assert snap.total == 0

    @pytest.mark.asyncio
    async def test_done_status_after_processing(self, ado_run_env):
        """_ado_run sets done status after processing all images without stop."""
        _, _, mock_load, runner, _ = ado_run_env
        mock_load.return_value = (MagicMock(), [MagicMock(), MagicMock()])

        opts = CaptionJobOptions(max_concurrent=1)
        job = AsyncCaptionJob(dataset_name="test_ds", options=opts, job_id="abc123", on_done=lambda: None, runner=runner)
        await job._ado_run()

        snap = await job.snapshot()
        assert snap.status == "done"

    @pytest.mark.asyncio
    async def test_cancelled_status_when_stop_event_set(self, ado_run_env):
        """_ado_run sets cancelled status when the stop event is set."""
        _, _, mock_load, runner, _ = ado_run_env
        mock_load.return_value = (MagicMock(), [MagicMock()])

        opts = CaptionJobOptions(max_concurrent=1)
        job = AsyncCaptionJob(dataset_name="test_ds", options=opts, job_id="abc123", on_done=lambda: None, runner=runner)
        job._stop_event.set()
        await job._ado_run()

        snap = await job.snapshot()
        assert snap.status == "cancelled"

    @pytest.mark.asyncio
    async def test_cancellation_propagates(self, ado_run_env):
        """_ado_run re-raises CancelledError (does not catch it)."""
        mock_instance, _, mock_load, runner, _ = ado_run_env
        mock_instance.caption_images = AsyncMock(side_effect=asyncio.CancelledError())
        mock_load.return_value = (MagicMock(), [MagicMock()])

        opts = CaptionJobOptions(max_concurrent=1)
        job = AsyncCaptionJob(dataset_name="test_ds", options=opts, job_id="abc123", on_done=lambda: None, runner=runner)
        with pytest.raises(asyncio.CancelledError):
            await job._ado_run()

    @pytest.mark.asyncio
    async def test_batch_aborted_error_propagates(self, ado_run_env):
        """_ado_run lets BatchAbortedError through; _arun turns it into status='error'."""
        from yadc.core.captioning.runner import BatchAbortedError

        mock_instance, _, mock_load, runner, _ = ado_run_env
        mock_instance.caption_images = AsyncMock(side_effect=BatchAbortedError("aborted"))
        mock_load.return_value = (MagicMock(), [MagicMock()])

        opts = CaptionJobOptions(max_concurrent=1)
        job = AsyncCaptionJob(dataset_name="test_ds", options=opts, job_id="abc123", on_done=lambda: None, runner=runner)
        with pytest.raises(BatchAbortedError):
            await job._ado_run()

    @pytest.mark.asyncio
    async def test_set_preflight_skips_runner_preflight(self, ado_run_env):
        """Regression: when ``set_preflight`` is used, ``_ado_run`` reuses
        the cached result and does not call ``runner.preflight`` again.

        This locks in the optimization that the synchronous preflight
        in ``CaptioningService.start_job_async`` is the only one
        (avoiding re-parsing the config and re-querying the DB on the
        background task).
        """
        mock_instance, _, _, runner, _ = ado_run_env

        # pre-built config + image list as if CaptioningService had
        # done the synchronous preflight.
        prebuilt_config = MagicMock()
        prebuilt_config.api.url = "http://prebuilt"
        prebuilt_config.api.model_name = "prebuilt-model"
        prebuilt_images = [MagicMock(), MagicMock()]

        opts = CaptionJobOptions(max_concurrent=1)
        job = AsyncCaptionJob(dataset_name="test_ds", options=opts, job_id="abc123", on_done=lambda: None, runner=runner)
        job.set_preflight(prebuilt_config, prebuilt_images)

        # Patch runner.preflight so we can assert it isn't called.
        # If the optimization is broken, this mock would be called
        # (and would fail because the dataset mock isn't wired up).
        with patch.object(runner, "preflight") as mock_runner_preflight:
            await job._ado_run()

        # runner.preflight must NOT have been called — the cached
        # preflight is the source of truth.
        mock_runner_preflight.assert_not_called()
        # And the image list passed to caption_images is the cached one.
        assert mock_instance.caption_images.call_args.args[0] is prebuilt_images

    @pytest.mark.asyncio
    async def test_set_preflight_seeds_snapshot_synchronously(self, ado_run_env):
        """``set_preflight`` populates ``api_url``/``api_model_name``/``total``
        so the response from ``start_job_async`` is accurate before the
        background task runs."""
        mock_instance, _, mock_load, runner, _ = ado_run_env
        mock_load.return_value = (MagicMock(), [MagicMock(), MagicMock(), MagicMock()])

        prebuilt_config = MagicMock()
        prebuilt_config.api.url = "http://prebuilt"
        prebuilt_config.api.model_name = "prebuilt-model"
        prebuilt_images = [MagicMock(), MagicMock(), MagicMock()]

        opts = CaptionJobOptions(max_concurrent=1)
        job = AsyncCaptionJob(dataset_name="test_ds", options=opts, job_id="abc123", on_done=lambda: None, runner=runner)
        job.set_preflight(prebuilt_config, prebuilt_images)

        snap = await job.snapshot()
        assert snap.api_url == "http://prebuilt"
        assert snap.api_model_name == "prebuilt-model"
        assert snap.total == 3

        # Don't bother running _ado_run — clean up the task to avoid
        # a dangling asyncio warning.
        del mock_instance  # silence unused warning


# ---------------------------------------------------------------------------
# AsyncCaptionJobRunner: expect_file_changes
# ---------------------------------------------------------------------------


class TestExpectFileChanges:
    """Runner's expect_file_changes — calls dataset_watcher.expect_file_change per path."""

    def test_registers_each_path(self, test_configuration):
        mock_watcher = MagicMock()
        runner = AsyncCaptionJobRunner(
            configuration=test_configuration,
            dataset_service=MagicMock(),
            dataset_watcher=mock_watcher,
            event_dispatcher=MagicMock(),
            logger=MagicMock(),
        )

        paths = ["/a.txt", "/b.toml", "/c.history~"]
        runner.expect_file_changes("test_ds", paths)

        assert mock_watcher.expect_file_change.call_count == 3
        for path in paths:
            mock_watcher.expect_file_change.assert_any_call("test_ds", path)

    def test_empty_paths(self, test_configuration):
        mock_watcher = MagicMock()
        runner = AsyncCaptionJobRunner(
            configuration=test_configuration,
            dataset_service=MagicMock(),
            dataset_watcher=mock_watcher,
            event_dispatcher=MagicMock(),
            logger=MagicMock(),
        )

        runner.expect_file_changes("test_ds", [])

        mock_watcher.expect_file_change.assert_not_called()


# ---------------------------------------------------------------------------
# AsyncCaptionJobRunner: emit_status
# ---------------------------------------------------------------------------


class TestRunnerEmitStatus:
    """Tests for AsyncCaptionJobRunner.emit_status."""

    @pytest.mark.asyncio
    async def test_emits_status_event_with_snapshot_data(self, test_configuration):
        mock_dispatcher = MagicMock()
        runner = AsyncCaptionJobRunner(
            configuration=test_configuration,
            dataset_service=MagicMock(),
            dataset_watcher=MagicMock(),
            event_dispatcher=mock_dispatcher,
            logger=MagicMock(),
        )

        from yadc.api.services.captioning.models import JobInfo

        snapshot = JobInfo(
            status="running",
            dataset_name="test_ds",
            job_id="abc123",
            processed=5,
            total=10,
            errors=1,
            api_url="http://test",
            api_model_name="model-v1",
            error_messages=["oops"],
            max_concurrent=2,
        )
        started_at = time.monotonic() - 5.0
        await runner.emit_status("test_ds", "abc123", snapshot, started_at)

        mock_dispatcher.dispatch.assert_called_once()
        event = mock_dispatcher.dispatch.call_args[0][0]
        assert isinstance(event, CaptioningStatusEvent)
        assert event.dataset_name == "test_ds"
        assert event.job_id == "abc123"
        assert event.processed == 5
        assert event.max_concurrent == 2
        assert 4.9 <= event.elapsed < 6.0

    @pytest.mark.asyncio
    async def test_elapsed_zero_when_started_at_none(self, test_configuration):
        mock_dispatcher = MagicMock()
        runner = AsyncCaptionJobRunner(
            configuration=test_configuration,
            dataset_service=MagicMock(),
            dataset_watcher=MagicMock(),
            event_dispatcher=mock_dispatcher,
            logger=MagicMock(),
        )

        from yadc.api.services.captioning.models import JobInfo

        snapshot = JobInfo(status="running", dataset_name="test_ds")
        await runner.emit_status("test_ds", "abc123", snapshot, None)

        event = mock_dispatcher.dispatch.call_args[0][0]
        assert isinstance(event, CaptioningStatusEvent)
        assert event.elapsed == 0.0


# ---------------------------------------------------------------------------
# AsyncCaptionJobRunner: emit_image_*
# ---------------------------------------------------------------------------


def _image_info(
    id_: int = 42,
    file_name: str = "img.jpg",
    path: str = "/img.jpg",
    has_caption: bool = True,
    has_toml: bool = False,
) -> MagicMock:
    """Build a MagicMock that quacks like DatasetService.get_image return value."""
    info = MagicMock()
    info.id = id_
    info.file_name = file_name
    info.path = path
    info.has_caption = has_caption
    info.has_toml = has_toml
    info.width = 100
    info.height = 100
    info.draft_names = []
    info.last_modified_t = 1.0
    return info


class TestRunnerEmitImageStarted:
    """emit_image_started — looks up the image in the DB and dispatches ImageCaptionStartedEvent."""

    def test_dispatches_event_with_db_fields(self, runner):
        """Dispatches ImageCaptionStartedEvent with the DB image's id and file_name."""
        info = _image_info()
        runner._dataset_service.get_image_by_path.return_value = info

        image = MagicMock(path="/img.jpg")
        runner.emit_image_started("test_ds", "abc123", image)

        runner._dataset_service.get_image_by_path.assert_called_once_with("test_ds", "/img.jpg")
        runner._event_dispatcher.dispatch.assert_called_once()
        event = runner._event_dispatcher.dispatch.call_args[0][0]
        assert isinstance(event, ImageCaptionStartedEvent)
        assert event.dataset_name == "test_ds"
        assert event.job_id == "abc123"
        assert event.image_id == 42
        assert event.file_name == "img.jpg"

    def test_skips_dispatch_when_image_not_in_db(self, runner):
        """If the image isn't in the DB, no event is dispatched."""
        runner._dataset_service.get_image_by_path.return_value = None

        runner.emit_image_started("test_ds", "abc123", MagicMock(path="/unknown.jpg"))

        runner._event_dispatcher.dispatch.assert_not_called()


class TestRunnerEmitImageCaptioned:
    """emit_image_captioned — refreshes the image index, re-fetches, then dispatches ImageCaptionedEvent."""

    def test_refreshes_index_then_dispatches(self, runner):
        """After a successful caption, refresh_image_index is called and the event
        uses the fresh image data (including has_caption/has_toml)."""
        before = _image_info(has_caption=False, has_toml=False)
        # get_image_by_path returns the pre-refresh row.
        runner._dataset_service.get_image_by_path.return_value = before
        # get_image (after refresh) returns the post-refresh row.
        after = _image_info(has_caption=True, has_toml=True)
        runner._dataset_service.get_image.return_value = after

        image = MagicMock(path="/img.jpg", caption="a fluffy cat")
        runner.emit_image_captioned(
            "test_ds",
            "abc123",
            image,
            api_url="http://api",
            api_model_name="m-v1",
            duration_ms=250,
        )

        runner._dataset_service.refresh_image_index.assert_called_once_with("test_ds", 42)
        runner._dataset_service.get_image.assert_called_once_with("test_ds", 42)
        runner._event_dispatcher.dispatch.assert_called_once()
        event = runner._event_dispatcher.dispatch.call_args[0][0]
        assert isinstance(event, ImageCaptionedEvent)
        assert event.dataset_name == "test_ds"
        assert event.job_id == "abc123"
        assert event.id == 42
        assert event.caption == "a fluffy cat"
        assert event.duration_ms == 250
        assert event.api_url == "http://api"
        assert event.api_model_name == "m-v1"
        # Fresh post-refresh fields make it onto the event.
        assert event.has_caption is True
        assert event.has_toml is True

    def test_skips_dispatch_when_image_not_in_db(self, runner):
        """If the image isn't in the DB, no event is dispatched and no refresh happens."""
        runner._dataset_service.get_image_by_path.return_value = None

        runner.emit_image_captioned(
            "test_ds",
            "abc123",
            MagicMock(path="/missing.jpg", caption="x"),
            api_url="",
            api_model_name="",
            duration_ms=0,
        )

        runner._dataset_service.refresh_image_index.assert_not_called()
        runner._dataset_service.get_image.assert_not_called()
        runner._event_dispatcher.dispatch.assert_not_called()

    def test_skips_dispatch_when_image_vanishes_after_refresh(self, runner):
        """If the image disappears between by-path lookup and the post-refresh
        get_image call, no event is dispatched (the row was deleted mid-caption)."""
        runner._dataset_service.get_image_by_path.return_value = _image_info()
        runner._dataset_service.get_image.return_value = None

        runner.emit_image_captioned(
            "test_ds",
            "abc123",
            MagicMock(path="/img.jpg", caption="x"),
            api_url="",
            api_model_name="",
            duration_ms=0,
        )

        runner._event_dispatcher.dispatch.assert_not_called()


class TestRunnerEmitImageError:
    """emit_image_error — uses image_id from the DB lookup, or -1 if the image is unknown."""

    def test_dispatches_with_db_image_id(self, runner):
        """When the image is in the DB, the event uses its real id."""
        runner._dataset_service.get_image_by_path.return_value = _image_info(id_=99)

        image = MagicMock(path="/img.jpg")
        runner.emit_image_error(
            "test_ds",
            "abc123",
            api_url="http://api",
            api_model_name="m",
            dataset_image=image,
            error="boom",
            duration_ms=500,
        )

        runner._event_dispatcher.dispatch.assert_called_once()
        event = runner._event_dispatcher.dispatch.call_args[0][0]
        assert isinstance(event, ImageCaptionErrorEvent)
        assert event.dataset_name == "test_ds"
        assert event.job_id == "abc123"
        assert event.image_id == 99
        assert event.error == "boom"
        assert event.duration_ms == 500
        assert event.api_url == "http://api"
        assert event.api_model_name == "m"

    def test_dispatches_with_sentinel_image_id_when_unknown(self, runner):
        """When the image is not in the DB, the event uses image_id = -1."""
        runner._dataset_service.get_image_by_path.return_value = None

        runner.emit_image_error(
            "test_ds",
            "abc123",
            api_url="",
            api_model_name="",
            dataset_image=MagicMock(path="/missing.jpg"),
            error="boom",
            duration_ms=0,
        )

        event = runner._event_dispatcher.dispatch.call_args[0][0]
        assert isinstance(event, ImageCaptionErrorEvent)
        assert event.image_id == -1


class TestRunnerEmitImageRefined:
    """emit_image_refined — dispatches ImageRefinedEvent for a dry-run refine result."""

    def test_dispatches_with_source_and_draft_name(self, runner):
        """Dispatches ImageRefinedEvent with refine.source and refine.refine_draft_name."""
        from yadc.api.services.captioning.models import RefineOptions

        runner._dataset_service.get_image_by_path.return_value = _image_info()

        image = MagicMock(path="/img.jpg")
        refine = RefineOptions(refine_source="draft", refine_draft_name="gemma")
        runner.emit_image_refined("test_ds", "abc123", image, "refined caption", refine)

        runner._event_dispatcher.dispatch.assert_called_once()
        event = runner._event_dispatcher.dispatch.call_args[0][0]
        assert isinstance(event, ImageRefinedEvent)
        assert event.dataset_name == "test_ds"
        assert event.job_id == "abc123"
        assert event.image_id == 42
        assert event.caption == "refined caption"
        assert event.source == "draft"
        assert event.draft_name == "gemma"

    def test_skips_dispatch_when_image_not_in_db(self, runner):
        """If the image isn't in the DB, no event is dispatched."""
        from yadc.api.services.captioning.models import RefineOptions

        runner._dataset_service.get_image_by_path.return_value = None

        runner.emit_image_refined(
            "test_ds",
            "abc123",
            MagicMock(path="/missing.jpg"),
            "x",
            RefineOptions(),
        )

        runner._event_dispatcher.dispatch.assert_not_called()


# ---------------------------------------------------------------------------
# CaptioningService cleanup + preflight
# ---------------------------------------------------------------------------


class TestCaptioningServiceCleanupRescan:
    """Tests for CaptioningService._cleanup_async triggering a final rescan."""

    @pytest.fixture
    def captioning_service(self, test_configuration, logging_factory):
        """Create a CaptioningService with mocked dependencies."""
        mock_ds = MagicMock()
        mock_dispatcher = MagicMock()
        mock_watcher = MagicMock()

        svc = CaptioningService(
            dataset_service=mock_ds,
            event_dispatcher=mock_dispatcher,
            dataset_watcher=mock_watcher,
            logging=logging_factory,
            configuration=test_configuration,
        )
        return svc

    @pytest.mark.asyncio
    async def test_cleanup_triggers_final_rescan(self, captioning_service):
        """_cleanup_async should call rescan_dataset after the delay."""
        mock_job = MagicMock()
        mock_job.alive = False
        mock_job.snapshot = AsyncMock(return_value=MagicMock(job_id="abc123"))
        captioning_service._async_jobs["test_ds"] = mock_job
        captioning_service._dataset_service.rescan_dataset = MagicMock()

        with patch("asyncio.sleep"):
            await captioning_service._cleanup_async("test_ds")

        captioning_service._dataset_service.rescan_dataset.assert_called_once_with("test_ds")

    @pytest.mark.asyncio
    async def test_cleanup_clears_expected_changes_before_rescan(self, captioning_service):
        """clear_expected_changes_for_job should be called before rescan."""
        mock_job = MagicMock()
        mock_job.alive = False
        mock_job.snapshot = AsyncMock(return_value=MagicMock(job_id="abc123"))
        captioning_service._async_jobs["test_ds"] = mock_job

        call_order = []
        captioning_service._dataset_watcher.clear_expected_changes_for_job = MagicMock(side_effect=lambda ds, jid: call_order.append(("clear", jid)))
        captioning_service._dataset_service.rescan_dataset = MagicMock(side_effect=lambda ds: call_order.append(("rescan", ds)))

        with patch("asyncio.sleep"):
            await captioning_service._cleanup_async("test_ds")

        assert call_order == [("clear", "abc123"), ("rescan", "test_ds")]


class TestCaptioningServiceStartup:
    """Tests for CaptioningService's @event_handler(StartupEvent) starting the cleanup task.

    Regression for the order-of-operations bug where StartupEvent was
    dispatched in Application.run() *before* uvicorn started, so async
    handlers got a "Event loop not available" warning and were silently
    skipped — leaving the cleanup task never started.
    """

    @pytest.mark.asyncio
    async def test_startup_starts_cleanup_task(self, test_configuration, logging_factory):
        """Dispatching StartupEvent with a running loop should start the periodic cleanup task."""
        event_dispatcher = EventDispatcher(logging_factory)
        event_dispatcher.set_loop(asyncio.get_running_loop())

        mock_ds = MagicMock()
        mock_watcher = MagicMock()
        svc = CaptioningService(
            dataset_service=mock_ds,
            event_dispatcher=event_dispatcher,
            dataset_watcher=mock_watcher,
            logging=logging_factory,
            configuration=test_configuration,
        )
        assert svc._cleanup_task is None

        event_dispatcher.register_service(svc)
        event_dispatcher.dispatch(StartupEvent())

        # Yield once so the dispatched coroutine can run.
        await asyncio.sleep(0)

        assert svc._cleanup_task is not None
        assert not svc._cleanup_task.done()

        # Clean up: cancel the background task so the test doesn't hang.
        svc._cleanup_task.cancel()
        try:
            await svc._cleanup_task
        except asyncio.CancelledError:
            pass


class TestStartJobPreflight:
    """Tests for start_job_async preflight — accurate totals and early exits."""

    @pytest.fixture
    def captioning_service(self, tmp_path, test_configuration, logging_factory):
        """Create a CaptioningService with a real config on disk."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[api]
url = "http://test"
model_name = "test-model"

[prompt]
template = "test"
""")

        mock_ds = MagicMock()
        mock_ds.get_dataset.return_value = MagicMock(config_path=str(config_path))

        mock_dispatcher = MagicMock()
        mock_watcher = MagicMock()

        svc = CaptioningService(
            dataset_service=mock_ds,
            event_dispatcher=mock_dispatcher,
            dataset_watcher=mock_watcher,
            logging=logging_factory,
            configuration=test_configuration,
        )
        return svc

    @pytest.mark.asyncio
    async def test_returns_done_when_no_images(self, captioning_service):
        """When preflight returns 0 images, start_job_async returns done without starting a task."""
        with patch("yadc.api.services.captioning.job_runner.load_dataset_config") as mock_load:
            mock_config = MagicMock()
            mock_load.return_value = (mock_config, [])
            with patch.object(AsyncCaptionJob, "start") as mock_start:
                info = await captioning_service.start_job_async("test_ds", CaptionJobOptions())

        assert info.status == "done"
        assert info.total == 0
        assert info.processed == 0
        mock_start.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_error_on_preflight_failure(self, captioning_service):
        """When preflight raises, start_job_async returns error without starting a task."""
        with patch("yadc.api.services.captioning.job_runner.load_dataset_config", side_effect=ValueError("bad config")):
            with patch.object(AsyncCaptionJob, "start") as mock_start:
                info = await captioning_service.start_job_async("test_ds", CaptionJobOptions())

        assert info.status == "error"
        assert info.error == "bad config"
        mock_start.assert_not_called()

    @pytest.mark.asyncio
    async def test_starts_task_when_images_present(self, captioning_service):
        """When preflight returns images, a task is started and status is running."""
        mock_img = MagicMock()
        mock_img.path = "/fake/img.jpg"

        with patch("yadc.api.services.captioning.job_runner.load_dataset_config") as mock_load:
            mock_config = MagicMock()
            mock_load.return_value = (mock_config, [mock_img])
            with patch.object(AsyncCaptionJob, "start") as mock_start:
                info = await captioning_service.start_job_async("test_ds", CaptionJobOptions())

        assert info.status == "running"
        assert info.total == 1
        mock_start.assert_called_once()
