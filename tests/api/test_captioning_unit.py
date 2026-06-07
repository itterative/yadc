"""Tests for AsyncCaptionJob — Protocol implementation, expected_change_registrar, runner integration, and CaptioningService lifecycle."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from yadc.api.events import (
    CaptioningStatusEvent,
    ImageCaptionedEvent,
    ImageCaptionErrorEvent,
    ImageCaptionStartedEvent,
)
from yadc.api.services.captioning import AsyncCaptionJob, CaptioningService
from yadc.core.captioning import CaptioningCallbacks, CaptionJobOptions

# Patch target paths for the API service module. Centralized so renames
# only need to be updated in one place.
_PATCH_CAPTIONING_RUNNER = "yadc.api.services.captioning.CaptioningRunner"
_PATCH_LOAD_DATASET_CONFIG = "yadc.api.services.captioning.load_dataset_config"


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def job(tmp_path):
    """Create an AsyncCaptionJob with mocked dependencies."""
    mock_watcher = MagicMock()
    mock_ds = MagicMock()
    mock_dispatcher = MagicMock()
    mock_logger = MagicMock()
    opts = CaptionJobOptions()

    return AsyncCaptionJob(
        dataset_name="test_ds",
        dataset_service=mock_ds,
        dataset_watcher=mock_watcher,
        event_dispatcher=mock_dispatcher,
        logger=mock_logger,
        options=opts,
        on_done=lambda: None,
        job_id="abc123",
        configuration=MagicMock(),
    )


def _image_info(id_: int = 42, file_name: str = "img.jpg", path: str = "/img.jpg") -> MagicMock:
    """Build a MagicMock that quacks like DatasetService.get_image return value."""
    info = MagicMock()
    info.id = id_
    info.file_name = file_name
    info.path = path
    info.has_caption = True
    info.has_toml = False
    info.width = 100
    info.height = 100
    info.draft_names = []
    info.last_modified_t = 1.0
    return info


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
        # The API doesn't stream tokens to clients — verify the method
        # is callable and doesn't error.
        await job.on_token("hello")
        await job.on_token(" world")

    @pytest.mark.asyncio
    async def test_on_image_started_dispatches_event(self, job):
        """on_image_started dispatches ImageCaptionStartedEvent with the right fields."""
        info = _image_info()
        job._dataset_service.get_image_by_path.return_value = info

        await job.on_image_started(MagicMock(path="/img.jpg"))

        job._event_dispatcher.dispatch.assert_called_once()
        event = job._event_dispatcher.dispatch.call_args[0][0]
        assert isinstance(event, ImageCaptionStartedEvent)
        assert event.dataset_name == "test_ds"
        assert event.job_id == "abc123"
        assert event.image_id == 42
        assert event.file_name == "img.jpg"

    @pytest.mark.asyncio
    async def test_on_image_started_skips_unknown_path(self, job):
        """If the image isn't in the dataset, no event is dispatched."""
        job._dataset_service.get_image_by_path.return_value = None
        await job.on_image_started(MagicMock(path="/unknown.jpg"))
        job._event_dispatcher.dispatch.assert_not_called()

    @pytest.mark.asyncio
    async def test_on_image_captioned_increments_processed(self, job):
        """on_image_captioned increments the processed counter."""
        info = _image_info()
        job._dataset_service.get_image_by_path.return_value = info
        job._dataset_service.get_image.return_value = info

        await job.on_image_captioned(MagicMock(path="/img.jpg", caption="hi"), duration_ms=100)

        assert job._processed == 1

    @pytest.mark.asyncio
    async def test_on_image_captioned_dispatches_captioned_and_status(self, job):
        """on_image_captioned dispatches ImageCaptionedEvent then CaptioningStatusEvent."""
        info = _image_info()
        job._dataset_service.get_image_by_path.return_value = info
        job._dataset_service.get_image.return_value = info

        image = MagicMock(path="/img.jpg", caption="hello")
        await job.on_image_captioned(image, duration_ms=120)

        dispatched = [c[0][0] for c in job._event_dispatcher.dispatch.call_args_list]
        assert len(dispatched) == 2
        assert isinstance(dispatched[0], ImageCaptionedEvent)
        assert dispatched[0].caption == "hello"
        assert dispatched[0].duration_ms == 120
        assert isinstance(dispatched[1], CaptioningStatusEvent)
        assert dispatched[1].processed == 1
        # The status event carries the configured max_concurrent
        # (default 1 in the test fixture) so the frontend can compute
        # wall-clock ETA under parallel runs.
        assert dispatched[1].max_concurrent == 1

    @pytest.mark.asyncio
    async def test_on_image_captioned_includes_elapsed_when_started(self, job):
        """Status event includes a non-zero ``elapsed`` once the job has started."""
        # Simulate ``_arun`` having set the start timestamp.
        job._started_at = time.monotonic() - 12.5

        info = _image_info()
        job._dataset_service.get_image_by_path.return_value = info
        job._dataset_service.get_image.return_value = info

        await job.on_image_captioned(MagicMock(path="/img.jpg", caption="hi"), duration_ms=100)

        status_event = job._event_dispatcher.dispatch.call_args_list[-1][0][0]
        assert isinstance(status_event, CaptioningStatusEvent)
        # 12.5s elapsed at the start of the callback, then the callback
        # itself runs for a few ms; we just check it's in the right
        # ballpark.
        assert 12.4 <= status_event.elapsed < 13.0

    @pytest.mark.asyncio
    async def test_status_event_elapsed_is_zero_before_start(self, job):
        """A status event before ``_arun`` has run reports ``elapsed=0``."""
        # _started_at is None until _arun runs.
        assert job._started_at is None

        info = _image_info()
        job._dataset_service.get_image_by_path.return_value = info
        job._dataset_service.get_image.return_value = info

        # ``snapshot()`` (used by external callers) emits a status event
        # through ``_emit_status``.
        await job._emit_status()

        status_event = job._event_dispatcher.dispatch.call_args_list[-1][0][0]
        assert isinstance(status_event, CaptioningStatusEvent)
        assert status_event.elapsed == 0.0
        assert status_event.max_concurrent == 1

    @pytest.mark.asyncio
    async def test_on_image_captioned_refreshes_image_index(self, job):
        """on_image_captioned refreshes the image index after saving."""
        info = _image_info()
        job._dataset_service.get_image_by_path.return_value = info
        job._dataset_service.get_image.return_value = info

        await job.on_image_captioned(MagicMock(path="/img.jpg"), duration_ms=100)

        job._dataset_service.refresh_image_index.assert_called_once_with("test_ds", 42)

    @pytest.mark.asyncio
    async def test_on_image_error_increments_and_records(self, job):
        """on_image_error increments errors and records the message."""
        await job.on_image_error(MagicMock(path="/img.jpg"), "boom", duration_ms=50)

        assert job._errors == 1
        assert job._error_messages == ["boom"]

    @pytest.mark.asyncio
    async def test_on_image_error_dispatches_error_and_status(self, job):
        """on_image_error dispatches ImageCaptionErrorEvent then CaptioningStatusEvent."""
        job._dataset_service.get_image_by_path.return_value = None
        await job.on_image_error(MagicMock(path="/unknown.jpg"), "boom", duration_ms=50)

        dispatched = [c[0][0] for c in job._event_dispatcher.dispatch.call_args_list]
        assert len(dispatched) == 2
        assert isinstance(dispatched[0], ImageCaptionErrorEvent)
        assert dispatched[0].error == "boom"
        assert dispatched[0].duration_ms == 50
        assert dispatched[0].image_id == -1
        assert isinstance(dispatched[1], CaptioningStatusEvent)
        assert dispatched[1].errors == 1

    @pytest.mark.asyncio
    async def test_on_image_error_logs_warning(self, job):
        """on_image_error logs a warning with the image path and error."""
        await job.on_image_error(MagicMock(path="/img.jpg"), "boom", duration_ms=50)
        job._logger.warning.assert_called_once()
        # log message is positional %-formatted — check the format string
        # and the positional args separately (see logging-format memory).
        fmt, path_arg, _ = job._logger.warning.call_args[0]
        assert path_arg == "/img.jpg"
        assert "boom" in (fmt % (path_arg, "boom"))


# ---------------------------------------------------------------------------
# expected_change_registrar
# ---------------------------------------------------------------------------


class TestExpectedChangeRegistrar:
    """_expected_change_registrar — calls dataset_watcher.expect_file_change per path."""

    def test_registers_each_path(self, job):
        paths = ["/a.txt", "/b.toml", "/c.history~"]
        job._expected_change_registrar(paths)

        assert job._dataset_watcher.expect_file_change.call_count == 3
        for path in paths:
            job._dataset_watcher.expect_file_change.assert_any_call("test_ds", path)

    def test_empty_paths(self, job):
        job._expected_change_registrar([])
        job._dataset_watcher.expect_file_change.assert_not_called()


# ---------------------------------------------------------------------------
# _ado_run integration with CaptioningRunner
# ---------------------------------------------------------------------------


class TestAdoRunWithRunner:
    """_ado_run — uses CaptioningRunner + processes images + dispatches status events."""

    @pytest.fixture
    def ado_run_env(self, tmp_path):
        """Set up the environment for _ado_run tests.

        Yields ``(config_path, mock_instance, MockRunner, mock_load)`` where
        ``config_path`` is a real file the dataset_service mock can return
        (so the real ``preflight_images`` passes its path check), and the
        other two are the patched CaptioningRunner and load_dataset_config.
        """
        config_path = tmp_path / "config.toml"
        config_path.write_text('[api]\nurl = "x"\nmodel_name = "m"\n[prompt]\ntemplate = "t"\n')

        with patch(_PATCH_CAPTIONING_RUNNER) as MockRunner, patch(_PATCH_LOAD_DATASET_CONFIG) as mock_load:
            mock_instance = MagicMock()
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_instance.caption_images = AsyncMock()
            MockRunner.return_value = mock_instance
            yield config_path, mock_instance, MockRunner, mock_load

    @pytest.mark.asyncio
    async def test_uses_captioning_runner(self, job, ado_run_env):
        """_ado_run builds a CaptioningRunner with the parsed config and self as callbacks."""
        config_path, mock_instance, MockRunner, mock_load = ado_run_env
        job._dataset_service.get_dataset.return_value = MagicMock(config_path=str(config_path))
        mock_config = MagicMock()
        mock_load.return_value = (mock_config, [MagicMock()])

        await job._ado_run()

        MockRunner.assert_called_once()
        args, kwargs = MockRunner.call_args
        assert args[0] is mock_config
        assert args[1] is job._opts
        assert kwargs.get("expected_change_registrar") == job._expected_change_registrar

    @pytest.mark.asyncio
    async def test_caption_images_called_with_all_images(self, job, ado_run_env):
        """_ado_run calls caption_images with the to-do list and self as callbacks."""
        config_path, mock_instance, _, mock_load = ado_run_env
        job._dataset_service.get_dataset.return_value = MagicMock(config_path=str(config_path))
        images = [MagicMock(), MagicMock(), MagicMock()]
        mock_load.return_value = (MagicMock(), images)

        await job._ado_run()

        mock_instance.caption_images.assert_awaited_once()
        args, kwargs = mock_instance.caption_images.call_args
        assert args[0] == images
        assert args[1] is job
        assert kwargs.get("max_concurrent") == 1  # default

    @pytest.mark.asyncio
    async def test_caption_images_passes_max_concurrent(self, job, ado_run_env):
        """_ado_run forwards the CaptionJobOptions.max_concurrent to the runner."""
        from yadc.core.captioning.options import CaptionJobOptions

        config_path, mock_instance, _, mock_load = ado_run_env
        job._dataset_service.get_dataset.return_value = MagicMock(config_path=str(config_path))
        job._opts = CaptionJobOptions(max_concurrent=4)
        mock_load.return_value = (MagicMock(), [MagicMock()])

        await job._ado_run()

        assert mock_instance.caption_images.call_args.kwargs.get("max_concurrent") == 4

    @pytest.mark.asyncio
    async def test_caption_images_receives_desc_sorted_to_do(self, job, ado_run_env):
        """``_ado_run`` passes the SQL id-DESC-sorted list to
        ``caption_images`` so concurrent captioning starts with the
        newest images first. Reorder is driven by a single SQL
        query (``get_image_paths_in_desc_order``), not N+1 callbacks.
        """
        from yadc.core.dataset import DatasetImage

        config_path, mock_instance, _, mock_load = ado_run_env
        job._dataset_service.get_dataset.return_value = MagicMock(config_path=str(config_path))

        # Real loader call returns images in filesystem order; the
        # API service then reorders via the SQL DESC query.
        img_a = DatasetImage(path=str(config_path.parent / "a.jpg"))
        img_b = DatasetImage(path=str(config_path.parent / "b.jpg"))
        img_c = DatasetImage(path=str(config_path.parent / "c.jpg"))

        # Path list returned by the SQL query (newest first). IDs
        # are intentionally non-monotonic with file names so we
        # can prove the sort comes from SQL, not path.
        job._dataset_service.get_image_paths_in_desc_order.return_value = [
            str(img_b.path),
            str(img_c.path),
            str(img_a.path),
        ]

        mock_load.return_value = (MagicMock(), [img_a, img_b, img_c])
        await job._ado_run()

        mock_instance.caption_images.assert_awaited_once()
        args, _ = mock_instance.caption_images.call_args
        assert args[0] == [img_b, img_c, img_a]
        # The reorder uses the SQL helper, not N+1 get_image_by_path calls.
        job._dataset_service.get_image_paths_in_desc_order.assert_called_once_with("test_ds")

    @pytest.mark.asyncio
    async def test_done_status_on_no_images(self, job, ado_run_env):
        """_ado_run early-exits with done status when there are no images."""
        config_path, _, _, mock_load = ado_run_env
        job._dataset_service.get_dataset.return_value = MagicMock(config_path=str(config_path))
        mock_load.return_value = (MagicMock(), [])

        await job._ado_run()

        snap = await job.snapshot()
        assert snap.status == "done"
        assert snap.total == 0

    @pytest.mark.asyncio
    async def test_done_status_after_processing(self, job, ado_run_env):
        """_ado_run sets done status after processing all images without stop."""
        config_path, _, _, mock_load = ado_run_env
        job._dataset_service.get_dataset.return_value = MagicMock(config_path=str(config_path))
        mock_load.return_value = (MagicMock(), [MagicMock(), MagicMock()])

        await job._ado_run()

        snap = await job.snapshot()
        assert snap.status == "done"

    @pytest.mark.asyncio
    async def test_cancelled_status_when_stop_event_set(self, job, ado_run_env):
        """_ado_run sets cancelled status when the stop event is set."""
        config_path, _, _, mock_load = ado_run_env
        job._dataset_service.get_dataset.return_value = MagicMock(config_path=str(config_path))
        mock_load.return_value = (MagicMock(), [MagicMock()])
        job._stop_event.set()

        await job._ado_run()

        snap = await job.snapshot()
        assert snap.status == "cancelled"

    @pytest.mark.asyncio
    async def test_cancellation_propagates(self, job, ado_run_env):
        """_ado_run re-raises CancelledError (does not catch it)."""
        config_path, mock_instance, _, mock_load = ado_run_env
        job._dataset_service.get_dataset.return_value = MagicMock(config_path=str(config_path))
        mock_instance.caption_images = AsyncMock(side_effect=asyncio.CancelledError())
        mock_load.return_value = (MagicMock(), [MagicMock()])

        with pytest.raises(asyncio.CancelledError):
            await job._ado_run()

    @pytest.mark.asyncio
    async def test_batch_aborted_error_propagates(self, job, ado_run_env):
        """_ado_run lets BatchAbortedError through; _arun turns it into status='error'."""
        from yadc.core.captioning.runner import BatchAbortedError

        config_path, mock_instance, _, mock_load = ado_run_env
        job._dataset_service.get_dataset.return_value = MagicMock(config_path=str(config_path))
        mock_instance.caption_images = AsyncMock(side_effect=BatchAbortedError("aborted"))
        mock_load.return_value = (MagicMock(), [MagicMock()])

        with pytest.raises(BatchAbortedError):
            await job._ado_run()


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
        with patch.object(AsyncCaptionJob, "preflight_images", return_value=[]):
            with patch.object(AsyncCaptionJob, "start") as mock_start:
                info = await captioning_service.start_job_async("test_ds", CaptionJobOptions())

        assert info.status == "done"
        assert info.total == 0
        assert info.processed == 0
        mock_start.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_error_on_preflight_failure(self, captioning_service):
        """When preflight raises, start_job_async returns error without starting a task."""
        with patch.object(AsyncCaptionJob, "preflight_images", side_effect=ValueError("bad config")):
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

        with patch.object(AsyncCaptionJob, "preflight_images", return_value=[mock_img]):
            with patch.object(AsyncCaptionJob, "start") as mock_start:
                info = await captioning_service.start_job_async("test_ds", CaptionJobOptions())

        assert info.status == "running"
        assert info.total == 1
        mock_start.assert_called_once()
