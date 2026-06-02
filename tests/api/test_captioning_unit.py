"""Tests for AsyncCaptionJob._acaption_one — verifying expect_file_change calls before writes."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from PIL import Image

from yadc.api.services.captioning import AsyncCaptionJob, CaptioningService, CaptionJobOptions
from yadc.core.dataset import DatasetImage


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


@pytest.fixture
def image_with_file(tmp_path):
    """Create a minimal image file on disk and return a DatasetImage for it."""
    img_path = tmp_path / "test.jpg"
    img = Image.new("RGB", (1, 1), color="red")
    img.save(img_path, format="JPEG")
    return DatasetImage(path=str(img_path))


async def _fake_stream(caption: str):
    yield caption


def _mock_model(caption: str = "a red square"):
    model = MagicMock()
    model.predict_stream = MagicMock(side_effect=lambda *args, **kwargs: _fake_stream(caption))
    return model


def _mock_settings():
    settings = MagicMock()
    settings.max_tokens = 512
    settings.advanced.assistant_prefill = ""
    return settings


class TestCaptionOneRegistersExpectedFiles:
    """Tests that _caption_one calls expect_file_change for each file it writes."""

    @pytest.mark.asyncio
    async def test_normal_mode_registers_three_files(self, job, image_with_file):
        """Non-draft captioning should register caption, TOML, and history paths."""
        model = _mock_model("a nice caption")
        settings = _mock_settings()

        await job._acaption_one(model, image_with_file, settings, {})

        ds = image_with_file
        expected_calls = [
            call("test_ds", str(ds.caption_path)),
            call("test_ds", str(ds.toml_path)),
            call("test_ds", str(ds.history_path)),
        ]
        job._dataset_watcher.expect_file_change.assert_has_calls(expected_calls)
        assert job._dataset_watcher.expect_file_change.call_count == 3

    @pytest.mark.asyncio
    async def test_draft_mode_registers_draft_path(self, job, image_with_file):
        """Draft captioning should register only the draft path."""
        job._opts.draft = "gemma"
        model = _mock_model("a draft caption")
        settings = _mock_settings()

        await job._acaption_one(model, image_with_file, settings, {})

        draft_path = str(image_with_file.draft_path("gemma"))
        job._dataset_watcher.expect_file_change.assert_called_once_with("test_ds", draft_path)

    @pytest.mark.asyncio
    async def test_empty_caption_no_registrations(self, job, image_with_file):
        """If model returns empty string, no files should be registered."""
        model = _mock_model("")
        settings = _mock_settings()

        result = await job._acaption_one(model, image_with_file, settings, {})

        assert result == ""
        job._dataset_watcher.expect_file_change.assert_not_called()

    @pytest.mark.asyncio
    async def test_normal_mode_writes_files(self, job, image_with_file):
        """Verify that caption, TOML, and history files are actually written."""
        model = _mock_model("hello world")
        settings = _mock_settings()

        await job._acaption_one(model, image_with_file, settings, {})

        ds = image_with_file
        assert ds.caption_path.exists()
        assert ds.caption_path.read_text() == "hello world"
        assert ds.history_path.exists()

    @pytest.mark.asyncio
    async def test_draft_mode_writes_draft_file(self, job, image_with_file):
        """Draft mode should write to the draft file, not the caption file."""
        job._opts.draft = "gemma"
        model = _mock_model("draft text")
        settings = _mock_settings()

        await job._acaption_one(model, image_with_file, settings, {})

        draft_path = image_with_file.draft_path("gemma")
        assert draft_path.exists()
        assert draft_path.read_text() == "draft text"
        # Caption file should NOT be written
        assert not image_with_file.caption_path.exists()

    @pytest.mark.asyncio
    async def test_expected_files_registered_before_write(self, job, image_with_file):
        """expect_file_change should be called BEFORE the actual file write.

        We verify ordering by checking that expect_file_change calls happen
        before the write operations produce side effects.
        """
        model = _mock_model("ordered caption")
        settings = _mock_settings()

        call_order = []

        job._dataset_watcher.expect_file_change = MagicMock(side_effect=lambda ds, p: call_order.append(("expect", p)))

        # Wrap update_caption to detect when writes happen
        original_update = image_with_file.update_caption
        call_order_tracker = call_order

        def tracked_update(caption):
            call_order_tracker.append(("write_caption", str(image_with_file.caption_path)))
            original_update(caption)

        image_with_file.update_caption = tracked_update

        await job._acaption_one(model, image_with_file, settings, {})

        # All expect calls should come before the caption write
        expect_indices = [i for i, (tag, _) in enumerate(call_order) if tag == "expect"]
        write_indices = [i for i, (tag, _) in enumerate(call_order) if tag == "write_caption"]

        assert len(expect_indices) == 3, f"Expected 3 expect calls, got: {call_order}"
        assert len(write_indices) == 1, f"Expected 1 write call, got: {call_order}"
        assert max(expect_indices) < write_indices[0], f"expect_file_change calls should precede writes. Order: {call_order}"


class TestCaptioningServiceCleanupRescan:
    """Tests for CaptioningService._cleanup_async triggering a final rescan."""

    @pytest.fixture
    def captioning_service(self):
        """Create a CaptioningService with mocked dependencies."""
        mock_ds = MagicMock()
        mock_dispatcher = MagicMock()
        mock_watcher = MagicMock()
        mock_logging = MagicMock()
        mock_logging.get_logger.return_value = MagicMock()

        svc = CaptioningService.__new__(CaptioningService)
        svc._dataset_service = mock_ds
        svc._event_dispatcher = mock_dispatcher
        svc._dataset_watcher = mock_watcher
        svc._logger = mock_logging.get_logger()
        svc._async_lock = asyncio.Lock()
        svc._async_jobs = {}
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
    def captioning_service(self, tmp_path):
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
        mock_logging = MagicMock()
        mock_logging.get_logger.return_value = MagicMock()

        svc = CaptioningService.__new__(CaptioningService)
        svc._dataset_service = mock_ds
        svc._event_dispatcher = mock_dispatcher
        svc._dataset_watcher = mock_watcher
        svc._logger = mock_logging.get_logger()
        svc._configuration = MagicMock()
        svc._async_lock = asyncio.Lock()
        svc._async_jobs = {}
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


class TestOverwriteFilter:
    """Tests for the overwrite filter in _ado_run."""

    @pytest.fixture
    def _setup_ado_run(self, job, tmp_path):
        """Return a helper that prepares mocks and runs _ado_run."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[api]
url = "http://test"
model_name = "test-model"

[prompt]
template = "test"
""")

        job._dataset_service.get_dataset.return_value = MagicMock(config_path=str(config_path))
        job._dataset_service.get_image.return_value = MagicMock(path=str(tmp_path / "img.jpg"))
        job._apply_overrides = MagicMock(return_value={})
        job._resolve_template = MagicMock(return_value="test template")
        job._acaption_one = AsyncMock(return_value="caption")

        async def _run(mock_images, *, image_ids=None, overwrite=False, draft=""):
            job._opts.image_ids = image_ids
            job._opts.overwrite = overwrite
            job._opts.draft = draft

            with patch("yadc.api.services.captioning.resolve_dataset", return_value=mock_images):
                with patch("yadc.api.services.captioning.parse_config") as mock_parse:
                    mock_cfg = MagicMock()
                    mock_cfg.dataset = MagicMock()
                    mock_cfg.caption_suffix = ".txt"
                    mock_cfg.prompt.name = ""
                    mock_cfg.prompt.template = "test"
                    mock_cfg.api.url = "http://test"
                    mock_cfg.api.token = ""
                    mock_cfg.api.model_name = "test"
                    mock_cfg.settings.max_tokens = 512
                    mock_cfg.settings.image_quality = "auto"
                    mock_cfg.settings.store_conversation = False
                    mock_cfg.reasoning.enable = False
                    mock_cfg.settings.advanced.model_dump.return_value = {}
                    mock_parse.return_value = mock_cfg

                    with patch("yadc.api.services.captioning.APICaptioner.create", new_callable=AsyncMock) as mock_create:
                        mock_model = AsyncMock()
                        mock_model.log_usage = MagicMock()
                        mock_create.return_value = mock_model
                        await job._ado_run()

            return await job.snapshot()

        return _run

    @pytest.mark.asyncio
    async def test_single_image_mode_bypasses_overwrite(self, _setup_ado_run, tmp_path):
        """When image_ids is set, already-captioned images are still processed."""
        mock_img = MagicMock()
        mock_img.path = str(tmp_path / "img.jpg")
        mock_img.caption_path.exists.return_value = True

        snap = await _setup_ado_run([mock_img], image_ids=[42], overwrite=False)
        assert snap.total == 1
        assert snap.processed == 1

    @pytest.mark.asyncio
    async def test_single_image_mode_bypasses_draft_overwrite(self, _setup_ado_run, tmp_path):
        """When image_ids is set, existing drafts are still re-processed."""
        mock_img = MagicMock()
        mock_img.path = str(tmp_path / "img.jpg")
        mock_img.draft_path.return_value.exists.return_value = True

        snap = await _setup_ado_run([mock_img], image_ids=[42], overwrite=False, draft="test_draft")
        assert snap.total == 1
        assert snap.processed == 1

    @pytest.mark.asyncio
    async def test_batch_mode_respects_overwrite_false(self, _setup_ado_run, tmp_path):
        """Batch mode with overwrite=False skips already-captioned images."""
        mock_img = MagicMock()
        mock_img.path = str(tmp_path / "img.jpg")
        mock_img.caption_path.exists.return_value = True

        snap = await _setup_ado_run([mock_img], overwrite=False)
        assert snap.total == 0
        assert snap.status == "done"
