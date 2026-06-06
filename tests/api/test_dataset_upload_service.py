"""Tests for DatasetUploadService — upload validation, writing, and registration."""

import asyncio
import time
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import tomlkit

from yadc.api.configuration import Configuration
from yadc.api.modules.dataset_watcher import DatasetWatcherService
from yadc.api.services.dataset_upload import DatasetUploadResult, DatasetUploadService, UploadProgressEvent
from yadc.api.services.datasets import DatasetInfo, DatasetService
from yadc.api.services.managed_datasets import ManagedDatasetsService
from yadc.api.services.managed_paths import MANAGED_FOLDERS_PREFIX, MANAGED_IMAGES_PREFIX
from yadc.utils.dict_utils import load_toml_file

# Path to the real test image shipped with the test suite.
TEST_IMAGE_PATH = Path(__file__).parent / "test_data" / "valid_image.png"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_test_image() -> BytesIO:
    """Return a BytesIO containing the real test image."""
    return BytesIO(TEST_IMAGE_PATH.read_bytes())


def _make_bytes(data: bytes = b"") -> BytesIO:
    return BytesIO(data)


def _file(name: str, data: BytesIO) -> tuple[str, BytesIO]:
    return (name, data)


def _fake_dataset_info(name: str = "test", source: str = "import") -> DatasetInfo:
    return DatasetInfo(name=name, source=source, image_count=1)


async def _collect(upload_service, *args, **kwargs) -> list[UploadProgressEvent]:
    """Call create_dataset_from_upload and collect all progress events."""
    events: list[UploadProgressEvent] = []
    async for event in upload_service.create_dataset_from_upload(*args, **kwargs):
        events.append(event)
    return events


async def _collect_append(upload_service, *args, **kwargs) -> list[UploadProgressEvent]:
    """Call append_dataset_from_upload and collect all progress events."""
    events: list[UploadProgressEvent] = []
    async for event in upload_service.append_dataset_from_upload(*args, **kwargs):
        events.append(event)
    return events


async def _collect_commit(upload_service, *args, **kwargs) -> list[UploadProgressEvent]:
    """Call commit_staged_upload and collect all progress events."""
    events: list[UploadProgressEvent] = []
    async for event in upload_service.commit_staged_upload(*args, **kwargs):
        events.append(event)
    return events


def _result_from_events(events: list[UploadProgressEvent]) -> DatasetUploadResult:
    """Extract the DatasetUploadResult from a list of progress events."""
    complete = [e for e in events if e.phase == "complete"]
    assert len(complete) == 1, f"Expected 1 complete event, got {len(complete)}"
    return DatasetUploadResult(dataset=complete[0].dataset, warnings=complete[0].warnings)


def _error_from_events(events: list[UploadProgressEvent]) -> str | None:
    """Extract error message from events, or None."""
    errors = [e for e in events if e.phase == "error"]
    return errors[0].message if errors else None


def _conflicts_from_events(events: list[UploadProgressEvent]) -> tuple[str, list[dict]] | None:
    """Extract (staging_id, conflicts) from events, or None."""
    conflicts = [e for e in events if e.phase == "conflicts"]
    if not conflicts:
        return None
    return conflicts[0].staging_id, conflicts[0].conflicts


def _upload_service_watcher_calls(svc: DatasetUploadService) -> int:
    """Return number of expect_file_change calls on the service's watcher mock."""
    return svc._watcher.expect_file_change.call_count


def run(coro):
    """Run an async coroutine synchronously in tests."""
    with asyncio.Runner() as runner:
        return runner.run(coro)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_datasets():
    svc = MagicMock(spec=DatasetService)
    svc.register.return_value = _fake_dataset_info()
    return svc


@pytest.fixture
def configuration(tmp_path):
    cfg = MagicMock(spec=Configuration)
    cfg.max_upload_size_bytes = 10 * 1024 * 1024  # 10 MB
    return cfg


@pytest.fixture
def mock_logging():
    lg = MagicMock()
    lg.get_logger.return_value = MagicMock()
    return lg


@pytest.fixture
def mock_watcher():
    return MagicMock(spec=DatasetWatcherService)


@pytest.fixture
def upload_service(mock_datasets, mock_watcher, configuration, mock_logging):
    return DatasetUploadService(
        datasets=mock_datasets,
        watcher=mock_watcher,
        configuration=configuration,
        logging=mock_logging,
    )


# Patch DATASETS_DIR so dataset writes go into tmp_path
@pytest.fixture(autouse=True)
def patch_state_path(tmp_path):
    state_path = tmp_path / "state"
    state_path.mkdir()
    datasets_dir = state_path / "datasets"
    datasets_dir.mkdir()
    with (
        patch("yadc.api.services.dataset_upload.DATASETS_DIR", datasets_dir),
        patch("yadc.api.services.dataset_upload._dataset_config_path") as mock_config,
    ):

        def _config(name: str) -> Path:
            return datasets_dir / name / "config.toml"

        mock_config.side_effect = _config
        yield


# By default, patch _validate_image to always succeed so tests focus on
# upload logic rather than PIL behaviour. Individual tests opt into
# real validation when they explicitly test image/TOML content checks.
@pytest.fixture(autouse=True)
def patch_validate(upload_service):
    with patch.object(upload_service, "_validate_image", return_value=True), patch.object(upload_service, "_validate_toml", return_value=True):
        yield


@pytest.fixture(autouse=True)
def patch_validate_append(append_service):
    with patch.object(append_service, "_validate_image", return_value=True), patch.object(append_service, "_validate_toml", return_value=True):
        yield


@pytest.fixture
def managed_dataset(tmp_path):
    """Create a real managed dataset directory and return its info."""
    base = tmp_path / "state" / "datasets" / "managed"
    base.mkdir(parents=True)
    images_dir = base / "images"
    folders_dir = base / "folders"
    images_dir.mkdir()
    folders_dir.mkdir()

    # Create a config.toml (managed datasets use relative paths)
    config_path = base / "config.toml"
    config_path.write_text(tomlkit.dumps({"dataset": [{"path": MANAGED_IMAGES_PREFIX}]}))

    # Pre-populate with an existing image and its sidecar
    (images_dir / "existing.jpg").write_bytes(b"existing image data")
    (images_dir / "existing.toml").write_bytes(b"x = 1\n")

    return {
        "base": base,
        "images_dir": images_dir,
        "folders_dir": folders_dir,
        "config_path": config_path,
    }


@pytest.fixture
def mock_datasets_for_append(managed_dataset):
    """Mock DatasetService that returns the real managed dataset info."""
    svc = MagicMock(spec=DatasetService)
    info = DatasetInfo(
        name="managed",
        source="upload",
        config_path=str(managed_dataset["config_path"]),
    )
    svc.get_dataset.return_value = info
    svc.rescan_dataset.return_value = None
    return svc


@pytest.fixture
def append_service(mock_datasets_for_append, mock_watcher, configuration, mock_logging):
    return DatasetUploadService(
        datasets=mock_datasets_for_append,
        watcher=mock_watcher,
        configuration=configuration,
        logging=mock_logging,
    )


@pytest.fixture
def dataset_service_for_delete(managed_dataset):
    """Create a real DatasetService with mocked DB/watcher for delete tests."""
    mock_db = MagicMock()
    mock_watcher = MagicMock()
    mock_logging = MagicMock()
    mock_logging.get_logger.return_value = MagicMock()

    svc = DatasetService.__new__(DatasetService)
    svc._db = mock_db
    svc._watcher = mock_watcher
    svc._logger = mock_logging.get_logger()
    svc._repo = MagicMock()

    info = DatasetInfo(
        name="managed",
        source="upload",
        config_path=str(managed_dataset["config_path"]),
    )
    svc.get_dataset = MagicMock(return_value=info)
    svc.rescan_dataset = MagicMock(return_value=True)

    return svc


@pytest.fixture
def managed_datasets_service(dataset_service_for_delete, mock_logging):
    """Create a real ManagedDatasetsService wired to the test's DatasetService."""
    svc = ManagedDatasetsService.__new__(ManagedDatasetsService)
    svc._datasets = dataset_service_for_delete
    svc._watcher = dataset_service_for_delete._watcher
    svc._logger = mock_logging.get_logger()
    return svc


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestNameAndFileValidation:
    """Dataset name and file argument validation — empty/whitespace names,
    missing files, unsupported extensions, and path-traversal-like names."""

    def test_empty_name_raises(self, upload_service):
        with pytest.raises(ValueError, match="Dataset name is required"):
            run(_collect(upload_service, "", [_file("a.jpg", _make_bytes())]))

    def test_whitespace_name_raises(self, upload_service):
        with pytest.raises(ValueError, match="Dataset name is required"):
            run(_collect(upload_service, "   ", [_file("a.jpg", _make_bytes())]))

    def test_no_files_raises(self, upload_service):
        with pytest.raises(ValueError, match="At least one file is required"):
            run(_collect(upload_service, "ds", []))

    def test_name_is_stripped(self, upload_service):
        run(_collect(upload_service, "  my_ds  ", [_file("a.jpg", _make_bytes())]))
        upload_service._datasets.register.assert_called_once()
        assert upload_service._datasets.register.call_args[0][0] == "my_ds"

    def test_unsupported_extension_raises(self, upload_service):
        with pytest.raises(ValueError, match="Unsupported file type"):
            run(_collect(upload_service, "ds", [_file("a.exe", _make_bytes(b"malware"))]))

    def test_dot_filename_raises(self, upload_service):
        with pytest.raises(ValueError, match="Unsupported file type|Invalid file path"):
            run(_collect(upload_service, "ds", [_file(".", _make_bytes())]))

    def test_dotdot_filename_raises(self, upload_service):
        with pytest.raises(ValueError, match="Unsupported file type|Invalid file path"):
            run(_collect(upload_service, "ds", [_file("..", _make_bytes())]))


class TestNestedFileHandling:
    """Files with nested directory components in their name."""

    def test_deeply_nested_file_skipped_with_warning(self, upload_service):
        events = run(
            _collect(
                upload_service,
                "ds",
                [_file("a/b/c.jpg", _make_bytes()), _file("root.jpg", _make_bytes())],
            )
        )
        result = _result_from_events(events)
        assert any("a/b/c.jpg" in w for w in result.warnings)

    def test_nested_file_only_results_in_error(self, upload_service):
        events = run(_collect(upload_service, "ds", [_file("a/b/c.jpg", _make_bytes())]))
        error = _error_from_events(events)
        assert error is not None
        assert "No valid files" in error


class TestImageValidation:
    """Image file validation — exercises the real ``_validate_image`` (PIL)
    for some tests and a custom mock-driven validator for others."""

    def test_valid_image_accepted(self, upload_service):
        """Patched validation returns True, file is accepted."""
        events = run(_collect(upload_service, "ds", [_file("img.png", _read_test_image())]))
        result = _result_from_events(events)
        assert len(result.warnings) == 0
        upload_service._datasets.register.assert_called_once()

    def test_corrupt_image_skipped(self, upload_service):
        """Patched validation returns False for corrupt images."""
        upload_service._validate_image.return_value = False
        events = run(_collect(upload_service, "ds", [_file("bad.jpg", _make_bytes(b"\x00\x01\x02"))]))
        error = _error_from_events(events)
        assert error is not None
        assert "No valid files" in error

    def test_mix_valid_and_corrupt_images(self, upload_service):
        """Patched validation: corrupt image returns False, valid returns True."""
        upload_service._validate_image.side_effect = lambda stream, filename: filename == "good.jpg"
        events = run(
            _collect(
                upload_service,
                "ds",
                [_file("good.jpg", _make_bytes()), _file("bad.jpg", _make_bytes())],
            )
        )
        result = _result_from_events(events)
        assert any("bad.jpg" in w for w in result.warnings)
        upload_service._datasets.register.assert_called_once()


class TestTomlValidation:
    """TOML sidecar file validation — exercises the real ``_validate_toml``."""

    def test_valid_toml_sidecar_accepted(self, upload_service):
        events = run(
            _collect(
                upload_service,
                "ds",
                [_file("cat.jpg", _make_bytes()), _file("cat.toml", _make_bytes(b"x = 1\n"))],
            )
        )
        result = _result_from_events(events)
        assert len(result.warnings) == 0

    def test_invalid_toml_sidecar_skipped(self, upload_service):
        upload_service._validate_toml.return_value = False
        events = run(
            _collect(
                upload_service,
                "ds",
                [_file("cat.jpg", _make_bytes()), _file("cat.toml", _make_bytes(b"key = [invalid"))],
            )
        )
        result = _result_from_events(events)
        assert any("cat.toml" in w for w in result.warnings)


class TestOrphanSidecars:
    """Sidecar files (txt, toml, draft, history) without a matching image are
    orphans — they should be skipped, with a warning, unless their image exists."""

    def test_orphan_txt_skipped(self, upload_service):
        events = run(
            _collect(
                upload_service,
                "ds",
                [_file("cat.jpg", _make_bytes()), _file("dog.txt", _make_bytes(b"a caption"))],
            )
        )
        result = _result_from_events(events)
        assert any("dog.txt" in w for w in result.warnings)

    def test_matching_txt_sidecar_accepted(self, upload_service):
        events = run(
            _collect(
                upload_service,
                "ds",
                [_file("cat.jpg", _make_bytes()), _file("cat.txt", _make_bytes(b"a caption"))],
            )
        )
        result = _result_from_events(events)
        assert not any("cat.txt" in w for w in result.warnings)

    def test_orphan_toml_sidecar_skipped(self, upload_service):
        events = run(
            _collect(
                upload_service,
                "ds",
                [_file("cat.jpg", _make_bytes()), _file("dog.toml", _make_bytes(b"x = 1\n"))],
            )
        )
        result = _result_from_events(events)
        assert any("dog.toml" in w for w in result.warnings)

    def test_orphan_draft_skipped(self, upload_service):
        events = run(
            _collect(
                upload_service,
                "ds",
                [_file("cat.jpg", _make_bytes()), _file("dog.alt.draft~", _make_bytes(b"draft"))],
            )
        )
        result = _result_from_events(events)
        assert any("dog.alt.draft~" in w for w in result.warnings)

    def test_matching_draft_accepted(self, upload_service):
        events = run(
            _collect(
                upload_service,
                "ds",
                [_file("cat.jpg", _make_bytes()), _file("cat.alt.draft~", _make_bytes(b"draft"))],
            )
        )
        result = _result_from_events(events)
        assert not any("cat.alt.draft~" in w for w in result.warnings)

    def test_orphan_history_skipped(self, upload_service):
        events = run(
            _collect(
                upload_service,
                "ds",
                [_file("cat.jpg", _make_bytes()), _file("dog.history~", _make_bytes(b"entry"))],
            )
        )
        result = _result_from_events(events)
        assert any("dog.history~" in w for w in result.warnings)

    def test_all_orphan_sidecars_aborts(self, upload_service):
        events = run(_collect(upload_service, "ds", [_file("orphan.txt", _make_bytes(b"text"))]))
        error = _error_from_events(events)
        assert error is not None
        assert "No valid files" in error


class TestFolderLayout:
    """How uploaded files are written to the dataset directory layout:
    root files go to ``images/``, nested files go to ``folders/<subdir>/``."""

    def test_folder_file_written_correctly(self, upload_service, tmp_path):
        run(_collect(upload_service, "ds", [_file("train/img.jpg", _make_bytes())]))
        state_path = tmp_path / "state"
        assert (state_path / "datasets" / "ds" / "folders" / "train" / "img.jpg").exists()

    def test_root_file_written_to_images_dir(self, upload_service, tmp_path):
        run(_collect(upload_service, "ds", [_file("img.jpg", _make_bytes())]))
        state_path = tmp_path / "state"
        assert (state_path / "datasets" / "ds" / "images" / "img.jpg").exists()

    def test_multiple_folders_generate_multiple_dataset_entries(self, upload_service, tmp_path):
        run(
            _collect(
                upload_service,
                "ds",
                [_file("train/a.jpg", _make_bytes()), _file("val/b.jpg", _make_bytes())],
            )
        )
        config = load_toml_file((tmp_path / "state" / "datasets" / "ds" / "config.toml").open())
        paths = [e["path"] for e in config["dataset"]]
        assert len(paths) == 2
        assert any("train" in p for p in paths)
        assert any("val" in p for p in paths)

    def test_root_and_folder_combined(self, upload_service, tmp_path):
        run(
            _collect(
                upload_service,
                "ds",
                [_file("root.jpg", _make_bytes()), _file("train/a.jpg", _make_bytes())],
            )
        )
        config = load_toml_file((tmp_path / "state" / "datasets" / "ds" / "config.toml").open())
        paths = [e["path"] for e in config["dataset"]]
        assert len(paths) == 2  # images/ + folders/train


class TestSizeLimit:
    """``max_upload_size_bytes`` config — the upload is rejected if the
    total written size would exceed the configured limit."""

    def test_size_limit_exceeded(self, upload_service, configuration):
        configuration.max_upload_size_bytes = 10  # 10 bytes
        # Include a valid image so .txt isn't treated as orphan.
        # The image is small but the .txt pushes past the limit during write.
        events = run(
            _collect(
                upload_service,
                "ds",
                [_file("cat.jpg", _make_bytes()), _file("cat.txt", _make_bytes(b"x" * 100))],
            )
        )
        error = _error_from_events(events)
        assert error is not None
        assert "exceeds" in error

    def test_size_limit_not_exceeded(self, upload_service, configuration):
        configuration.max_upload_size_bytes = 1024 * 1024  # 1 MB
        events = run(_collect(upload_service, "ds", [_file("cat.jpg", _make_bytes())]))
        result = _result_from_events(events)
        assert result is not None


class TestCleanupOnFailure:
    """If dataset registration fails after files have been written, the
    half-created dataset directory should be cleaned up."""

    def test_cleanup_on_register_failure(self, upload_service, tmp_path):
        upload_service._datasets.register.side_effect = RuntimeError("db error")
        state_path = tmp_path / "state"

        events = run(_collect(upload_service, "ds", [_file("cat.jpg", _make_bytes())]))
        error = _error_from_events(events)
        assert error is not None
        assert "db error" in error

        # The dataset directory should be cleaned up
        assert not (state_path / "datasets" / "ds").exists()


class TestRealValidationMethods:
    """Exercises the real ``_validate_image`` and ``_validate_toml``
    implementations (no mocking of these methods)."""

    def test_validate_image_real_valid_image(self):
        """_validate_image returns True for the shipped test image."""
        svc = DatasetUploadService.__new__(DatasetUploadService)
        svc._logger = MagicMock()
        assert svc._validate_image(_read_test_image(), "img.png") is True

    def test_validate_image_real_corrupt(self):
        """_validate_image returns False for garbage bytes."""
        svc = DatasetUploadService.__new__(DatasetUploadService)
        svc._logger = MagicMock()
        assert svc._validate_image(BytesIO(b"\x00\x01\x02"), "bad.jpg") is False

    def test_validate_toml_real_valid(self):
        """_validate_toml returns True for valid TOML."""
        svc = DatasetUploadService.__new__(DatasetUploadService)
        assert svc._validate_toml(BytesIO(b"x = 1\n"), "test.toml") is True

    def test_validate_toml_real_invalid(self):
        """_validate_toml returns False for invalid TOML."""
        svc = DatasetUploadService.__new__(DatasetUploadService)
        assert svc._validate_toml(BytesIO(b"key = [invalid"), "bad.toml") is False


class TestReturnType:
    """The successful upload's complete event carries a properly-typed
    ``DatasetUploadResult`` and the upload source is forwarded to the
    dataset registration call."""

    def test_returns_dataset_upload_result(self, upload_service):
        events = run(_collect(upload_service, "ds", [_file("cat.jpg", _make_bytes())]))
        result = _result_from_events(events)
        assert isinstance(result, DatasetUploadResult)
        assert isinstance(result.dataset, DatasetInfo)
        assert isinstance(result.warnings, list)

    def test_upload_passes_source_to_register(self, upload_service):
        run(_collect(upload_service, "ds", [_file("cat.jpg", _make_bytes())]))
        assert upload_service._datasets.register.call_args.kwargs["source"] == "upload"


class TestSourcePropagation:
    """``source`` is forwarded to ``watcher.expect_file_change`` and the
    follow-up ``rescan_dataset`` call so the watcher can suppress echo
    notifications back to the same client tab."""

    def test_create_upload_registers_expected_file_changes_with_source(self, upload_service):
        """A non-empty source is forwarded to watcher.expect_file_change for each file."""
        run(
            _collect(
                upload_service,
                "ds",
                [
                    _file("cat.jpg", _make_bytes()),
                    _file("train/dog.jpg", _make_bytes()),
                ],
                source="ui:tab-1",
            )
        )
        # Two files => two expect_file_change calls
        assert upload_service._watcher.expect_file_change.call_count == 2
        for call in upload_service._watcher.expect_file_change.call_args_list:
            # Each call tags the dataset name and the originating source
            assert call.args[0] == "ds"
            assert call.kwargs["source"] == "ui:tab-1"

    def test_create_upload_registers_no_expected_file_changes_without_source(self, upload_service):
        """With an empty source we skip expect_file_change to avoid suppressing
        notifications for unrelated callers (legacy behavior)."""
        run(_collect(upload_service, "ds", [_file("cat.jpg", _make_bytes())]))
        upload_service._watcher.expect_file_change.assert_not_called()

    def test_append_registers_expected_file_changes_with_source(self, append_service, managed_dataset):
        """Append stages files and registers them with the source."""
        run(
            _collect_append(
                append_service,
                "managed",
                [_file("new.jpg", _make_bytes(b"new image"))],
                source="ui:tab-2",
            )
        )
        # At least one expect_file_change for the staging write
        assert _upload_service_watcher_calls(append_service) >= 1
        for call in append_service._watcher.expect_file_change.call_args_list:
            assert call.kwargs["source"] == "ui:tab-2"

    def test_commit_passes_source_to_rescan(self, append_service, managed_dataset):
        """After a no-conflict auto-commit, the rescan is tagged with the source."""
        run(
            _collect_append(
                append_service,
                "managed",
                [_file("new.jpg", _make_bytes(b"new image"))],
                source="ui:tab-3",
            )
        )
        append_service._datasets.rescan_dataset.assert_called_with("managed", source="ui:tab-3")

    def test_commit_explicit_passes_source_to_rescan(self, append_service, managed_dataset):
        """An explicit commit_staged_upload after conflict resolution tags rescan with source."""
        events = run(
            _collect_append(
                append_service,
                "managed",
                [_file("existing.jpg", _make_bytes(b"new content"))],
                source="ui:tab-4",
            )
        )
        staging_id, _ = _conflicts_from_events(events)

        append_service._datasets.rescan_dataset.reset_mock()
        run(
            _collect_commit(
                append_service,
                "managed",
                staging_id,
                {"existing.jpg": "overwrite"},
                source="ui:tab-4",
            )
        )
        append_service._datasets.rescan_dataset.assert_called_with("managed", source="ui:tab-4")

    def test_commit_no_source_omits_rescan_source(self, append_service, managed_dataset):
        """An empty source flows through as the default empty rescan source."""
        run(
            _collect_append(
                append_service,
                "managed",
                [_file("new.jpg", _make_bytes(b"new image"))],
            )
        )
        append_service._datasets.rescan_dataset.assert_called_with("managed", source="")


class TestWindowsPaths:
    """Windows-style backslash separators in file names get normalized to
    forward slashes so the same folder layout is used on all platforms."""

    def test_backslash_converted_to_forward_slash(self, upload_service, tmp_path):
        events = run(_collect(upload_service, "ds", [_file("train\\img.jpg", _make_bytes())]))
        result = _result_from_events(events)
        assert len(result.warnings) == 0
        state_path = tmp_path / "state"
        assert (state_path / "datasets" / "ds" / "folders" / "train" / "img.jpg").exists()


class TestProgressEvents:
    """``UploadProgressEvent`` lifecycle: validating → writing → complete,
    in the right order, with the right per-file index."""

    def test_validating_events_emitted(self, upload_service):
        events = run(
            _collect(
                upload_service,
                "ds",
                [_file("a.jpg", _make_bytes()), _file("b.png", _make_bytes()), _file("c.gif", _make_bytes())],
            )
        )
        validating = [e for e in events if e.phase == "validating"]
        assert len(validating) == 3
        assert validating[0] == UploadProgressEvent(phase="validating", file="a.jpg", index=1, total=3)
        assert validating[1] == UploadProgressEvent(phase="validating", file="b.png", index=2, total=3)
        assert validating[2] == UploadProgressEvent(phase="validating", file="c.gif", index=3, total=3)

    def test_writing_events_emitted(self, upload_service):
        events = run(
            _collect(
                upload_service,
                "ds",
                [_file("a.jpg", _make_bytes()), _file("b.png", _make_bytes())],
            )
        )
        writing = [e for e in events if e.phase == "writing"]
        assert len(writing) == 2
        assert writing[0] == UploadProgressEvent(phase="writing", file="a.jpg", index=1, total=2)
        assert writing[1] == UploadProgressEvent(phase="writing", file="b.png", index=2, total=2)

    def test_complete_event_emitted(self, upload_service):
        events = run(_collect(upload_service, "ds", [_file("cat.jpg", _make_bytes())]))
        complete = [e for e in events if e.phase == "complete"]
        assert len(complete) == 1
        assert isinstance(complete[0].dataset, DatasetInfo)
        assert complete[0].warnings == []

    def test_skipped_file_still_emits_validating_event(self, upload_service):
        upload_service._validate_image.side_effect = lambda stream, filename: filename == "good.jpg"
        events = run(
            _collect(
                upload_service,
                "ds",
                [_file("good.jpg", _make_bytes()), _file("bad.jpg", _make_bytes())],
            )
        )
        validating = [e for e in events if e.phase == "validating"]
        assert len(validating) == 2  # both files get events, even skipped ones

    def test_event_order(self, upload_service):
        events = run(
            _collect(
                upload_service,
                "ds",
                [_file("a.jpg", _make_bytes()), _file("b.png", _make_bytes())],
            )
        )
        phases = [e.phase for e in events]
        # validating events first, then writing events, then complete
        assert phases == ["validating", "validating", "writing", "writing", "complete"]


class TestAppend:
    """``append_dataset_from_upload`` basic flow — new files auto-commit,
    conflicting files go through the staging/conflict resolution path."""

    def test_append_no_conflicts_auto_commits(self, append_service, managed_dataset):
        """Uploading new files to a dataset with no conflicts auto-commits."""
        events = run(
            _collect_append(
                append_service,
                "managed",
                [_file("new.jpg", _make_bytes(b"new image"))],
            )
        )
        complete = [e for e in events if e.phase == "complete"]
        assert len(complete) == 1

        # File should be in live dir, not in staging
        assert (managed_dataset["images_dir"] / "new.jpg").exists()
        # Only the uuid subdir is removed; empty .staging/ may remain
        staging_subdirs = list((managed_dataset["base"] / ".staging").glob("*"))
        assert len(staging_subdirs) == 0

    def test_append_with_conflicts_emits_conflicts_phase(self, append_service, managed_dataset):
        """Uploading files that already exist emits conflicts phase."""
        events = run(
            _collect_append(
                append_service,
                "managed",
                [_file("existing.jpg", _make_bytes(b"new content"))],
            )
        )
        conflicts = _conflicts_from_events(events)
        assert conflicts is not None
        staging_id, conflict_list = conflicts
        assert len(conflict_list) == 1
        assert conflict_list[0]["file"] == "existing.jpg"

        # Staging dir should still exist
        staging_base = managed_dataset["base"] / ".staging" / staging_id
        assert staging_base.exists()


class TestCommitResolutions:
    """``commit_staged_upload`` resolution policies: skip, overwrite, keep-both."""

    def test_commit_skip_resolution(self, append_service, managed_dataset):
        """Skip resolution leaves the original file unchanged."""
        events = run(
            _collect_append(
                append_service,
                "managed",
                [_file("existing.jpg", _make_bytes(b"new content"))],
            )
        )
        staging_id, _ = _conflicts_from_events(events)

        original = (managed_dataset["images_dir"] / "existing.jpg").read_bytes()
        commit_events = run(
            _collect_commit(
                append_service,
                "managed",
                staging_id,
                {"existing.jpg": "skip"},
            )
        )
        complete = [e for e in events if e.phase == "complete"]
        assert len(complete) == 1 or any(e.phase == "complete" for e in commit_events)

        # Original file unchanged
        assert (managed_dataset["images_dir"] / "existing.jpg").read_bytes() == original
        # Staging cleaned up
        assert not (managed_dataset["base"] / ".staging" / staging_id).exists()

    def test_commit_overwrite_resolution(self, append_service, managed_dataset):
        """Overwrite resolution replaces the original file."""
        events = run(
            _collect_append(
                append_service,
                "managed",
                [_file("existing.jpg", _make_bytes(b"new content"))],
            )
        )
        staging_id, _ = _conflicts_from_events(events)

        commit_events = run(
            _collect_commit(
                append_service,
                "managed",
                staging_id,
                {"existing.jpg": "overwrite"},
            )
        )
        assert any(e.phase == "complete" for e in commit_events)

        # File overwritten
        assert (managed_dataset["images_dir"] / "existing.jpg").read_bytes() == b"new content"

    def test_commit_keep_both_resolution(self, append_service, managed_dataset):
        """Keep-both creates a renamed copy."""
        events = run(
            _collect_append(
                append_service,
                "managed",
                [_file("existing.jpg", _make_bytes(b"new content"))],
            )
        )
        staging_id, _ = _conflicts_from_events(events)

        commit_events = run(
            _collect_commit(
                append_service,
                "managed",
                staging_id,
                {"existing.jpg": "keep_both"},
            )
        )
        assert any(e.phase == "complete" for e in commit_events)

        # Both files exist
        assert (managed_dataset["images_dir"] / "existing.jpg").exists()
        assert (managed_dataset["images_dir"] / "existing_1.jpg").exists()
        assert (managed_dataset["images_dir"] / "existing_1.jpg").read_bytes() == b"new content"


class TestSidecarGrouping:
    """When a conflict includes an image and its sidecars, the resolution
    applies to the whole group so image and sidecars stay in sync."""

    def test_group_resolution_applies_to_sidecars(self, append_service, managed_dataset):
        """When image + sidecar conflict, resolution applies to both."""
        events = run(
            _collect_append(
                append_service,
                "managed",
                [
                    _file("existing.jpg", _make_bytes(b"new image")),
                    _file("existing.toml", _make_bytes(b"y = 2\n")),
                ],
            )
        )
        staging_id, conflicts = _conflicts_from_events(events)
        # Only image shown in conflict list
        assert len(conflicts) == 1
        assert conflicts[0]["file"] == "existing.jpg"

        commit_events = run(
            _collect_commit(
                append_service,
                "managed",
                staging_id,
                {"existing.jpg": "overwrite"},
            )
        )
        assert any(e.phase == "complete" for e in commit_events)

        # Both overwritten
        assert (managed_dataset["images_dir"] / "existing.jpg").read_bytes() == b"new image"
        assert (managed_dataset["images_dir"] / "existing.toml").read_bytes() == b"y = 2\n"

    def test_group_skip_applies_to_sidecars(self, append_service, managed_dataset):
        """Skip resolution on image also skips its sidecars."""
        events = run(
            _collect_append(
                append_service,
                "managed",
                [
                    _file("existing.jpg", _make_bytes(b"new image")),
                    _file("existing.toml", _make_bytes(b"y = 2\n")),
                ],
            )
        )
        staging_id, _ = _conflicts_from_events(events)

        original_img = (managed_dataset["images_dir"] / "existing.jpg").read_bytes()
        original_toml = (managed_dataset["images_dir"] / "existing.toml").read_bytes()

        commit_events = run(
            _collect_commit(
                append_service,
                "managed",
                staging_id,
                {"existing.jpg": "skip"},
            )
        )
        assert any(e.phase == "complete" for e in commit_events)

        # Both unchanged
        assert (managed_dataset["images_dir"] / "existing.jpg").read_bytes() == original_img
        assert (managed_dataset["images_dir"] / "existing.toml").read_bytes() == original_toml

    def test_group_keep_both_renames_sidecars(self, append_service, managed_dataset):
        """Keep-both renames image and all sidecars in sync."""
        events = run(
            _collect_append(
                append_service,
                "managed",
                [
                    _file("existing.jpg", _make_bytes(b"new image")),
                    _file("existing.toml", _make_bytes(b"y = 2\n")),
                ],
            )
        )
        staging_id, _ = _conflicts_from_events(events)

        commit_events = run(
            _collect_commit(
                append_service,
                "managed",
                staging_id,
                {"existing.jpg": "keep_both"},
            )
        )
        assert any(e.phase == "complete" for e in commit_events)

        # Original preserved, renamed copies created
        assert (managed_dataset["images_dir"] / "existing.jpg").exists()
        assert (managed_dataset["images_dir"] / "existing_1.jpg").exists()
        assert (managed_dataset["images_dir"] / "existing_1.jpg").read_bytes() == b"new image"
        assert (managed_dataset["images_dir"] / "existing_1.toml").exists()
        assert (managed_dataset["images_dir"] / "existing_1.toml").read_bytes() == b"y = 2\n"


class TestSidecarOnlyUploads:
    """Uploads that contain only sidecar files (no image) for either an
    existing or non-existing image."""

    def test_sidecar_only_for_existing_image_accepted(self, append_service, managed_dataset):
        """Uploading just a sidecar for an existing image is accepted (orphan check passes)."""
        events = run(
            _collect_append(
                append_service,
                "managed",
                [_file("existing.toml", _make_bytes(b"z = 3\n"))],
            )
        )
        # Should conflict since existing.toml already exists
        conflicts = _conflicts_from_events(events)
        if conflicts:
            staging_id, conflict_list = conflicts
            assert any(c["file"] == "existing.toml" for c in conflict_list)
        else:
            # No conflict if the original didn't have existing.toml (but we created it in fixture)
            complete = [e for e in events if e.phase == "complete"]
            assert len(complete) == 1

    def test_sidecar_only_for_missing_image_dropped(self, append_service, managed_dataset):
        """Uploading just a sidecar with no matching image anywhere is dropped as orphan."""
        events = run(
            _collect_append(
                append_service,
                "managed",
                [_file("nonexistent.toml", _make_bytes(b"z = 3\n"))],
            )
        )
        error = _error_from_events(events)
        assert error is not None
        assert "No valid files" in error


class TestStagingCleanup:
    """``_cleanup_staging_dirs`` — removes staging directories older than 24h,
    preserves fresh ones."""

    def test_cleanup_staging_dirs_removes_old(self, append_service, managed_dataset):
        """Cleanup removes staging directories older than 24h."""
        # Create an old staging dir
        staging_base = managed_dataset["base"] / ".staging" / "old-staging"
        staging_base.mkdir(parents=True)
        (staging_base / "dummy.txt").write_text("x")

        # Set its mtime to 25 hours ago
        old_time = time.time() - (25 * 3600)
        (staging_base / "dummy.txt").touch()
        staging_base.touch()
        import os

        os.utime(staging_base, (old_time, old_time))

        append_service._cleanup_staging_dirs()

        assert not staging_base.exists()

    def test_cleanup_staging_dirs_keeps_recent(self, append_service, managed_dataset):
        """Cleanup preserves staging directories newer than 24h."""
        staging_base = managed_dataset["base"] / ".staging" / "fresh-staging"
        staging_base.mkdir(parents=True)
        (staging_base / "dummy.txt").write_text("x")

        append_service._cleanup_staging_dirs()

        assert staging_base.exists()


class TestFolderAppend:
    """Appending files to a dataset's folder subdirectory: a new folder
    auto-commits, an existing folder's file conflicts and goes through
    conflict resolution."""

    def test_append_new_folder_no_conflict(self, append_service, managed_dataset):
        """Uploading files to a new folder auto-commits without conflict."""
        events = run(
            _collect_append(
                append_service,
                "managed",
                [_file("train/new.jpg", _make_bytes(b"train image"))],
            )
        )
        complete = [e for e in events if e.phase == "complete"]
        assert len(complete) == 1

        assert (managed_dataset["folders_dir"] / "train" / "new.jpg").exists()

    def test_append_existing_folder_file_conflicts(self, append_service, managed_dataset):
        """Uploading a file to an existing folder with same name conflicts."""
        # Pre-create a folder with a file
        (managed_dataset["folders_dir"] / "train").mkdir()
        (managed_dataset["folders_dir"] / "train" / "img.jpg").write_bytes(b"old train img")

        events = run(
            _collect_append(
                append_service,
                "managed",
                [_file("train/img.jpg", _make_bytes(b"new train img"))],
            )
        )
        conflicts = _conflicts_from_events(events)
        assert conflicts is not None
        staging_id, conflict_list = conflicts
        assert any(c["file"] == "train/img.jpg" for c in conflict_list)

        # Commit with overwrite
        commit_events = run(
            _collect_commit(
                append_service,
                "managed",
                staging_id,
                {"train/img.jpg": "overwrite"},
            )
        )
        assert any(e.phase == "complete" for e in commit_events)
        assert (managed_dataset["folders_dir"] / "train" / "img.jpg").read_bytes() == b"new train img"


class TestDeleteItems:
    """``ManagedDatasetsService.delete_items`` — remove images, sidecars,
    folders, and reject non-managed / root paths."""

    def test_delete_image_and_sidecars(self, managed_dataset, managed_datasets_service):
        """Deleting an image also removes its sidecars."""
        # Add a sidecar
        (managed_dataset["images_dir"] / "existing.txt").write_text("a caption")

        deleted, warnings = managed_datasets_service.delete_items("managed", [f"{MANAGED_IMAGES_PREFIX}/existing.jpg"])
        assert f"{MANAGED_IMAGES_PREFIX}/existing.jpg" in deleted
        assert len(warnings) == 0

        # Image and sidecar gone
        assert not (managed_dataset["images_dir"] / "existing.jpg").exists()
        assert not (managed_dataset["images_dir"] / "existing.txt").exists()
        # TOML sidecar from fixture also gone
        assert not (managed_dataset["images_dir"] / "existing.toml").exists()

    def test_delete_folder(self, managed_dataset, managed_datasets_service):
        """Deleting a folder removes the entire directory and its config entry."""
        # Pre-create a folder with files and add it to config
        train_dir = managed_dataset["folders_dir"] / "train"
        train_dir.mkdir()
        (train_dir / "img.jpg").write_bytes(b"train img")
        (train_dir / "img.txt").write_text("train caption")

        config_path = managed_dataset["config_path"]
        config_path.write_text(tomlkit.dumps({"dataset": [{"path": MANAGED_IMAGES_PREFIX}, {"path": f"{MANAGED_FOLDERS_PREFIX}/train"}]}))

        deleted, warnings = managed_datasets_service.delete_items("managed", [f"{MANAGED_FOLDERS_PREFIX}/train"])
        assert f"{MANAGED_FOLDERS_PREFIX}/train" in deleted
        assert len(warnings) == 0

        assert not train_dir.exists()

        # Config should no longer reference the deleted folder
        config = load_toml_file(config_path.open())
        paths = [e["path"] for e in config["dataset"]]
        assert f"{MANAGED_FOLDERS_PREFIX}/train" not in paths
        assert MANAGED_IMAGES_PREFIX in paths

    def test_delete_missing_path_warns(self, managed_dataset, managed_datasets_service):
        """Deleting a non-existent path returns a warning."""
        deleted, warnings = managed_datasets_service.delete_items("managed", [f"{MANAGED_IMAGES_PREFIX}/nonexistent.jpg"])
        assert len(deleted) == 0
        assert any("nonexistent.jpg" in w for w in warnings)

    def test_delete_non_managed_dataset_rejects(self, managed_dataset, managed_datasets_service):
        """Delete is rejected for non-managed datasets."""
        info = DatasetInfo(name="external", source="import", config_path="/tmp/fake/config.toml")
        managed_datasets_service._datasets.get_dataset.return_value = info

        with pytest.raises(ValueError, match="not a managed dataset"):
            managed_datasets_service.delete_items("external", [f"{MANAGED_IMAGES_PREFIX}/foo.jpg"])

    def test_delete_mixed_batch(self, managed_dataset, managed_datasets_service):
        """A batch can delete both files and folders."""
        # Setup
        (managed_dataset["images_dir"] / "extra.jpg").write_bytes(b"extra")
        train_dir = managed_dataset["folders_dir"] / "train"
        train_dir.mkdir()
        (train_dir / "img.jpg").write_bytes(b"train")

        deleted, warnings = managed_datasets_service.delete_items(
            "managed", [f"{MANAGED_IMAGES_PREFIX}/extra.jpg", f"{MANAGED_FOLDERS_PREFIX}/train", f"{MANAGED_IMAGES_PREFIX}/missing.jpg"]
        )

        assert f"{MANAGED_IMAGES_PREFIX}/extra.jpg" in deleted
        assert f"{MANAGED_FOLDERS_PREFIX}/train" in deleted
        assert f"{MANAGED_IMAGES_PREFIX}/missing.jpg" not in deleted
        assert any("missing.jpg" in w for w in warnings)

        assert not (managed_dataset["images_dir"] / "extra.jpg").exists()
        assert not train_dir.exists()

    def test_delete_root_images_folder_rejected(self, managed_dataset, managed_datasets_service):
        """Deleting the root images folder is explicitly rejected."""
        deleted, warnings = managed_datasets_service.delete_items("managed", [MANAGED_IMAGES_PREFIX])
        assert len(deleted) == 0
        assert any("Cannot delete root images folder" in w for w in warnings)

    def test_delete_subfolder_named_images(self, managed_dataset, managed_datasets_service):
        """A subfolder literally named 'images' can be deleted via prefixed path."""
        images_subdir = managed_dataset["folders_dir"] / "images"
        images_subdir.mkdir()
        (images_subdir / "sub.jpg").write_bytes(b"sub")

        config_path = managed_dataset["config_path"]
        config_path.write_text(tomlkit.dumps({"dataset": [{"path": MANAGED_IMAGES_PREFIX}, {"path": f"{MANAGED_FOLDERS_PREFIX}/images"}]}))

        deleted, warnings = managed_datasets_service.delete_items("managed", [f"{MANAGED_FOLDERS_PREFIX}/images"])
        assert f"{MANAGED_FOLDERS_PREFIX}/images" in deleted
        assert len(warnings) == 0
        assert not images_subdir.exists()

        config = load_toml_file(config_path.open())
        paths = [e["path"] for e in config["dataset"]]
        assert f"{MANAGED_FOLDERS_PREFIX}/images" not in paths
        assert MANAGED_IMAGES_PREFIX in paths


class TestFolderListing:
    """``ManagedDatasetsService.list_folders`` — root images entry plus
    subfolders, with image counts and a ``can_delete`` flag."""

    def test_list_folders(self, managed_dataset, managed_datasets_service):
        """list_folders returns root images and subfolders with correct counts."""
        # Setup: root image, subfolder with image
        (managed_dataset["images_dir"] / "root.jpg").write_bytes(b"root")
        sub_dir = managed_dataset["folders_dir"] / "train"
        sub_dir.mkdir()
        (sub_dir / "sub.jpg").write_bytes(b"sub")

        folders = managed_datasets_service.list_folders("managed")
        assert len(folders) == 2

        # Root entry (fixture already has existing.jpg)
        root = folders[0]
        assert root["name"] == MANAGED_IMAGES_PREFIX
        assert root["path"] == MANAGED_IMAGES_PREFIX
        assert root["can_delete"] is False
        assert root["image_count"] == 2  # existing.jpg + root.jpg

        # Subfolder entry
        sub = folders[1]
        assert sub["name"] == "train"
        assert sub["path"] == f"{MANAGED_FOLDERS_PREFIX}/train"
        assert sub["can_delete"] is True
        assert sub["image_count"] == 1

    def test_list_folders_paths_match_delete_items(self, managed_dataset, managed_datasets_service):
        """Paths returned by list_folders work correctly with delete_items."""
        sub_dir = managed_dataset["folders_dir"] / "train"
        sub_dir.mkdir()
        (sub_dir / "sub.jpg").write_bytes(b"sub")

        folders = managed_datasets_service.list_folders("managed")
        train_folder = next(f for f in folders if f["name"] == "train")

        # Delete using the path from list_folders
        deleted, warnings = managed_datasets_service.delete_items("managed", [train_folder["path"]])
        assert f"{MANAGED_FOLDERS_PREFIX}/train" in deleted
        assert len(warnings) == 0
        assert not sub_dir.exists()
