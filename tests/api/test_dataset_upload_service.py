"""Tests for DatasetUploadService — upload validation, writing, and registration."""

import asyncio
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import toml

from yadc.api.configuration import Configuration
from yadc.api.services.dataset_upload import DatasetUploadResult, DatasetUploadService, UploadProgressEvent
from yadc.api.services.datasets import DatasetInfo, DatasetService

# Path to the real test image shipped with the test suite.
TEST_IMAGE_PATH = Path(__file__).parent / "test_data" / "valid_image.png"


# --- Helpers ---


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


def _result_from_events(events: list[UploadProgressEvent]) -> DatasetUploadResult:
    """Extract the DatasetUploadResult from a list of progress events."""
    complete = [e for e in events if e.phase == "complete"]
    assert len(complete) == 1, f"Expected 1 complete event, got {len(complete)}"
    return DatasetUploadResult(dataset=complete[0].dataset, warnings=complete[0].warnings)


def _error_from_events(events: list[UploadProgressEvent]) -> str | None:
    """Extract error message from events, or None."""
    errors = [e for e in events if e.phase == "error"]
    return errors[0].message if errors else None


def run(coro):
    """Run an async coroutine synchronously in tests."""
    with asyncio.Runner() as runner:
        return runner.run(coro)


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
def upload_service(mock_datasets, configuration, mock_logging):
    return DatasetUploadService(
        datasets=mock_datasets,
        configuration=configuration,
        logging=mock_logging,
    )


# Patch STATE_PATH so _dataset_state_dir writes into tmp_path
@pytest.fixture(autouse=True)
def patch_state_path(tmp_path):
    state_path = tmp_path / "state"
    state_path.mkdir()
    with (
        patch("yadc.api.services.dataset_upload._dataset_state_dir") as mock_dir,
        patch("yadc.api.services.dataset_upload._dataset_config_path") as mock_config,
    ):

        def _dir(name: str) -> Path:
            return state_path / name

        def _config(name: str) -> Path:
            return state_path / name / "config.toml"

        mock_dir.side_effect = _dir
        mock_config.side_effect = _config
        yield


# By default, patch _validate_image to always succeed so tests focus on
# upload logic rather than PIL behaviour. Individual tests opt into
# real validation when they explicitly test image/TOML content checks.
@pytest.fixture(autouse=True)
def patch_validate(upload_service):
    with patch.object(upload_service, "_validate_image", return_value=True), patch.object(upload_service, "_validate_toml", return_value=True):
        yield


# --- Validation: name and file checks ---


def test_empty_name_raises(upload_service):
    with pytest.raises(ValueError, match="Dataset name is required"):
        run(_collect(upload_service, "", [_file("a.jpg", _make_bytes())]))


def test_whitespace_name_raises(upload_service):
    with pytest.raises(ValueError, match="Dataset name is required"):
        run(_collect(upload_service, "   ", [_file("a.jpg", _make_bytes())]))


def test_no_files_raises(upload_service):
    with pytest.raises(ValueError, match="At least one file is required"):
        run(_collect(upload_service, "ds", []))


def test_name_is_stripped(upload_service):
    run(_collect(upload_service, "  my_ds  ", [_file("a.jpg", _make_bytes())]))
    upload_service._datasets.register.assert_called_once()
    assert upload_service._datasets.register.call_args[0][0] == "my_ds"


def test_unsupported_extension_raises(upload_service):
    with pytest.raises(ValueError, match="Unsupported file type"):
        run(_collect(upload_service, "ds", [_file("a.exe", _make_bytes(b"malware"))]))


def test_dot_filename_raises(upload_service):
    with pytest.raises(ValueError, match="Unsupported file type|Invalid file path"):
        run(_collect(upload_service, "ds", [_file(".", _make_bytes())]))


def test_dotdot_filename_raises(upload_service):
    with pytest.raises(ValueError, match="Unsupported file type|Invalid file path"):
        run(_collect(upload_service, "ds", [_file("..", _make_bytes())]))


# --- Nested file handling ---


def test_deeply_nested_file_skipped_with_warning(upload_service):
    events = run(
        _collect(
            upload_service,
            "ds",
            [_file("a/b/c.jpg", _make_bytes()), _file("root.jpg", _make_bytes())],
        )
    )
    result = _result_from_events(events)
    assert any("a/b/c.jpg" in w for w in result.warnings)


def test_nested_file_only_results_in_error(upload_service):
    events = run(_collect(upload_service, "ds", [_file("a/b/c.jpg", _make_bytes())]))
    error = _error_from_events(events)
    assert error is not None
    assert "No valid files" in error


# --- Image validation (exercises real _validate_image) ---


def test_valid_image_accepted(upload_service):
    """Patched validation returns True, file is accepted."""
    events = run(_collect(upload_service, "ds", [_file("img.png", _read_test_image())]))
    result = _result_from_events(events)
    assert len(result.warnings) == 0
    upload_service._datasets.register.assert_called_once()


def test_corrupt_image_skipped(upload_service):
    """Patched validation returns False for corrupt images."""
    upload_service._validate_image.return_value = False
    events = run(_collect(upload_service, "ds", [_file("bad.jpg", _make_bytes(b"\x00\x01\x02"))]))
    error = _error_from_events(events)
    assert error is not None
    assert "No valid files" in error


def test_mix_valid_and_corrupt_images(upload_service):
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


# --- TOML validation (exercises real _validate_toml) ---


def test_valid_toml_sidecar_accepted(upload_service):
    events = run(
        _collect(
            upload_service,
            "ds",
            [_file("cat.jpg", _make_bytes()), _file("cat.toml", _make_bytes(b"x = 1\n"))],
        )
    )
    result = _result_from_events(events)
    assert len(result.warnings) == 0


def test_invalid_toml_sidecar_skipped(upload_service):
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


# --- Orphan sidecar checks ---


def test_orphan_txt_skipped(upload_service):
    events = run(
        _collect(
            upload_service,
            "ds",
            [_file("cat.jpg", _make_bytes()), _file("dog.txt", _make_bytes(b"a caption"))],
        )
    )
    result = _result_from_events(events)
    assert any("dog.txt" in w for w in result.warnings)


def test_matching_txt_sidecar_accepted(upload_service):
    events = run(
        _collect(
            upload_service,
            "ds",
            [_file("cat.jpg", _make_bytes()), _file("cat.txt", _make_bytes(b"a caption"))],
        )
    )
    result = _result_from_events(events)
    assert not any("cat.txt" in w for w in result.warnings)


def test_orphan_toml_sidecar_skipped(upload_service):
    events = run(
        _collect(
            upload_service,
            "ds",
            [_file("cat.jpg", _make_bytes()), _file("dog.toml", _make_bytes(b"x = 1\n"))],
        )
    )
    result = _result_from_events(events)
    assert any("dog.toml" in w for w in result.warnings)


def test_orphan_draft_skipped(upload_service):
    events = run(
        _collect(
            upload_service,
            "ds",
            [_file("cat.jpg", _make_bytes()), _file("dog.alt.draft~", _make_bytes(b"draft"))],
        )
    )
    result = _result_from_events(events)
    assert any("dog.alt.draft~" in w for w in result.warnings)


def test_matching_draft_accepted(upload_service):
    events = run(
        _collect(
            upload_service,
            "ds",
            [_file("cat.jpg", _make_bytes()), _file("cat.alt.draft~", _make_bytes(b"draft"))],
        )
    )
    result = _result_from_events(events)
    assert not any("cat.alt.draft~" in w for w in result.warnings)


def test_orphan_history_skipped(upload_service):
    events = run(
        _collect(
            upload_service,
            "ds",
            [_file("cat.jpg", _make_bytes()), _file("dog.history~", _make_bytes(b"entry"))],
        )
    )
    result = _result_from_events(events)
    assert any("dog.history~" in w for w in result.warnings)


def test_all_orphan_sidecars_aborts(upload_service):
    events = run(_collect(upload_service, "ds", [_file("orphan.txt", _make_bytes(b"text"))]))
    error = _error_from_events(events)
    assert error is not None
    assert "No valid files" in error


# --- Folder layout ---


def test_folder_file_written_correctly(upload_service, tmp_path):
    run(_collect(upload_service, "ds", [_file("train/img.jpg", _make_bytes())]))
    state_path = tmp_path / "state"
    assert (state_path / "ds" / "folders" / "train" / "img.jpg").exists()


def test_root_file_written_to_images_dir(upload_service, tmp_path):
    run(_collect(upload_service, "ds", [_file("img.jpg", _make_bytes())]))
    state_path = tmp_path / "state"
    assert (state_path / "ds" / "images" / "img.jpg").exists()


def test_multiple_folders_generate_multiple_dataset_entries(upload_service, tmp_path):
    run(
        _collect(
            upload_service,
            "ds",
            [_file("train/a.jpg", _make_bytes()), _file("val/b.jpg", _make_bytes())],
        )
    )
    config = toml.load(tmp_path / "state" / "ds" / "config.toml")
    paths = [e["path"] for e in config["dataset"]]
    assert len(paths) == 2
    assert any("train" in p for p in paths)
    assert any("val" in p for p in paths)


def test_root_and_folder_combined(upload_service, tmp_path):
    run(
        _collect(
            upload_service,
            "ds",
            [_file("root.jpg", _make_bytes()), _file("train/a.jpg", _make_bytes())],
        )
    )
    config = toml.load(tmp_path / "state" / "ds" / "config.toml")
    paths = [e["path"] for e in config["dataset"]]
    assert len(paths) == 2  # images/ + folders/train


# --- Size limit ---


def test_size_limit_exceeded(upload_service, configuration):
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


def test_size_limit_not_exceeded(upload_service, configuration):
    configuration.max_upload_size_bytes = 1024 * 1024  # 1 MB
    events = run(_collect(upload_service, "ds", [_file("cat.jpg", _make_bytes())]))
    result = _result_from_events(events)
    assert result is not None


# --- Cleanup on failure ---


def test_cleanup_on_register_failure(upload_service, tmp_path):
    upload_service._datasets.register.side_effect = RuntimeError("db error")
    state_path = tmp_path / "state"

    events = run(_collect(upload_service, "ds", [_file("cat.jpg", _make_bytes())]))
    error = _error_from_events(events)
    assert error is not None
    assert "db error" in error

    # The dataset directory should be cleaned up
    assert not (state_path / "ds").exists()


# --- Real validation methods (exercises actual PIL / toml) ---


def test_validate_image_real_valid_image():
    """_validate_image returns True for the shipped test image."""
    svc = DatasetUploadService.__new__(DatasetUploadService)
    from unittest.mock import MagicMock

    svc._logger = MagicMock()
    assert svc._validate_image(_read_test_image(), "img.png") is True


def test_validate_image_real_corrupt():
    """_validate_image returns False for garbage bytes."""
    svc = DatasetUploadService.__new__(DatasetUploadService)
    from unittest.mock import MagicMock

    svc._logger = MagicMock()
    assert svc._validate_image(BytesIO(b"\x00\x01\x02"), "bad.jpg") is False


def test_validate_toml_real_valid():
    """_validate_toml returns True for valid TOML."""
    svc = DatasetUploadService.__new__(DatasetUploadService)
    assert svc._validate_toml(BytesIO(b"x = 1\n"), "test.toml") is True


def test_validate_toml_real_invalid():
    """_validate_toml returns False for invalid TOML."""
    svc = DatasetUploadService.__new__(DatasetUploadService)
    assert svc._validate_toml(BytesIO(b"key = [invalid"), "bad.toml") is False


# --- Return type ---


def test_returns_dataset_upload_result(upload_service):
    events = run(_collect(upload_service, "ds", [_file("cat.jpg", _make_bytes())]))
    result = _result_from_events(events)
    assert isinstance(result, DatasetUploadResult)
    assert isinstance(result.dataset, DatasetInfo)
    assert isinstance(result.warnings, list)


def test_upload_passes_source_to_register(upload_service):
    run(_collect(upload_service, "ds", [_file("cat.jpg", _make_bytes())]))
    assert upload_service._datasets.register.call_args.kwargs["source"] == "upload"


# --- Windows-style paths ---


def test_backslash_converted_to_forward_slash(upload_service, tmp_path):
    events = run(_collect(upload_service, "ds", [_file("train\\img.jpg", _make_bytes())]))
    result = _result_from_events(events)
    assert len(result.warnings) == 0
    state_path = tmp_path / "state"
    assert (state_path / "ds" / "folders" / "train" / "img.jpg").exists()


# --- Progress events ---


def test_validating_events_emitted(upload_service):
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


def test_writing_events_emitted(upload_service):
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


def test_complete_event_emitted(upload_service):
    events = run(_collect(upload_service, "ds", [_file("cat.jpg", _make_bytes())]))
    complete = [e for e in events if e.phase == "complete"]
    assert len(complete) == 1
    assert isinstance(complete[0].dataset, DatasetInfo)
    assert complete[0].warnings == []


def test_skipped_file_still_emits_validating_event(upload_service):
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


def test_event_order(upload_service):
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
