"""Tests for DatasetService.preview_prompt — verifying caption and extras are loaded."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from yadc.api.services.datasets import DatasetService, ImageInfo


@pytest.fixture
def service():
    """Create a DatasetService with mocked dependencies."""
    mock_event_dispatcher = MagicMock()
    mock_dataset_watcher = MagicMock()
    mock_logging = MagicMock()
    mock_logging.get_logger.return_value = MagicMock()

    svc = DatasetService.__new__(DatasetService)
    svc._event_dispatcher = mock_event_dispatcher
    svc._dataset_watcher = mock_dataset_watcher
    svc._logger = mock_logging.get_logger()
    return svc


@pytest.fixture
def image_with_caption(tmp_path):
    """Create a test image with a caption .txt file."""
    img_path = tmp_path / "test.jpg"
    # Create a minimal valid JPEG (1x1 pixel)
    from PIL import Image

    img = Image.new("RGB", (1, 1), color="red")
    img.save(img_path, format="JPEG")

    caption_path = img_path.with_suffix(".txt")
    caption_path.write_text("a red square")

    return img_path, caption_path


def _setup_service_get_image(service: DatasetService, image_id: int, image_path: Path):
    """Patch get_image to return an ImageInfo pointing to the given file."""
    info = ImageInfo(
        id=image_id,
        file_name=image_path.name,
        path=str(image_path),
        has_caption=True,
    )
    service.get_image = MagicMock(return_value=info)


def test_preview_prompt_includes_caption(service, image_with_caption):
    """The current caption should be available as {{ caption }} in the template context."""
    img_path, _ = image_with_caption
    _setup_service_get_image(service, image_id=1, image_path=img_path)

    result = service.preview_prompt("test_ds", 1, "")
    assert result is not None
    assert result["template_context"]["caption"] == "a red square"


def test_preview_prompt_empty_caption_when_no_txt(service, image_with_caption):
    """Without a .txt file, the caption should be empty string."""
    img_path, caption_path = image_with_caption
    caption_path.unlink()  # remove caption file
    _setup_service_get_image(service, image_id=1, image_path=img_path)

    result = service.preview_prompt("test_ds", 1, "")
    assert result is not None
    assert result["template_context"]["caption"] == ""


def test_preview_prompt_includes_toml_extras(service, image_with_caption):
    """Extra fields from the TOML sidecar should appear in the template context."""
    img_path, _ = image_with_caption

    # Write TOML sidecar with extras
    toml_path = img_path.with_suffix(".toml")
    toml_path.write_text('custom_field = "hello"\nnumber = 42\n')

    _setup_service_get_image(service, image_id=1, image_path=img_path)

    result = service.preview_prompt("test_ds", 1, "")
    assert result is not None
    assert result["template_context"]["caption"] == "a red square"
    assert result["template_context"]["custom_field"] == "hello"
    assert result["template_context"]["number"] == 42


def test_preview_prompt_includes_drafts(service, image_with_caption):
    """Drafts should appear in the template context."""
    img_path, _ = image_with_caption

    # Write a draft file
    draft_path = img_path.parent / (img_path.stem + ".gemma.draft~")
    draft_path.write_text("draft caption from gemma")

    _setup_service_get_image(service, image_id=1, image_path=img_path)

    result = service.preview_prompt("test_ds", 1, "")
    assert result is not None
    assert result["template_context"]["caption"] == "a red square"
    assert result["template_context"]["drafts"]["gemma"] == "draft caption from gemma"


def test_preview_prompt_with_custom_template(service, image_with_caption):
    """A custom template can reference {{ caption }} and get the current caption."""
    img_path, _ = image_with_caption
    _setup_service_get_image(service, image_id=1, image_path=img_path)

    custom_template = """
{% set system_prompt %}You are a caption refiner.{% endset %}
{% set user_prompt %}Current caption: {{ caption }}
Please improve it.{% endset %}
"""
    result = service.preview_prompt("test_ds", 1, custom_template)
    assert result is not None
    assert "a red square" in result["user_prompt"]
    assert "Current caption:" in result["user_prompt"]


def test_preview_prompt_returns_none_for_missing_image(service):
    """Should return None if the image is not found."""
    service.get_image = MagicMock(return_value=None)
    result = service.preview_prompt("test_ds", 999, "")
    assert result is None


def test_preview_prompt_returns_none_for_missing_file(service, tmp_path):
    """Should return None if the image file doesn't exist on disk."""
    nonexistent = tmp_path / "missing.jpg"
    info = ImageInfo(id=1, file_name="missing.jpg", path=str(nonexistent))
    service.get_image = MagicMock(return_value=info)

    result = service.preview_prompt("test_ds", 1, "")
    assert result is None
