"""Tests for DatasetService — ``list_images`` cursor decoding, ``preview_prompt`` context, and dispatch logic for ``DatasetChangedEvent``.

Tests for ``update_extras`` / ``update_caption`` / ``get_history`` live in
:class:`TestUpdateExtrasHistoryRoundTrip` and use a real ``DatasetService``
with a real database to exercise the on-disk history sidecar round-trip.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from yadc.api.events import DatasetChangedEvent
from yadc.api.modules import DatasetWatcherService, DBConnectionFactory, EventDispatcher
from yadc.api.services import DatasetLoader, DatasetScanner
from yadc.api.services.dataset_repository import DatasetRepository, ImageInfo
from yadc.api.services.datasets import DatasetService


@pytest.fixture
def service(test_configuration, logging_factory):
    return DatasetService(
        db=MagicMock(spec=DBConnectionFactory),
        watcher=MagicMock(spec=DatasetWatcherService),
        configuration=test_configuration,
        event_dispatcher=MagicMock(spec=EventDispatcher),
        logging=logging_factory,
        repo=MagicMock(spec=DatasetRepository),
        scanner=MagicMock(spec=DatasetScanner),
        loader=MagicMock(spec=DatasetLoader),
    )


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


class TestListImages:
    """``list_images`` — opaque ``next`` cursor decoding."""

    def test_first_page_passes_unbounded_cursor(self, service):
        """No ``next`` token means "first page" — repo gets an
        effectively-unbounded ``before_id`` (max int) so all rows
        pass the ``id < ?`` filter."""
        page = service.list_images("alpha", limit=10)

        service._repo.list_images.assert_called_once()
        _, kwargs = service._repo.list_images.call_args
        # The repo receives the largest possible int (2**63 - 1) so
        # ``id < before_id`` is true for every row.
        assert kwargs["before_id"] == 2**63 - 1
        assert kwargs["limit"] == 11  # limit + 1
        assert page.next_token is None

    def test_next_token_decoded_to_before_id(self, service):
        """An opaque ``next`` token is decoded into the internal
        ``before_id`` cursor; the client never sees the encoding."""
        service._repo.list_images.return_value = []  # no more pages
        service.list_images("alpha", limit=10, next="42")

        _, kwargs = service._repo.list_images.call_args
        assert kwargs["before_id"] == 42

    def test_next_token_emitted_from_smallest_id(self, service):
        """When there's a next page, the returned ``next_token`` is
        the smallest id on the current page (the last image in DESC
        order) so the client can use it as the next cursor."""
        # Simulate limit+1 rows returned (2 more than asked, so a
        # next page exists). The repo returns the full list; the
        # service trims to ``limit`` and emits a token.
        rows = []
        for i in [50, 49, 48]:  # ids descending
            info = MagicMock(spec=ImageInfo)
            info.id = i
            rows.append(info)
        service._repo.list_images.return_value = rows

        page = service.list_images("alpha", limit=2, next="100")

        # Trimmed to first 2; token = smallest id on the trimmed page.
        assert len(page.images) == 2
        assert page.next_token == "49"

    def test_empty_token_means_first_page(self, service):
        """An empty ``next`` string is treated as "first page"."""
        service.list_images("alpha", limit=10, next="")

        _, kwargs = service._repo.list_images.call_args
        assert kwargs["before_id"] == 2**63 - 1


class TestPreviewPrompt:
    """DatasetService.preview_prompt — verify the template context is populated
    with caption, TOML extras, and drafts as expected."""

    def _setup_service_get_image(self, service: DatasetService, image_id: int, image_path: Path):
        """Patch get_image to return an ImageInfo pointing to the given file."""
        info = ImageInfo(
            id=image_id,
            file_name=image_path.name,
            path=str(image_path),
            has_caption=True,
        )
        service.get_image = MagicMock(return_value=info)

    def test_includes_caption(self, service, image_with_caption):
        """The current caption should be available as {{ caption }} in the template context."""
        img_path, _ = image_with_caption
        self._setup_service_get_image(service, image_id=1, image_path=img_path)

        result = service.preview_prompt("test_ds", 1, "")
        assert result is not None
        assert result["template_context"]["caption"] == "a red square"

    def test_empty_caption_when_no_txt(self, service, image_with_caption):
        """Without a .txt file, the caption should be empty string."""
        img_path, caption_path = image_with_caption
        caption_path.unlink()  # remove caption file
        self._setup_service_get_image(service, image_id=1, image_path=img_path)

        result = service.preview_prompt("test_ds", 1, "")
        assert result is not None
        assert result["template_context"]["caption"] == ""

    def test_includes_toml_extras(self, service, image_with_caption):
        """Extra fields from the TOML sidecar should appear in the template context."""
        img_path, _ = image_with_caption

        # Write TOML sidecar with extras
        toml_path = img_path.with_suffix(".toml")
        toml_path.write_text('custom_field = "hello"\nnumber = 42\n')

        self._setup_service_get_image(service, image_id=1, image_path=img_path)

        result = service.preview_prompt("test_ds", 1, "")
        assert result is not None
        assert result["template_context"]["caption"] == "a red square"
        assert result["template_context"]["custom_field"] == "hello"
        assert result["template_context"]["number"] == 42

    def test_includes_drafts(self, service, image_with_caption):
        """Drafts should appear in the template context."""
        img_path, _ = image_with_caption

        # Write a draft file
        draft_path = img_path.parent / (img_path.stem + ".gemma.draft~")
        draft_path.write_text("draft caption from gemma")

        self._setup_service_get_image(service, image_id=1, image_path=img_path)

        result = service.preview_prompt("test_ds", 1, "")
        assert result is not None
        assert result["template_context"]["caption"] == "a red square"
        assert result["template_context"]["drafts"]["gemma"] == "draft caption from gemma"

    def test_with_custom_template(self, service, image_with_caption):
        """A custom template can reference {{ caption }} and get the current caption."""
        img_path, _ = image_with_caption
        self._setup_service_get_image(service, image_id=1, image_path=img_path)

        custom_template = """
{% set system_prompt %}You are a caption refiner.{% endset %}
{% set user_prompt %}Current caption: {{ caption }}
Please improve it.{% endset %}
"""
        result = service.preview_prompt("test_ds", 1, custom_template)
        assert result is not None
        assert "a red square" in result["user_prompt"]
        assert "Current caption:" in result["user_prompt"]

    def test_returns_none_for_missing_image(self, service):
        """Should return None if the image is not found."""
        service.get_image = MagicMock(return_value=None)
        result = service.preview_prompt("test_ds", 999, "")
        assert result is None

    def test_returns_none_for_missing_file(self, service, tmp_path):
        """Should return None if the image file doesn't exist on disk."""
        nonexistent = tmp_path / "missing.jpg"
        info = ImageInfo(id=1, file_name="missing.jpg", path=str(nonexistent))
        service.get_image = MagicMock(return_value=info)

        result = service.preview_prompt("test_ds", 1, "")
        assert result is None


class TestOnDatasetChangedSkipsSelfOriginated:
    """Tests for _on_dataset_changed dispatching to scanner vs skipping.

    The service no longer does the disk walk itself — it delegates to
    :class:`DatasetScanner`. These tests verify the service's branch
    logic (which event types route to which scanner method) by
    mocking the scanner and the repo's ``get_dataset_row``.
    """

    def test_skips_rescan_when_job_id_is_real(self, service):
        """Events with a real job_id (captioning) should not trigger rescan."""
        service._scanner.scan_disk = MagicMock(return_value=False)
        service._scanner.scan_targeted = MagicMock(return_value=False)
        event = DatasetChangedEvent(dataset_name="test_ds", job_id="abc123")

        service._on_dataset_changed(event)

        service._scanner.scan_disk.assert_not_called()
        service._scanner.scan_targeted.assert_not_called()

    def test_skips_rescan_when_job_id_is_self(self, service):
        """Events tagged as SELF_JOB_ID (webui edits) should not trigger rescan."""
        service._scanner.scan_disk = MagicMock(return_value=False)
        service._scanner.scan_targeted = MagicMock(return_value=False)
        event = DatasetChangedEvent(dataset_name="test_ds", job_id="self")

        service._on_dataset_changed(event)

        service._scanner.scan_disk.assert_not_called()
        service._scanner.scan_targeted.assert_not_called()

    def test_full_scan_when_job_id_is_none(self, service):
        """Events with no job_id and no changed_paths fall back to a full disk scan."""
        service._scanner.scan_disk = MagicMock(return_value=False)
        service._scanner.scan_targeted = MagicMock(return_value=False)
        service._repo.get_dataset_row = MagicMock(return_value=(1, "/tmp/cfg.toml"))
        event = DatasetChangedEvent(dataset_name="test_ds", job_id=None)

        service._on_dataset_changed(event)

        service._scanner.scan_disk.assert_called_once_with(1, "/tmp/cfg.toml")
        service._scanner.scan_targeted.assert_not_called()

    def test_full_scan_when_job_id_is_empty(self, service):
        """Events with empty string job_id fall back to a full disk scan."""
        service._scanner.scan_disk = MagicMock(return_value=False)
        service._scanner.scan_targeted = MagicMock(return_value=False)
        service._repo.get_dataset_row = MagicMock(return_value=(1, "/tmp/cfg.toml"))
        event = DatasetChangedEvent(dataset_name="test_ds", job_id="")

        service._on_dataset_changed(event)

        service._scanner.scan_disk.assert_called_once_with(1, "/tmp/cfg.toml")
        service._scanner.scan_targeted.assert_not_called()

    def test_targeted_update_when_changed_paths_present(self, service):
        """Events with changed_paths use the targeted update path, not the full scan."""
        service._scanner.scan_disk = MagicMock(return_value=False)
        service._scanner.scan_targeted = MagicMock(return_value=False)
        service._repo.get_dataset_row = MagicMock(return_value=(1, "/tmp/cfg.toml"))
        event = DatasetChangedEvent(
            dataset_name="test_ds",
            job_id=None,
            changed_paths=["/data/images/photo.jpg"],
        )

        service._on_dataset_changed(event)

        service._scanner.scan_targeted.assert_called_once_with(1, ["/data/images/photo.jpg"])
        service._scanner.scan_disk.assert_not_called()

    def test_skips_when_dataset_not_found(self, service):
        """Unknown dataset name should be a no-op (no crash, no scan)."""
        service._scanner.scan_disk = MagicMock(return_value=False)
        service._scanner.scan_targeted = MagicMock(return_value=False)
        service._repo.get_dataset_row = MagicMock(return_value=None)
        event = DatasetChangedEvent(dataset_name="missing", job_id=None, changed_paths=["/data/x.jpg"])

        service._on_dataset_changed(event)

        service._scanner.scan_disk.assert_not_called()
        service._scanner.scan_targeted.assert_not_called()


class TestUpdateExtrasHistoryRoundTrip:
    """Regression tests for the webui extras update flow.

    The webui loads current extras via ``tomlkit.loads()`` and passes them
    to ``DatasetImage.model_validate()``. Because tomlkit wraps values in
    its own types (``tomlkit.items.String``, etc.), the service must convert
    them to plain Python types via ``toml_to_plain()`` before passing to
    ``model_validate``. Without this conversion, ``toml.dumps()`` in
    ``dump_toml()`` serializes strings as character lists (e.g.
    ``artist = "abc"`` becomes ``artist = ["a", "b", "c"]``).

    These tests exercise ``DatasetService.update_extras()`` end-to-end to
    ensure the history file always contains correctly serialized values.
    """

    @pytest.fixture
    def service(
        self,
        db_connection_factory,
        test_configuration,
        logging_factory,
    ):
        repo = DatasetRepository(db=db_connection_factory, logging=logging_factory)
        loader = DatasetLoader(logging=logging_factory)
        scanner = DatasetScanner(
            db=db_connection_factory,
            repo=repo,
            loader=loader,
            logging=logging_factory,
            configuration=test_configuration,
        )
        watcher = MagicMock(spec=DatasetWatcherService)
        event_dispatcher = MagicMock(spec=EventDispatcher)
        return DatasetService(
            db=db_connection_factory,
            watcher=watcher,
            configuration=test_configuration,
            event_dispatcher=event_dispatcher,
            logging=logging_factory,
            repo=repo,
            scanner=scanner,
            loader=loader,
        )

    def _setup_dataset(self, service: DatasetService, tmp_path: Path) -> tuple[Path, str]:
        """Register a dataset with one image and return (image_path, dataset_name)."""
        from PIL import Image

        img_dir = tmp_path / "images"
        img_dir.mkdir()
        img_path = img_dir / "photo.jpg"
        Image.new("RGB", (1, 1), color="red").save(img_path, format="JPEG")

        config_path = tmp_path / "config.toml"
        config_path.write_text(f'[[dataset]]\npath = "{img_dir}"\n')
        service.register("test_ds", str(config_path), source="import")
        return img_path, "test_ds"

    def _get_image_id(self, service: DatasetService, dataset_name: str, img_path: Path) -> int:
        info = service.get_image_by_path(dataset_name, str(img_path))
        assert info is not None
        return info.id

    def test_history_preserves_string_extras_after_update(self, service, tmp_path):
        """After updating extras, the history must contain the previous string value correctly."""
        img_path, ds_name = self._setup_dataset(service, tmp_path)
        image_id = self._get_image_id(service, ds_name, img_path)

        # Write initial extras (simulating an existing TOML sidecar)
        toml_path = img_path.with_suffix(".toml")
        toml_path.write_text('artist = "Monet"\nstyle = "impressionism"\n')

        # Update extras through the service (webui flow)
        new_extras = 'artist = "Picasso"\nstyle = "cubism"\n'
        service.update_extras(ds_name, image_id, new_extras)

        # Verify the current TOML has the new values
        assert toml_path.read_text() == new_extras

        # Verify history preserved the old values correctly
        history = service.get_history(ds_name, image_id)
        assert history is not None
        assert len(history) == 1
        extras = history[0].extras
        assert extras["artist"] == "Monet"
        assert isinstance(extras["artist"], str)
        assert extras["style"] == "impressionism"

    def test_history_preserves_integer_and_list_extras(self, service, tmp_path):
        """After updating extras, history must correctly preserve int and list values."""
        img_path, ds_name = self._setup_dataset(service, tmp_path)
        image_id = self._get_image_id(service, ds_name, img_path)

        # Write initial extras with mixed types
        toml_path = img_path.with_suffix(".toml")
        toml_path.write_text('year = 1872\ntags = ["painting", "landscape"]\n')

        # Update to new values
        service.update_extras(ds_name, image_id, 'year = 1937\ntags = ["abstract"]\n')

        history = service.get_history(ds_name, image_id)
        assert history is not None
        assert len(history) == 1
        extras = history[0].extras
        assert extras["year"] == 1872
        assert isinstance(extras["year"], int)
        assert extras["tags"] == ["painting", "landscape"]

    def test_multiple_updates_produce_correct_history(self, service, tmp_path):
        """Multiple extras updates should produce correct history entries."""
        img_path, ds_name = self._setup_dataset(service, tmp_path)
        image_id = self._get_image_id(service, ds_name, img_path)

        # First update: create extras
        service.update_extras(ds_name, image_id, 'artist = "A"\n')

        # Second update: change extras
        service.update_extras(ds_name, image_id, 'artist = "B"\n')

        # Third update: change again
        service.update_extras(ds_name, image_id, 'artist = "C"\n')

        history = service.get_history(ds_name, image_id)
        assert history is not None
        assert len(history) == 3  # 3 history entries from the 3 updates
        # Most recent first
        assert history[0].extras.get("artist") == "B"
        assert history[1].extras.get("artist") == "A"
        assert history[2].extras.get("artist") is None  # first update had no prior extras

    def test_update_caption_preserves_existing_extras_in_history(self, service, tmp_path):
        """Updating a caption should save the current extras to history correctly."""
        img_path, ds_name = self._setup_dataset(service, tmp_path)
        image_id = self._get_image_id(service, ds_name, img_path)

        # Write initial extras
        toml_path = img_path.with_suffix(".toml")
        toml_path.write_text('artist = "Rembrandt"\n')

        # Also write initial caption
        caption_path = img_path.with_suffix(".txt")
        caption_path.write_text("old caption")

        # Update caption through the service
        service.update_caption(ds_name, image_id, "new caption")

        # History should preserve the extras
        history = service.get_history(ds_name, image_id)
        assert history is not None
        assert len(history) == 1
        assert history[0].extras.get("artist") == "Rembrandt"
        assert isinstance(history[0].extras["artist"], str)
        assert history[0].caption == "old caption"
