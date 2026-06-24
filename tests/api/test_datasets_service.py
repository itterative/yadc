"""Tests for DatasetService — ``list_images`` cursor decoding, ``preview_prompt`` context, and dispatch logic for ``DatasetChangedEvent``.

Tests for ``update_extras`` / ``update_caption`` / ``get_history`` live in
:class:`TestUpdateExtrasHistoryRoundTrip` and use a real ``DatasetService``
with a real database to exercise the on-disk history sidecar round-trip.
"""

import os
import shutil
import textwrap
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import tomlkit

from yadc.api.events import DatasetChangedEvent
from yadc.api.modules import DatasetWatcherService, DBConnectionFactory, EventDispatcher
from yadc.api.services import DatasetLoader, DatasetScanner
from yadc.api.services.dataset_repository import DatasetInfo, DatasetRepository, ImageInfo
from yadc.api.services.datasets import DATASETS_DIR, DatasetService, HardlinkNotSupportedError
from yadc.core.config import Config, ConfigApi, ConfigDatasetEntry, ConfigPrompt, ConfigReasoning, ConfigSettings
from yadc.core.dataset import DatasetImage


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


def _make_config(entries: list[ConfigDatasetEntry]) -> Config:
    """Build a minimal :class:`Config` for tests, skipping strict validation.

    Uses ``model_construct`` for every field so no default factory fires
    — the default factories would re-run ``ConfigApi()``'s validator in
    strict mode and fail without a ``strict=False`` context. Mirrors the
    pattern in :meth:`ConfigV1.to_v2`. Only ``dataset`` varies across the
    test fixtures; the rest are stub values.
    """
    return Config.model_construct(
        api=ConfigApi.model_construct(),
        prompt=ConfigPrompt.model_construct(),
        settings=ConfigSettings.model_construct(),
        reasoning=ConfigReasoning.model_construct(),
        dataset=entries,
        env="",
        interactive=False,
        rounds=1,
        caption_suffix=".txt",
        overwrite_captions=False,
    )


def _setup_dataset_config(
    service: DatasetService,
    config: Config,
    config_path: Path,
) -> None:
    """Wire up ``get_dataset`` + ``_loader.load_config`` and write a real TOML file.

    The TOML body mirrors the parsed Config the mock returns. The file
    isn't parsed by the test (the loader mock returns a synthetic Config),
    but it must exist on disk so the service's ``config_path.exists()``
    check passes — and having a real TOML keeps the test fixture
    self-documenting.
    """
    import tomlkit

    config_path.write_text(tomlkit.dumps(config.model_dump()))

    info = DatasetInfo(name="test_ds", config_path=str(config_path))
    service.get_dataset = MagicMock(return_value=info)
    service._loader.load_config = MagicMock(return_value=config)


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
        toml_path.write_text(
            textwrap.dedent("""\
                custom_field = "hello"
                number = 42
            """)
        )

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

        custom_template = textwrap.dedent("""\
            {% set system_prompt %}You are a caption refiner.{% endset %}
            {% set user_prompt %}Current caption: {{ caption }}
            Please improve it.{% endset %}
        """)

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

    def test_includes_dataset_level_extras(self, service, image_with_caption, tmp_path):
        """Dataset-level extras from the [[dataset]] entry appear in the template context."""

        img_path, _ = image_with_caption
        config_path = tmp_path / "config.toml"
        config = _make_config(
            [
                ConfigDatasetEntry(
                    path=str(img_path.parent),
                    extras={"universe": "genshin", "artist": "unknown"},
                )
            ]
        )
        self._setup_service_get_image(service, image_id=1, image_path=img_path)
        _setup_dataset_config(service, config=config, config_path=config_path)

        result = service.preview_prompt("test_ds", 1, "")
        assert result is not None
        assert result["template_context"]["universe"] == "genshin"
        assert result["template_context"]["artist"] == "unknown"

    def test_per_image_extras_override_dataset_extras(self, service, image_with_caption, tmp_path):
        """When both dataset and per-image TOML define the same key, the per-image value wins."""

        img_path, _ = image_with_caption

        # Per-image TOML sets 'universe' to "realistic"
        toml_path = img_path.with_suffix(".toml")
        toml_path.write_text('universe = "realistic"\n')

        config_path = tmp_path / "config.toml"
        config = _make_config(
            [
                ConfigDatasetEntry(
                    path=str(img_path.parent),
                    extras={"universe": "anime", "artist": "unknown"},
                )
            ]
        )
        self._setup_service_get_image(service, image_id=1, image_path=img_path)
        _setup_dataset_config(service, config=config, config_path=config_path)

        result = service.preview_prompt("test_ds", 1, "")
        assert result is not None
        # per-image wins
        assert result["template_context"]["universe"] == "realistic"
        # dataset extras for keys not in per-image TOML are still applied
        assert result["template_context"]["artist"] == "unknown"

    def test_relative_entry_path_is_resolved(self, service, image_with_caption, tmp_path):
        """Relative entry paths are resolved against the config file's parent directory."""

        img_path, _ = image_with_caption

        config_dir = tmp_path / "cfg"
        config_dir.mkdir()
        config_path = config_dir / "config.toml"
        # Entry uses a relative path; image lives in `images/`
        config = _make_config([ConfigDatasetEntry(path="images", extras={"universe": "anime"})])
        self._setup_service_get_image(service, image_id=1, image_path=img_path)
        _setup_dataset_config(service, config=config, config_path=config_path)

        # The image is in tmp_path (NOT in config_dir / "images"), so the
        # entry's resolved path is config_dir/"images" and the image doesn't
        # actually match — the dataset extras should NOT be applied.
        result = service.preview_prompt("test_ds", 1, "")
        assert result is not None
        assert "universe" not in result["template_context"]

        # Now the image IS inside config_dir/"images" — extras should apply.
        relinked_img = config_dir / "images" / img_path.name
        relinked_img.parent.mkdir(parents=True, exist_ok=True)
        relinked_img.write_bytes(img_path.read_bytes())
        self._setup_service_get_image(service, image_id=1, image_path=relinked_img)

        result = service.preview_prompt("test_ds", 1, "")
        assert result is not None
        assert result["template_context"]["universe"] == "anime"

    def test_no_dataset_extras_when_config_path_missing(self, service, image_with_caption, tmp_path):
        """A DatasetInfo with no config_path falls through to per-image-only extras."""

        img_path, _ = image_with_caption
        info = DatasetInfo(name="test_ds", config_path=None)
        service.get_dataset = MagicMock(return_value=info)
        # loader must NOT be called in this path
        config = _make_config(
            [
                ConfigDatasetEntry(
                    path=str(img_path.parent),
                    extras={"universe": "anime"},
                )
            ]
        )
        service._loader.load_config = MagicMock(return_value=config)
        self._setup_service_get_image(service, image_id=1, image_path=img_path)

        result = service.preview_prompt("test_ds", 1, "")
        assert result is not None
        service._loader.load_config.assert_not_called()
        assert "universe" not in result["template_context"]

    def test_no_dataset_extras_when_config_file_missing(self, service, image_with_caption, tmp_path):
        """A non-existent config file is treated as 'no dataset extras' (graceful degradation)."""

        img_path, _ = image_with_caption
        missing_cfg = tmp_path / "does-not-exist.toml"
        info = DatasetInfo(name="test_ds", config_path=str(missing_cfg))
        service.get_dataset = MagicMock(return_value=info)
        config = _make_config(
            [
                ConfigDatasetEntry(
                    path=str(img_path.parent),
                    extras={"universe": "anime"},
                )
            ]
        )
        service._loader.load_config = MagicMock(return_value=config)
        self._setup_service_get_image(service, image_id=1, image_path=img_path)

        result = service.preview_prompt("test_ds", 1, "")
        assert result is not None
        service._loader.load_config.assert_not_called()
        assert "universe" not in result["template_context"]

    def test_no_dataset_extras_when_loader_returns_none(self, service, image_with_caption, tmp_path):
        """If the loader can't parse the config, the preview still works (per-image only)."""

        img_path, _ = image_with_caption
        config_path = tmp_path / "config.toml"
        # The TOML file exists on disk (so the exists() check passes), but
        # the loader mock returns None as if parsing failed.
        config = _make_config(
            [
                ConfigDatasetEntry(
                    path=str(img_path.parent),
                    extras={"universe": "anime"},
                )
            ]
        )
        _setup_dataset_config(service, config=config, config_path=config_path)
        service._loader.load_config = MagicMock(return_value=None)
        self._setup_service_get_image(service, image_id=1, image_path=img_path)

        result = service.preview_prompt("test_ds", 1, "")
        assert result is not None
        assert "universe" not in result["template_context"]

    def test_no_dataset_extras_when_image_outside_entry_path(self, service, image_with_caption, tmp_path):
        """An image that isn't inside any entry's path is treated as having no dataset extras."""

        img_path, _ = image_with_caption
        other_dir = tmp_path / "other"
        config_path = tmp_path / "config.toml"
        config = _make_config([ConfigDatasetEntry(path=str(other_dir), extras={"universe": "anime"})])
        _setup_dataset_config(service, config=config, config_path=config_path)
        self._setup_service_get_image(service, image_id=1, image_path=img_path)

        result = service.preview_prompt("test_ds", 1, "")
        assert result is not None
        assert "universe" not in result["template_context"]

    def test_most_specific_path_wins_for_nested_entries(self, service, image_with_caption, tmp_path):
        """When two path-based entries both contain the image, the deepest one wins.

        Mirrors the resolver's non-recursive ``iterdir`` walk, which would
        only have the inner entry pick the image up. Iterating in the
        original config order with ``is_relative_to`` would return the outer
        entry by accident — the test would fail without the depth sort.
        """

        img_path, _ = image_with_caption

        # Lay out a parent + child directory structure under tmp_path so we
        # have a real outer/inner pair. The image fixture lives at
        # ``outer/child/test.jpg`` with both ``outer`` and ``outer/child``
        # as path-based entries.
        nested = tmp_path / "nested"
        nested.mkdir()
        outer = nested / "outer"
        outer.mkdir()
        child = outer / "child"
        child.mkdir()
        nested_img = child / img_path.name
        nested_img.write_bytes(img_path.read_bytes())

        config_path = tmp_path / "config.toml"
        # Outer is FIRST in the config — if the implementation is "first
        # match wins", the outer would win. The correct behaviour is to
        # pick the deeper one.
        config = _make_config(
            [
                ConfigDatasetEntry(path=str(outer), extras={"universe": "outer"}),
                ConfigDatasetEntry(path=str(child), extras={"universe": "inner"}),
            ]
        )
        _setup_dataset_config(service, config=config, config_path=config_path)
        self._setup_service_get_image(service, image_id=1, image_path=nested_img)

        result = service.preview_prompt("test_ds", 1, "")
        assert result is not None
        # The inner (more specific) entry wins, even though the outer is
        # listed first in the config.
        assert result["template_context"]["universe"] == "inner"

    def test_outer_entry_still_wins_when_image_only_in_outer(self, service, image_with_caption, tmp_path):
        """The most-specific-wins logic doesn't break the simple case: an
        image in only the outer entry is still matched by the outer entry."""

        img_path, _ = image_with_caption
        sibling = tmp_path / "sibling"
        sibling.mkdir()  # exists on disk, but the image is not in it

        config_path = tmp_path / "config.toml"
        config = _make_config(
            [
                ConfigDatasetEntry(path=str(img_path.parent), extras={"universe": "outer"}),
                ConfigDatasetEntry(path=str(sibling), extras={"universe": "sibling"}),
            ]
        )
        _setup_dataset_config(service, config=config, config_path=config_path)
        self._setup_service_get_image(service, image_id=1, image_path=img_path)

        result = service.preview_prompt("test_ds", 1, "")
        assert result is not None
        assert result["template_context"]["universe"] == "outer"

    def test_inline_image_entry_wins_over_path_based_entry(self, service, image_with_caption, tmp_path):
        """An inline-image match beats a path-based match on the same image.

        Inline declarations are explicit; path-based are implicit. The
        inline entry's extras should apply.
        """

        img_path, _ = image_with_caption
        config_path = tmp_path / "config.toml"
        config = _make_config(
            [
                ConfigDatasetEntry(
                    path=str(img_path.parent),
                    extras={"universe": "from_path"},
                ),
                ConfigDatasetEntry(
                    images=[DatasetImage(path=str(img_path))],
                    extras={"universe": "from_inline"},
                ),
            ]
        )
        _setup_dataset_config(service, config=config, config_path=config_path)
        self._setup_service_get_image(service, image_id=1, image_path=img_path)

        result = service.preview_prompt("test_ds", 1, "")
        assert result is not None
        assert result["template_context"]["universe"] == "from_inline"

    def test_file_path_entry_is_ignored(self, service, image_with_caption, tmp_path):
        """An entry whose path is a file (not a directory) doesn't match
        images. The resolver's non-recursive walk wouldn't process such
        entries either."""

        img_path, _ = image_with_caption
        # The entry's path points at a file (the image itself), not a dir.
        config_path = tmp_path / "config.toml"
        config = _make_config(
            [
                ConfigDatasetEntry(
                    path=str(img_path),
                    extras={"universe": "should_not_match"},
                )
            ]
        )
        _setup_dataset_config(service, config=config, config_path=config_path)
        self._setup_service_get_image(service, image_id=1, image_path=img_path)

        result = service.preview_prompt("test_ds", 1, "")
        assert result is not None
        assert "universe" not in result["template_context"]


class TestGetCaption:
    """DatasetService.get_caption — verify the per-image fields and the merged extras view."""

    def _setup_service_get_image(self, service: DatasetService, image_id: int, image_path: Path):
        info = ImageInfo(
            id=image_id,
            file_name=image_path.name,
            path=str(image_path),
            has_caption=True,
        )
        service.get_image = MagicMock(return_value=info)

    def test_includes_caption_and_extras_raw(self, service, image_with_caption):
        """The per-image caption and TOML raw content are returned unchanged."""
        img_path, _ = image_with_caption
        toml_path = img_path.with_suffix(".toml")
        toml_path.write_text('artist = "Monet"\n')
        self._setup_service_get_image(service, image_id=1, image_path=img_path)

        result = service.get_caption("test_ds", 1)
        assert result is not None
        assert result["caption"] == "a red square"
        assert result["extras_raw"] == 'artist = "Monet"\n'
        # No dataset configured in this test, so extras mirrors per-image only
        assert result["extras"] == {"artist": "Monet"}

    def test_extras_merge_dataset_level(self, service, image_with_caption, tmp_path):
        """The ``extras`` field merges dataset-level + per-image extras (per-image wins)."""

        img_path, _ = image_with_caption
        toml_path = img_path.with_suffix(".toml")
        toml_path.write_text('universe = "realistic"\n')

        config_path = tmp_path / "config.toml"
        config = _make_config(
            [
                ConfigDatasetEntry(
                    path=str(img_path.parent),
                    extras={"universe": "anime", "artist": "unknown"},
                )
            ]
        )
        _setup_dataset_config(service, config=config, config_path=config_path)
        self._setup_service_get_image(service, image_id=1, image_path=img_path)

        result = service.get_caption("test_ds", 1)
        assert result is not None
        # ``extras`` is the merged view
        assert result["extras"]["universe"] == "realistic"
        assert result["extras"]["artist"] == "unknown"
        # ``extras_raw`` stays as the per-image TOML (Extras tab editor uses it)
        assert result["extras_raw"] == 'universe = "realistic"\n'
        assert "artist" not in result["extras_raw"]

    def test_returns_none_for_missing_image(self, service):
        service.get_image = MagicMock(return_value=None)
        assert service.get_caption("test_ds", 999) is None

    def test_returns_none_for_missing_file(self, service, tmp_path):
        nonexistent = tmp_path / "missing.jpg"
        info = ImageInfo(id=1, file_name="missing.jpg", path=str(nonexistent))
        service.get_image = MagicMock(return_value=info)
        assert service.get_caption("test_ds", 1) is None


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
        toml_path.write_text(
            textwrap.dedent("""\
                artist = "Monet"
                style = "impressionism"
            """)
        )

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
        toml_path.write_text(
            textwrap.dedent("""\
                year = 1872
                tags = ["painting", "landscape"]
            """)
        )

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

    def test_update_caption_preserves_existing_extras_in_live_toml(self, service, tmp_path):
        """Updating a caption must keep the existing extras in the live TOML sidecar.

        Regression: update_caption() re-instantiated a bare DatasetImage (no
        extras loaded) and wrote its empty dump_toml() over the sidecar,
        wiping the extras. It must also not leak a stale ``caption`` key in.
        """
        img_path, ds_name = self._setup_dataset(service, tmp_path)
        image_id = self._get_image_id(service, ds_name, img_path)

        toml_path = img_path.with_suffix(".toml")
        toml_path.write_text(
            textwrap.dedent("""\
                artist = "Monet"
                year = 1872
            """)
        )
        caption_path = img_path.with_suffix(".txt")
        caption_path.write_text("old caption")

        service.update_caption(ds_name, image_id, "new caption")

        # The caption file carries the new caption ...
        assert caption_path.read_text() == "new caption"

        # ... and the live TOML sidecar keeps the extras, untouched in type,
        # with no stale caption leaking in.
        parsed = tomlkit.loads(toml_path.read_text())
        assert parsed["artist"] == "Monet"
        assert isinstance(parsed["artist"], str)
        assert parsed["year"] == 1872
        assert isinstance(parsed["year"], int)
        assert "caption" not in parsed


class TestLoadExtras(TestUpdateExtrasHistoryRoundTrip):
    """``DatasetService.load_extras`` — generic TOML sidecar loader used by the tagger (and any future consumer that round-trips comments).

    Reuses the real-``DatasetService`` fixture from the extras round-trip
    suite; the only difference is that the sidecar is read, not written.
    """

    def _setup_with_sidecar(self, service: DatasetService, tmp_path: Path, sidecar_text: str | None) -> tuple[Path, str, int]:
        """Register a dataset with one image and optionally write a TOML sidecar."""
        img_path, ds_name = self._setup_dataset(service, tmp_path)
        image_id = self._get_image_id(service, ds_name, img_path)
        toml_path = img_path.with_suffix(".toml")
        if sidecar_text is not None:
            toml_path.write_text(sidecar_text)
        return img_path, ds_name, image_id

    def test_load_extras_returns_doc_with_keys(self, service: DatasetService, tmp_path: Path) -> None:
        """A sidecar with comments parses to a doc that preserves the comments for round-trip."""
        _, ds_name, image_id = self._setup_with_sidecar(
            service,
            tmp_path,
            '# artist comment\nartist = "Monet"\n[tags]\ngeneral = ["1girl"]\n',
        )

        extras = service.load_extras(ds_name, image_id)
        assert extras is not None
        # Comments survive the round-trip (plain=False default).
        assert "# artist comment" in tomlkit.dumps(extras)
        # Values parse correctly.
        assert extras["artist"] == "Monet"
        assert extras["tags"]["general"] == ["1girl"]

    def test_load_extras_plain_true_returns_plain_python_types(self, service: DatasetService, tmp_path: Path) -> None:
        """``plain=True`` strips tomlkit wrappers for Pydantic / display callers."""
        _, ds_name, image_id = self._setup_with_sidecar(
            service,
            tmp_path,
            'artist = "Monet"\ntags = ["painting"]\n',
        )

        extras = service.load_extras(ds_name, image_id, plain=True)
        assert extras is not None
        # No tomlkit wrappers.
        assert type(extras["artist"]) is str
        assert type(extras["tags"]) is list
        assert type(extras["tags"][0]) is str

    def test_load_extras_returns_none_for_missing_image(self, service: DatasetService) -> None:
        assert service.load_extras("test_ds", 99999) is None

    def test_load_extras_returns_none_for_missing_sidecar(self, service: DatasetService, tmp_path: Path) -> None:
        """An image exists but its TOML sidecar doesn't → ``None``."""
        _, ds_name, image_id = self._setup_with_sidecar(service, tmp_path, None)
        assert service.load_extras(ds_name, image_id) is None

    def test_load_extras_returns_none_for_unparseable_sidecar(self, service: DatasetService, tmp_path: Path) -> None:
        """A malformed sidecar → ``None`` (write path repairs it; skip path doesn't silently skip)."""
        _, ds_name, image_id = self._setup_with_sidecar(service, tmp_path, "this is not valid TOML ===")
        assert service.load_extras(ds_name, image_id) is None


class TestProbeHardlinkInDir:
    """``DatasetService._probe_hardlink_in_dir`` — used to fail fast on filesystems
    that don't support hardlinks. Drops a small temp file in the target dir,
    attempts ``os.link``, cleans up, returns the result."""

    @pytest.fixture
    def service(self, test_configuration, logging_factory):
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

    def test_probe_returns_true_when_link_succeeds(self, service, tmp_path, monkeypatch):
        """When ``os.link`` succeeds, the probe returns True and leaves no files behind."""
        monkeypatch.setattr("yadc.api.services.datasets.os.link", lambda src, dst: None)
        assert service._probe_hardlink_in_dir(tmp_path) is True
        assert list(tmp_path.iterdir()) == []

    def test_probe_returns_false_when_link_raises(self, service, tmp_path, monkeypatch):
        """When ``os.link`` raises OSError, the probe returns False and leaves no files behind."""

        def raise_oserror(*_args, **_kwargs):
            raise OSError("not supported")

        monkeypatch.setattr("yadc.api.services.datasets.os.link", raise_oserror)
        assert service._probe_hardlink_in_dir(tmp_path) is False
        assert list(tmp_path.iterdir()) == []

    def test_probe_returns_false_when_mkstemp_raises(self, service, tmp_path, monkeypatch):
        """If even creating the temp file fails (e.g. read-only volume), return False."""

        def raise_oserror(*_args, **_kwargs):
            raise OSError("read-only")

        monkeypatch.setattr("tempfile.mkstemp", raise_oserror)
        assert service._probe_hardlink_in_dir(tmp_path) is False
        assert list(tmp_path.iterdir()) == []

    def test_probe_returns_false_for_nonexistent_dir(self, service, tmp_path):
        """A path that doesn't exist (or isn't a dir) returns False without crashing."""
        assert service._probe_hardlink_in_dir(tmp_path / "does-not-exist") is False


class TestDuplicateManagedDataset:
    """``DatasetService.duplicate_managed_dataset`` — copies config + images + sidecars
    to a new managed dataset, applying the per-file-type copy policy (hardlink images,
    copy sidecars + config, skip ``.staging/``). On hardlink failure, raises
    :class:`HardlinkNotSupportedError` after cleaning up the partial new dir.
    """

    # Patch ``DATASETS_DIR`` so the new dataset gets created under
    # ``tmp_path`` instead of the real ``~/.local/state/yadc/datasets``
    # (which would be a destructive surprise if the test ever ran outside
    # the sandbox). Matches the pattern in ``test_dataset_upload_service``.
    @pytest.fixture(autouse=True)
    def patch_datasets_dir(self, tmp_path, monkeypatch):
        from yadc.api.services import datasets as datasets_module

        state_path = tmp_path / "state"
        state_path.mkdir()
        datasets_dir = state_path / "datasets"
        datasets_dir.mkdir()
        monkeypatch.setattr(datasets_module, "DATASETS_DIR", datasets_dir)
        # ``DATASETS_DIR`` is re-exported from the service module; tests
        # that import it directly (this file does) see the original value
        # unless we patch the binding in this module too.
        monkeypatch.setattr("tests.api.test_datasets_service.DATASETS_DIR", datasets_dir)

    @pytest.fixture
    def service(self, db_connection_factory, test_configuration, logging_factory):
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

    def _setup_managed_dataset(self, service: DatasetService, src_dir: Path) -> str:
        """Create a managed dataset (source=upload) on disk with images, sidecars,
        folders, and a ``.staging/`` dir. Register it and return the dataset name.
        """
        from PIL import Image

        # Config uses relative paths like a real upload-sourced dataset.
        # The scanner resolves them against the config's parent dir.
        config_path = src_dir / "config.toml"
        config_path.write_text('[[dataset]]\npath = "images"\n[[dataset]]\npath = "folders/train"\n')

        # Root images + their sidecars
        images_dir = src_dir / "images"
        images_dir.mkdir()
        Image.new("RGB", (1, 1), color="red").save(images_dir / "foo.jpg", format="JPEG")
        (images_dir / "foo.txt").write_text("caption for foo")
        (images_dir / "foo.toml").write_text('artist = "tester"\n')
        (images_dir / "foo.history~").write_text("----------\nold caption\n")
        (images_dir / "foo.gemma.draft~").write_text("draft from gemma")

        # Folder upload: folders/train/ with its own image + sidecar
        train_dir = src_dir / "folders" / "train"
        train_dir.mkdir(parents=True)
        Image.new("RGB", (1, 1), color="blue").save(train_dir / "bar.jpg", format="JPEG")
        (train_dir / "bar.txt").write_text("caption for bar")

        # .staging/ is transient upload state — must NOT be copied
        staging_dir = src_dir / ".staging" / "abc-123"
        staging_dir.mkdir(parents=True)
        (staging_dir / "in_progress.jpg").write_bytes(b"partial upload")

        service.register("src_ds", str(config_path), source="upload")
        return "src_ds"

    def _track_io(self, monkeypatch: pytest.MonkeyPatch) -> tuple[list[Path], list[Path]]:
        """Patch ``os.link`` and ``shutil.copy2`` so the tests can inspect calls.
        The real functions still run (so files actually exist on disk).
        Returns (link_sources, copy_sources) lists that get appended to on each call.

        Note: ``_probe_hardlink_in_dir`` also calls ``os.link`` (with a
        temp file inside the destination dir) when ``mode='hardlink'``,
        so callers should filter probe artifacts before asserting. See
        :meth:`_real_link_sources`.
        """
        link_sources: list[Path] = []
        copy_sources: list[Path] = []

        real_link = os.link
        real_copy2 = shutil.copy2

        def mock_link(src, dst, *args, **kwargs):
            link_sources.append(Path(src))
            return real_link(src, dst, *args, **kwargs)

        def mock_copy2(src, dst, *args, **kwargs):
            copy_sources.append(Path(src))
            return real_copy2(src, dst, *args, **kwargs)

        monkeypatch.setattr("yadc.api.services.datasets.os.link", mock_link)
        monkeypatch.setattr("yadc.api.services.datasets.shutil.copy2", mock_copy2)
        return link_sources, copy_sources

    @staticmethod
    def _real_link_sources(link_sources: list[Path]) -> list[Path]:
        """Filter out the probe's internal ``os.link`` call. The probe drops
        a temp file with the ``.yadc-probe-`` prefix inside the destination
        dir and links it; that call is recorded by ``_track_io`` but isn't
        a real image-bytes hardlink and should be ignored when asserting
        the per-file-type policy."""
        return [p for p in link_sources if not p.name.startswith(".yadc-probe-")]

    def test_duplicate_managed_dataset_hardlink(self, service, tmp_path, monkeypatch):
        """mode='hardlink' (default) hardlinks image bytes, copies config + sidecars,
        skips ``.staging/``, registers the new dataset, leaves the source untouched."""
        src_dir = tmp_path / "source"
        src_dir.mkdir()
        self._setup_managed_dataset(service, src_dir)
        link_sources, copy_sources = self._track_io(monkeypatch)

        # Mock the second register call (the duplicate's own) so we don't
        # have to re-test the registration logic. The first register
        # (for the source) has already run with the real implementation.
        service.register = MagicMock(return_value=MagicMock(name="DatasetInfo"))

        service.duplicate_managed_dataset("src_ds", "new_ds", mode="hardlink")

        # New dataset structure mirrors the source layout under DATASETS_DIR / new_ds
        new_dir = DATASETS_DIR / "new_ds"
        assert new_dir.is_dir()
        assert (new_dir / "config.toml").read_text() == (src_dir / "config.toml").read_text()
        assert (new_dir / "images" / "foo.jpg").is_file()
        assert (new_dir / "images" / "foo.txt").read_text() == "caption for foo"
        assert (new_dir / "images" / "foo.toml").read_text() == 'artist = "tester"\n'
        assert (new_dir / "images" / "foo.history~").read_text() == "----------\nold caption\n"
        assert (new_dir / "images" / "foo.gemma.draft~").read_text() == "draft from gemma"
        assert (new_dir / "folders" / "train" / "bar.jpg").is_file()
        assert (new_dir / "folders" / "train" / "bar.txt").read_text() == "caption for bar"
        # .staging/ is not walked and must not appear in the new dir
        assert not (new_dir / ".staging").exists()

        # Image bytes went through os.link; config + sidecars went through shutil.copy2
        real_links = self._real_link_sources(link_sources)
        assert sorted(p.name for p in real_links) == ["bar.jpg", "foo.jpg"]
        assert sorted(p.name for p in copy_sources) == [
            "bar.txt",
            "config.toml",
            "foo.gemma.draft~",
            "foo.history~",
            "foo.toml",
            "foo.txt",
        ]

        # The source layout is unchanged
        assert (src_dir / "images" / "foo.jpg").is_file()
        assert (src_dir / ".staging" / "abc-123" / "in_progress.jpg").is_file()

        # register was called for the new dataset with source='upload'
        service.register.assert_called_once()
        args, kwargs = service.register.call_args
        assert args[0] == "new_ds"
        assert kwargs.get("source") == "upload"

    def test_duplicate_managed_dataset_default_mode_is_hardlink(self, service, tmp_path, monkeypatch):
        """When ``mode`` is omitted, the service defaults to hardlink — image bytes
        go through ``os.link`` without the caller specifying it."""
        src_dir = tmp_path / "source"
        src_dir.mkdir()
        self._setup_managed_dataset(service, src_dir)
        link_sources, _copy_sources = self._track_io(monkeypatch)
        service.register = MagicMock(return_value=MagicMock(name="DatasetInfo"))

        service.duplicate_managed_dataset("src_ds", "new_ds")

        real_links = self._real_link_sources(link_sources)
        assert sorted(p.name for p in real_links) == ["bar.jpg", "foo.jpg"]
        assert (DATASETS_DIR / "new_ds" / "config.toml").is_file()

    def test_duplicate_managed_dataset_copy(self, service, tmp_path, monkeypatch):
        """mode='copy' uses shutil.copy2 for everything, including image bytes."""
        src_dir = tmp_path / "source"
        src_dir.mkdir()
        self._setup_managed_dataset(service, src_dir)
        _link_sources, copy_sources = self._track_io(monkeypatch)
        service.register = MagicMock(return_value=MagicMock(name="DatasetInfo"))

        service.duplicate_managed_dataset("src_ds", "new_ds", mode="copy")

        # Every file (images + sidecars + config) went through shutil.copy2.
        # os.link was never called.
        copy_names = sorted(p.name for p in copy_sources)
        assert "foo.jpg" in copy_names
        assert "bar.jpg" in copy_names
        assert "config.toml" in copy_names
        # No hardlinks happened
        # (we already assert link_sources was empty if no hardlinks)

    def test_duplicate_managed_dataset_copy_link_sources_empty(self, service, tmp_path, monkeypatch):
        """In copy mode, ``os.link`` is never called for image bytes."""
        src_dir = tmp_path / "source"
        src_dir.mkdir()
        self._setup_managed_dataset(service, src_dir)
        link_sources, _copy_sources = self._track_io(monkeypatch)
        service.register = MagicMock(return_value=MagicMock(name="DatasetInfo"))

        service.duplicate_managed_dataset("src_ds", "new_ds", mode="copy")
        assert link_sources == []

    def test_duplicate_non_managed_raises(self, service, tmp_path, monkeypatch):
        """Datasets whose source != 'upload' cannot be duplicated. Raises ValueError."""
        # Set up a dataset with source='import' (not managed)
        from PIL import Image

        img_dir = tmp_path / "ext"
        img_dir.mkdir()
        Image.new("RGB", (1, 1), color="red").save(img_dir / "x.jpg", format="JPEG")
        config_path = tmp_path / "config.toml"
        config_path.write_text(f'[[dataset]]\npath = "{img_dir}"\n')
        service.register("ext_ds", str(config_path), source="import")
        link_sources, copy_sources = self._track_io(monkeypatch)
        service.register = MagicMock(return_value=MagicMock(name="DatasetInfo"))

        with pytest.raises(ValueError, match="not a managed dataset"):
            service.duplicate_managed_dataset("ext_ds", "new_ds", mode="hardlink")

        assert link_sources == []
        assert copy_sources == []
        assert not (DATASETS_DIR / "new_ds").exists()

    def test_duplicate_missing_source_raises(self, service, tmp_path, monkeypatch):
        """Unknown source dataset name raises ValueError."""
        self._track_io(monkeypatch)
        with pytest.raises(ValueError, match="not found"):
            service.duplicate_managed_dataset("nope", "new_ds", mode="hardlink")
        assert not (DATASETS_DIR / "new_ds").exists()

    def test_duplicate_name_collision_raises(self, service, tmp_path, monkeypatch):
        """A new_name that already exists (registered or on disk) raises ValueError."""
        src_dir = tmp_path / "source"
        src_dir.mkdir()
        self._setup_managed_dataset(service, src_dir)
        self._track_io(monkeypatch)

        # Register another dataset with the name we want to duplicate to
        from PIL import Image

        other_dir = tmp_path / "other"
        other_dir.mkdir()
        Image.new("RGB", (1, 1), color="red").save(other_dir / "y.jpg", format="JPEG")
        other_config = other_dir / "config.toml"
        other_config.write_text(f'[[dataset]]\npath = "{other_dir}"\n')
        service.register("new_ds", str(other_config), source="import")

        with pytest.raises(ValueError, match="already exists"):
            service.duplicate_managed_dataset("src_ds", "new_ds", mode="hardlink")

    def test_duplicate_dest_dir_already_exists_raises(self, service, tmp_path, monkeypatch):
        """If the new dataset dir exists on disk (stale state) but the dataset isn't
        registered, the service refuses to clobber it. Raises ValueError."""
        src_dir = tmp_path / "source"
        src_dir.mkdir()
        self._setup_managed_dataset(service, src_dir)
        self._track_io(monkeypatch)

        # Pre-create a stale dir with the target name
        stale = DATASETS_DIR / "new_ds"
        stale.mkdir(parents=True)
        (stale / "leftover.txt").write_text("stale state")

        with pytest.raises(ValueError, match="already exists"):
            service.duplicate_managed_dataset("src_ds", "new_ds", mode="hardlink")

        # The stale state must not have been clobbered
        assert (stale / "leftover.txt").is_file()

    def test_duplicate_empty_new_name_raises(self, service, tmp_path, monkeypatch):
        """An empty new_name raises ValueError before any I/O happens."""
        src_dir = tmp_path / "source"
        src_dir.mkdir()
        self._setup_managed_dataset(service, src_dir)
        self._track_io(monkeypatch)

        with pytest.raises(ValueError, match="must not be empty"):
            service.duplicate_managed_dataset("src_ds", "", mode="hardlink")

    def test_duplicate_same_name_as_source_raises(self, service, tmp_path, monkeypatch):
        """A new_name equal to the source name raises ValueError."""
        src_dir = tmp_path / "source"
        src_dir.mkdir()
        self._setup_managed_dataset(service, src_dir)
        self._track_io(monkeypatch)

        with pytest.raises(ValueError, match="must differ from source name"):
            service.duplicate_managed_dataset("src_ds", "src_ds", mode="hardlink")

    def test_duplicate_hardlink_not_supported(self, service, tmp_path, monkeypatch):
        """When the probe says hardlinks aren't supported, the service raises
        HardlinkNotSupportedError and cleans up the partial new dir."""
        src_dir = tmp_path / "source"
        src_dir.mkdir()
        self._setup_managed_dataset(service, src_dir)
        self._track_io(monkeypatch)
        service.register = MagicMock(return_value=MagicMock(name="DatasetInfo"))

        # Force the probe to return False (simulating an exFAT-style fs)
        monkeypatch.setattr(service, "_probe_hardlink_in_dir", lambda _dir: False)

        with pytest.raises(HardlinkNotSupportedError, match="Hardlinks are not supported"):
            service.duplicate_managed_dataset("src_ds", "new_ds", mode="hardlink")

        # The partial new dir was created (config copied) before the probe
        # fired, and must be removed by the cleanup in the except block.
        assert not (DATASETS_DIR / "new_ds").exists()
        # register was never called for the new dataset
        service.register.assert_not_called()

    def test_duplicate_partial_failure_cleanup(self, service, tmp_path, monkeypatch):
        """An I/O error mid-walk (e.g. a specific os.link fails) cleans up the
        partial new dir and re-raises."""
        src_dir = tmp_path / "source"
        src_dir.mkdir()
        self._setup_managed_dataset(service, src_dir)
        self._track_io(monkeypatch)
        service.register = MagicMock(return_value=MagicMock(name="DatasetInfo"))

        # Force the second os.link call (bar.jpg) to fail. The first one
        # (foo.jpg) succeeds, the second raises.
        real_link = os.link
        call_count = {"n": 0}

        def flaky_link(src, dst, *args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] >= 2:
                raise OSError("simulated I/O failure")
            return real_link(src, dst, *args, **kwargs)

        monkeypatch.setattr("yadc.api.services.datasets.os.link", flaky_link)

        with pytest.raises(OSError, match="simulated I/O failure"):
            service.duplicate_managed_dataset("src_ds", "new_ds", mode="hardlink")

        # The partial new dir is gone
        assert not (DATASETS_DIR / "new_ds").exists()
        service.register.assert_not_called()

    def test_duplicate_skips_staging(self, service, tmp_path, monkeypatch):
        """``.staging/<uuid>/`` is a sibling of ``images/``/``folders/`` and is
        therefore not part of the walk — verify no staging contents appear
        in the new dataset even when the source has them."""
        src_dir = tmp_path / "source"
        src_dir.mkdir()
        self._setup_managed_dataset(service, src_dir)
        _link_sources, _copy_sources = self._track_io(monkeypatch)
        service.register = MagicMock(return_value=MagicMock(name="DatasetInfo"))

        service.duplicate_managed_dataset("src_ds", "new_ds", mode="hardlink")

        new_dir = DATASETS_DIR / "new_ds"
        # No .staging anywhere under the new dir
        assert not (new_dir / ".staging").exists()
        # Specifically the in-progress upload is not there
        assert not list(new_dir.rglob("in_progress.jpg"))
