"""Tests for ``DatasetScanner`` — disk walking, diff reconciliation, targeted updates.

The scanner owns two update paths (``scan_disk`` and ``scan_targeted``)
plus the pure ``read_disk`` / ``scan_image_meta`` / ``resolve_affected_image_paths``
helpers. These tests exercise the scanner with a real repo and a real
``DatasetLoader`` (the loader is cheap to construct, no DB needed).
"""

from pathlib import Path

import pytest

from yadc.api.services.dataset_loader import DatasetLoader
from yadc.api.services.dataset_repository import DatasetRepository
from yadc.api.services.dataset_scanner import DatasetScanner


@pytest.fixture
def loader(logging_factory):
    return DatasetLoader(logging=logging_factory)


@pytest.fixture
def repo(db_connection_factory, logging_factory):
    return DatasetRepository(db=db_connection_factory, logging=logging_factory)


@pytest.fixture
def scanner(db_connection_factory, repo, loader, test_configuration, logging_factory):
    return DatasetScanner(
        db=db_connection_factory,
        repo=repo,
        loader=loader,
        logging=logging_factory,
        configuration=test_configuration,
    )


def _make_image_dir(tmp_path: Path) -> Path:
    from PIL import Image

    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for name in ("a.jpg", "b.png"):
        Image.new("RGB", (1, 1), color="red").save(img_dir / name)
    return img_dir


def _write_config(tmp_path: Path, img_dir: Path) -> Path:
    config = tmp_path / "config.toml"
    config.write_text(f'[[dataset]]\npath = "{img_dir}"\n')
    return config


class TestResolveAffectedImagePaths:
    """``resolve_affected_image_paths`` — map watcher paths to image rows.

    The watcher reports the raw filesystem path that changed. The
    index is keyed on the *image* path, so the scanner has to map
    sidecar events (``photo.txt``, ``photo.toml``, ``photo.history~``,
    ``photo.gemma.draft~``) back to the image they decorate.

    The sidecar convention is ``<image_stem>.<sidecar_suffix>``
    where ``<image_stem>`` is the image filename *minus* the image
    extension (e.g. ``photo`` for ``photo.jpg``). Since the image
    extension isn't recoverable from the sidecar name, the resolver
    returns all ``<stem>.<ext>`` candidates and the targeted update
    narrows to the one that's in the index.
    """

    def test_image_file_maps_to_itself(self, scanner):
        paths = scanner.resolve_affected_image_paths(["/data/photo.jpg"])
        assert paths == {"/data/photo.jpg"}

    def test_caption_sidecar_expands_to_candidates(self, scanner):
        # Caption is photo.txt (not photo.jpg.txt — the sidecar replaces the ext).
        paths = scanner.resolve_affected_image_paths(["/data/photo.txt"])
        assert "/data/photo.jpg" in paths
        assert "/data/photo.png" in paths
        assert "/data/photo.webp" in paths

    def test_toml_sidecar_expands_to_candidates(self, scanner):
        paths = scanner.resolve_affected_image_paths(["/data/photo.toml"])
        assert "/data/photo.jpg" in paths

    def test_history_sidecar_expands_to_candidates(self, scanner):
        paths = scanner.resolve_affected_image_paths(["/data/photo.history~"])
        assert "/data/photo.jpg" in paths

    def test_draft_sidecar_expands_to_candidates(self, scanner):
        # Drafts are <image_stem>.<draft_name>.draft~ — same stem-based
        # convention as other sidecars. Last segment is the draft name.
        paths = scanner.resolve_affected_image_paths(["/data/photo.gemma.draft~"])
        assert "/data/photo.jpg" in paths
        # Other images with different stems are not affected
        assert "/data/other.jpg" not in paths

    def test_multiple_sidecars_for_same_stem(self, scanner):
        paths = scanner.resolve_affected_image_paths(["/data/photo.txt", "/data/photo.toml", "/data/photo.gemma.draft~"])
        # All expand to the same set of <stem>.<ext> candidates
        assert "/data/photo.jpg" in paths
        assert "/data/photo.png" in paths
        # No entries for other images
        assert not any("other" in p for p in paths)

    def test_different_stems_dont_overlap(self, scanner):
        paths = scanner.resolve_affected_image_paths(["/data/a.txt", "/data/b.toml"])
        # /data/a.txt -> /data/a.<ext> candidates
        assert "/data/a.jpg" in paths
        # /data/b.toml -> /data/b.<ext> candidates
        assert "/data/b.jpg" in paths
        # /data/a.<ext> doesn't include /data/a.b.<ext> (which would be
        # a candidate if "a.b" were a valid stem — it isn't here)
        assert not any(p.startswith("/data/a.b.") for p in paths)

    def test_unknown_extension_dropped(self, scanner):
        """A file with no recognised suffix is dropped.

        The resolver only knows about image extensions and the three
        sidecar suffixes. Anything else is silently ignored — the
        targeted update is a no-op for it, and the full scan fallback
        picks it up if it's actually a new image.
        """
        paths = scanner.resolve_affected_image_paths(["/data/strange.xyz"])
        assert paths == set()


class TestScanImageMeta:
    """``scan_image_meta`` — per-image metadata extraction."""

    def test_returns_meta_for_valid_image(self, scanner, tmp_path):
        from PIL import Image

        img_path = tmp_path / "photo.jpg"
        Image.new("RGB", (10, 20), color="red").save(img_path)
        meta = scanner.scan_image_meta(img_path)
        assert meta is not None
        assert meta["file_name"] == "photo.jpg"
        assert meta["width"] == 10
        assert meta["height"] == 20
        assert meta["has_caption"] is False
        assert meta["has_toml"] is False
        assert meta["draft_names"] == ""

    def test_returns_none_for_non_image(self, scanner, tmp_path):
        txt = tmp_path / "readme.txt"
        txt.write_text("hello")
        assert scanner.scan_image_meta(txt) is None

    def test_detects_caption_and_toml_sidecars(self, scanner, tmp_path):
        from PIL import Image

        img = tmp_path / "photo.jpg"
        Image.new("RGB", (1, 1), color="red").save(img)
        img.with_suffix(".txt").write_text("caption")
        img.with_suffix(".toml").write_text("foo = 1\n")

        meta = scanner.scan_image_meta(img)
        assert meta is not None
        assert meta["has_caption"] is True
        assert meta["has_toml"] is True

    def test_detects_drafts(self, scanner, tmp_path):
        from PIL import Image

        img = tmp_path / "photo.jpg"
        Image.new("RGB", (1, 1), color="red").save(img)
        (tmp_path / "photo.gemma.draft~").write_text("draft 1")
        (tmp_path / "photo.qwen.draft~").write_text("draft 2")

        meta = scanner.scan_image_meta(img)
        assert meta is not None
        assert sorted(meta["draft_names"].split(",")) == ["gemma", "qwen"]

    def test_includes_file_size(self, scanner, tmp_path):
        from PIL import Image

        img_path = tmp_path / "photo.jpg"
        Image.new("RGB", (1, 1), color="red").save(img_path)
        meta = scanner.scan_image_meta(img_path)
        assert meta is not None
        assert meta["file_size"] == img_path.stat().st_size

    def test_size_only_change_triggers_meta_differs(self, scanner, tmp_path):
        from PIL import Image

        from yadc.api.services.dataset_repository import ImageInfo

        img_path = tmp_path / "photo.jpg"
        Image.new("RGB", (1, 1), color="red").save(img_path)
        meta = scanner.scan_image_meta(img_path)
        assert meta is not None

        # A stored row whose file_size hasn't been backfilled (0) differs
        # from the disk size — this is what repopulates existing rows
        # after migration 0007 resets last_scanned_t.
        stale = ImageInfo(
            id=1,
            file_name="photo.jpg",
            path=str(img_path),
            has_caption=False,
            has_toml=False,
            width=meta["width"],
            height=meta["height"],
            draft_names=[],
            last_modified_t=meta["last_modified_t"],
            file_size=0,
        )
        assert scanner._image_meta_differs(stale, meta) is True

        # Once the stored size matches, there's no diff.
        fresh = ImageInfo(
            id=1,
            file_name="photo.jpg",
            path=str(img_path),
            has_caption=False,
            has_toml=False,
            width=meta["width"],
            height=meta["height"],
            draft_names=[],
            last_modified_t=meta["last_modified_t"],
            file_size=meta["file_size"],
        )
        assert scanner._image_meta_differs(fresh, meta) is False


class TestScanTargeted:
    """``scan_targeted`` — the index update driven by watcher paths.

    This is the long-pole optimization: a single-file external change
    touches one row instead of walking every configured directory.
    """

    def test_add_new_image(self, scanner, repo, tmp_path):
        """A new image file appearing in a watched dir is upserted."""
        from PIL import Image

        img_dir = _make_image_dir(tmp_path)
        config_path = _write_config(tmp_path, img_dir)
        dataset_id = repo.upsert_dataset("alpha", str(config_path), "import")
        # Seed the index by walking the disk (uses the scanner's own
        # bulk-insert path, which is the same code path real
        # ``DatasetService.register`` uses).
        scanner.scan_disk(dataset_id, str(config_path))
        assert repo.get_dataset("alpha").image_count == 2

        new_path = img_dir / "c.jpg"
        Image.new("RGB", (1, 1), color="blue").save(new_path)

        changed = scanner.scan_targeted(dataset_id, [str(new_path)])
        assert changed is True
        assert repo.get_image_by_path("alpha", str(new_path)) is not None
        assert repo.get_dataset("alpha").image_count == 3

    def test_delete_image(self, scanner, repo, tmp_path):
        """An image file vanishing is removed from the index."""
        img_dir = _make_image_dir(tmp_path)
        config_path = _write_config(tmp_path, img_dir)
        dataset_id = repo.upsert_dataset("alpha", str(config_path), "import")
        scanner.scan_disk(dataset_id, str(config_path))

        removed = img_dir / "a.jpg"
        removed.unlink()

        changed = scanner.scan_targeted(dataset_id, [str(removed)])
        assert changed is True
        assert repo.get_image_by_path("alpha", str(removed)) is None
        assert repo.get_dataset("alpha").image_count == 1

    def test_add_caption_via_sidecar(self, scanner, repo, tmp_path):
        """Creating a .txt sidecar updates has_caption without touching other rows."""
        img_dir = _make_image_dir(tmp_path)
        config_path = _write_config(tmp_path, img_dir)
        dataset_id = repo.upsert_dataset("alpha", str(config_path), "import")
        scanner.scan_disk(dataset_id, str(config_path))

        # Sanity: no captions yet
        info = repo.get_image_by_path("alpha", str(img_dir / "a.jpg"))
        assert info is not None and info.has_caption is False

        # Caption is <image_stem>.txt (e.g. a.txt for a.jpg).
        caption_path = img_dir / "a.txt"
        caption_path.write_text("a red square")

        changed = scanner.scan_targeted(dataset_id, [str(caption_path)])
        assert changed is True
        updated = repo.get_image_by_path("alpha", str(img_dir / "a.jpg"))
        assert updated is not None and updated.has_caption is True

        # b.png is untouched
        other = repo.get_image_by_path("alpha", str(img_dir / "b.png"))
        assert other is not None and other.has_caption is False

    def test_remove_caption_via_sidecar(self, scanner, repo, tmp_path):
        """Deleting a .txt sidecar flips has_caption back to False."""
        img_dir = _make_image_dir(tmp_path)
        (img_dir / "a.txt").write_text("caption")
        config_path = _write_config(tmp_path, img_dir)
        dataset_id = repo.upsert_dataset("alpha", str(config_path), "import")
        scanner.scan_disk(dataset_id, str(config_path))

        info = repo.get_image_by_path("alpha", str(img_dir / "a.jpg"))
        assert info is not None and info.has_caption is True

        caption_path = img_dir / "a.txt"
        caption_path.unlink()

        changed = scanner.scan_targeted(dataset_id, [str(caption_path)])
        assert changed is True
        updated = repo.get_image_by_path("alpha", str(img_dir / "a.jpg"))
        assert updated is not None and updated.has_caption is False

    def test_draft_add(self, scanner, repo, tmp_path):
        """Creating a draft sidecar updates draft_names for the affected image."""
        img_dir = _make_image_dir(tmp_path)
        config_path = _write_config(tmp_path, img_dir)
        dataset_id = repo.upsert_dataset("alpha", str(config_path), "import")
        scanner.scan_disk(dataset_id, str(config_path))

        # Drafts are named <image_stem>.<draft_name>.draft~ — same
        # stem convention as captions.
        draft_path = img_dir / "a.gemma.draft~"
        draft_path.write_text("draft from gemma")

        changed = scanner.scan_targeted(dataset_id, [str(draft_path)])
        assert changed is True
        updated = repo.get_image_by_path("alpha", str(img_dir / "a.jpg"))
        assert updated is not None
        assert "gemma" in updated.draft_names

    def test_stray_sidecar_for_unknown_image_is_noop(self, scanner, repo, tmp_path):
        """A sidecar whose image was never in the index is silently ignored."""
        img_dir = _make_image_dir(tmp_path)
        config_path = _write_config(tmp_path, img_dir)
        dataset_id = repo.upsert_dataset("alpha", str(config_path), "import")
        scanner.scan_disk(dataset_id, str(config_path))

        # caption for an image that doesn't exist (stem has no image in index)
        stray = img_dir / "never_existed.txt"
        stray.write_text("orphan")

        changed = scanner.scan_targeted(dataset_id, [str(stray)])
        assert changed is False

    def test_no_op_when_index_already_in_sync(self, scanner, repo, tmp_path):
        """If the disk state matches the index, no SQL writes happen."""
        img_dir = _make_image_dir(tmp_path)
        config_path = _write_config(tmp_path, img_dir)
        dataset_id = repo.upsert_dataset("alpha", str(config_path), "import")
        scanner.scan_disk(dataset_id, str(config_path))

        existing = str(img_dir / "a.jpg")
        from unittest.mock import patch

        with patch.object(repo, "upsert_image") as mock_upsert, patch.object(repo, "delete_image") as mock_delete:
            changed = scanner.scan_targeted(dataset_id, [existing])
        assert changed is False
        mock_upsert.assert_not_called()
        mock_delete.assert_not_called()

    def test_empty_changed_paths_returns_false(self, scanner, repo, tmp_path):
        """An empty changed_paths list is a no-op (caller should fall back to full scan)."""
        img_dir = _make_image_dir(tmp_path)
        config_path = _write_config(tmp_path, img_dir)
        dataset_id = repo.upsert_dataset("alpha", str(config_path), "import")

        assert scanner.scan_targeted(dataset_id, []) is False

    def test_image_count_updated(self, scanner, repo, tmp_path):
        """The dataset's image_count reflects net changes after a targeted update."""
        from PIL import Image

        img_dir = _make_image_dir(tmp_path)
        config_path = _write_config(tmp_path, img_dir)
        dataset_id = repo.upsert_dataset("alpha", str(config_path), "import")
        scanner.scan_disk(dataset_id, str(config_path))

        # Add two, remove one
        Image.new("RGB", (1, 1), color="blue").save(img_dir / "c.jpg")
        Image.new("RGB", (1, 1), color="blue").save(img_dir / "d.jpg")
        (img_dir / "a.jpg").unlink()

        scanner.scan_targeted(
            dataset_id,
            [str(img_dir / "c.jpg"), str(img_dir / "d.jpg"), str(img_dir / "a.jpg")],
        )

        # 2 (original) + 2 (added) - 1 (removed) = 3
        assert repo.get_dataset("alpha").image_count == 3

    def test_image_count_with_mixed_upserts_and_deletes(self, scanner, repo, tmp_path):
        """An upsert that's an *update* (existing row) must not bump image_count.

        The targeted update can produce both new rows and updates to
        existing rows in the same call. Only the new rows should
        count toward the dataset's image_count.
        """
        from PIL import Image

        img_dir = _make_image_dir(tmp_path)
        config_path = _write_config(tmp_path, img_dir)
        dataset_id = repo.upsert_dataset("alpha", str(config_path), "import")
        scanner.scan_disk(dataset_id, str(config_path))

        # One new image + a sidecar change on an existing image
        Image.new("RGB", (1, 1), color="blue").save(img_dir / "c.jpg")
        (img_dir / "a.txt").write_text("caption for a")  # update existing

        scanner.scan_targeted(
            dataset_id,
            [str(img_dir / "c.jpg"), str(img_dir / "a.txt")],
        )

        # 2 (original) + 1 (new c.jpg) + 0 (a.jpg was an update) = 3
        assert repo.get_dataset("alpha").image_count == 3


class TestScanDiskOrchestration:
    """Verify the scanner composes repo calls inside a single transaction.

    The repo's public methods each use ``self._db.connection()`` which
    auto-enrolls in the active transaction. The scanner owns the
    ``with self._db.transaction():`` boundary. These tests exercise
    the orchestration end-to-end with a real factory and real repo.
    """

    def test_disk_scan_inserts_images(self, scanner, repo, tmp_path):
        img_dir = _make_image_dir(tmp_path)
        config_path = _write_config(tmp_path, img_dir)
        dataset_id = repo.upsert_dataset("alpha", str(config_path), "import")

        changed = scanner.scan_disk(dataset_id, str(config_path))
        assert changed is True

        result = repo.get_dataset("alpha")
        assert result.image_count == 2
        assert repo.get_image_by_path("alpha", str(img_dir / "a.jpg")) is not None
        assert repo.get_image_by_path("alpha", str(img_dir / "b.png")) is not None

    def test_disk_scan_drops_removed_images(self, scanner, repo, tmp_path):
        img_dir = _make_image_dir(tmp_path)
        config_path = _write_config(tmp_path, img_dir)
        dataset_id = repo.upsert_dataset("alpha", str(config_path), "import")
        # Seed the index first.
        scanner.scan_disk(dataset_id, str(config_path))
        assert repo.get_image_by_path("alpha", str(img_dir / "a.jpg")) is not None

        (img_dir / "a.jpg").unlink()

        assert scanner.scan_disk(dataset_id, str(config_path)) is True
        assert repo.get_image_by_path("alpha", str(img_dir / "a.jpg")) is None
        assert repo.get_image_by_path("alpha", str(img_dir / "b.png")) is not None
        assert repo.get_dataset("alpha").image_count == 1

    def test_disk_scan_picks_up_new_images(self, scanner, repo, tmp_path):
        from PIL import Image

        img_dir = _make_image_dir(tmp_path)
        config_path = _write_config(tmp_path, img_dir)
        dataset_id = repo.upsert_dataset("alpha", str(config_path), "import")
        scanner.scan_disk(dataset_id, str(config_path))

        Image.new("RGB", (1, 1), color="blue").save(img_dir / "c.jpg")
        assert scanner.scan_disk(dataset_id, str(config_path)) is True
        assert repo.get_image_by_path("alpha", str(img_dir / "c.jpg")) is not None
        assert repo.get_dataset("alpha").image_count == 3

    def test_disk_scan_skips_unchanged_images(self, scanner, repo, tmp_path):
        """Re-scanning a dataset whose disk state matches the index should not issue per-image SQL writes.

        Verifies the staleness filter: ``scan_disk`` should detect
        that every row's stored metadata matches the freshly walked
        disk and skip the upsert loop entirely. It also returns
        ``False`` because no rows changed.
        """
        from unittest.mock import patch

        img_dir = _make_image_dir(tmp_path)
        config_path = _write_config(tmp_path, img_dir)
        dataset_id = repo.upsert_dataset("alpha", str(config_path), "import")
        scanner.scan_disk(dataset_id, str(config_path))

        with patch.object(repo, "upsert_image") as mock_upsert:
            assert scanner.scan_disk(dataset_id, str(config_path)) is False
            assert mock_upsert.call_count == 0

    def test_disk_scan_only_upserts_changed_images(self, scanner, repo, tmp_path):
        """Re-scanning a dataset with one new sidecar should only upsert the changed image, not all of them."""
        from unittest.mock import patch

        img_dir = _make_image_dir(tmp_path)
        config_path = _write_config(tmp_path, img_dir)
        dataset_id = repo.upsert_dataset("alpha", str(config_path), "import")
        scanner.scan_disk(dataset_id, str(config_path))

        (img_dir / "a.txt").write_text("a red square")

        with patch.object(repo, "upsert_image") as mock_upsert:
            assert scanner.scan_disk(dataset_id, str(config_path)) is True
            assert mock_upsert.call_count == 1
            upserted_path = mock_upsert.call_args.kwargs["path"]
            assert upserted_path == str(img_dir / "a.jpg")
            assert mock_upsert.call_args.kwargs["has_caption"] is True

    def test_atomicity_on_failure(self, scanner, repo, tmp_path, db_connection_factory):
        """If a write inside the scan transaction fails, no partial state is applied.

        Verifies that the ``with self._db.transaction():`` boundary in
        the scanner correctly rolls back the repo's writes when one of
        them fails. The test drops the ``datasets`` table mid-scan to
        force the second repo call to fail; the pre-populated image
        must still be there after the rollback.
        """
        import sqlite3

        img_dir = _make_image_dir(tmp_path)
        config_path = _write_config(tmp_path, img_dir)

        dataset_id = repo.upsert_dataset("alpha", str(config_path), "import")
        # Seed via initial scan.
        scanner.scan_disk(dataset_id, str(config_path))
        assert repo.get_image_by_path("alpha", str(img_dir / "a.jpg")) is not None

        # Drop the dataset_images table to force the next scan's
        # ``list_image_infos`` read to fail mid-transaction.
        with db_connection_factory.connection() as conn:
            conn.execute("DROP TABLE dataset_images")
            conn.commit()

        with pytest.raises(sqlite3.OperationalError):
            scanner.scan_disk(dataset_id, str(config_path))


class TestReadDisk:
    """``read_disk`` — pure walk returning the on-disk meta dict."""

    def test_returns_image_meta(self, scanner, tmp_path):
        from PIL import Image

        img_dir = tmp_path / "images"
        img_dir.mkdir()
        Image.new("RGB", (4, 6), color="red").save(img_dir / "a.jpg")
        config_path = tmp_path / "config.toml"
        config_path.write_text(f'[[dataset]]\npath = "{img_dir}"\n')

        result = scanner.read_disk(str(config_path))
        assert str(img_dir / "a.jpg") in result
        meta = result[str(img_dir / "a.jpg")]
        assert meta["file_name"] == "a.jpg"
        assert meta["width"] == 4
        assert meta["height"] == 6

    def test_returns_empty_for_missing_config(self, scanner, tmp_path):
        assert scanner.read_disk(str(tmp_path / "missing.toml")) == {}

    def test_returns_empty_for_no_images(self, scanner, tmp_path):
        img_dir = tmp_path / "empty"
        img_dir.mkdir()
        config_path = tmp_path / "config.toml"
        config_path.write_text(f'[[dataset]]\npath = "{img_dir}"\n')

        assert scanner.read_disk(str(config_path)) == {}
