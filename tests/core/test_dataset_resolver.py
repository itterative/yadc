from yadc.core.config import ConfigDatasetEntry
from yadc.core.dataset import DatasetImage
from yadc.core.dataset_resolver import apply_extras_defaults, reapply_dataset_extras, resolve_dataset


def _make_image(path: str, caption: str = "", **extras) -> DatasetImage:
    """Helper to create a DatasetImage with extras."""
    img = DatasetImage(path=path, caption=caption, **extras)
    assert img.__pydantic_extra__ is not None
    return img


def _get_extras(img: DatasetImage) -> dict:
    """Get the extras dict from a DatasetImage, with a type-narrowing assert."""
    assert img.__pydantic_extra__ is not None
    return img.__pydantic_extra__


class TestApplyExtrasDefaults:
    def test_no_extras(self):
        img = _make_image("test.png", style="anime")
        apply_extras_defaults(img, {})
        assert _get_extras(img) == {"style": "anime"}

    def test_applies_missing_keys(self):
        img = _make_image("test.png", style="anime")
        apply_extras_defaults(img, {"style": "photo", "artist": "bob"})
        assert _get_extras(img) == {"style": "anime", "artist": "bob"}

    def test_does_not_override_existing(self):
        img = _make_image("test.png", style="anime")
        apply_extras_defaults(img, {"style": "photo"})
        assert _get_extras(img)["style"] == "anime"

    def test_sets_on_empty_extras(self):
        img = _make_image("test.png")
        apply_extras_defaults(img, {"style": "photo"})
        assert _get_extras(img) == {"style": "photo"}

    def test_none_extras_on_image(self):
        img = DatasetImage(path="test.png")
        assert _get_extras(img) == {}
        apply_extras_defaults(img, {"style": "photo"})
        assert _get_extras(img) == {"style": "photo"}


class TestReapplyDatasetExtras:
    def test_no_dataset_extras(self):
        img = _make_image("test.png", style="anime")
        reapply_dataset_extras(img)
        assert _get_extras(img) == {"style": "anime"}

    def test_reapplies_after_reconstruction(self):
        img = _make_image("test.png", style="anime", artist="bob")
        img._dataset_extras = {"style": "photo", "universe": "genshin"}

        # simulate edit path: reconstruct from TOML without extras
        reconstructed = _make_image(
            img.path,
            caption=img.caption,
            style="portrait",  # user edited this
        )
        reconstructed._dataset_extras = img._dataset_extras

        reapply_dataset_extras(reconstructed)
        extras = _get_extras(reconstructed)
        assert extras["style"] == "portrait"  # per-image (edited) takes priority
        assert extras["universe"] == "genshin"  # dataset extra applied as default

    def test_per_image_overwrites_dataset_on_conflict(self):
        img = _make_image("test.png", style="anime")
        img._dataset_extras = {"style": "photo", "artist": "unknown"}

        # simulate edit path: user changed style to 'portrait' in TOML
        reconstructed = _make_image(
            img.path,
            caption=img.caption,
            style="portrait",
        )
        reconstructed._dataset_extras = img._dataset_extras

        reapply_dataset_extras(reconstructed)

        extras = _get_extras(reconstructed)
        assert extras["style"] == "portrait"  # per-image wins, not 'photo' from dataset
        assert extras["artist"] == "unknown"  # dataset extra still applied as default


class TestResolveDataset:
    def test_empty_entries(self):
        images = resolve_dataset([], ".txt")
        assert images == []

    def test_path_entry_scans_directory(self, tmp_path):
        img_file = tmp_path / "photo.png"
        img_file.write_bytes(b"fake")
        toml_file = tmp_path / "photo.toml"
        toml_file.write_text('style = "landscape"\n')

        scanned = _make_image(str(img_file.resolve()), style="landscape")

        def mock_read(path, suffix):
            if path == str(img_file.resolve()):
                return scanned
            return None

        entry = ConfigDatasetEntry(path=str(tmp_path))
        images = resolve_dataset([entry], ".txt", read_image=mock_read)

        assert len(images) == 1
        assert _get_extras(images[0])["style"] == "landscape"

    def test_path_entry_applies_extras(self, tmp_path):
        img_file = tmp_path / "photo.png"
        img_file.write_bytes(b"fake")
        toml_file = tmp_path / "photo.toml"
        toml_file.write_text('style = "landscape"\n')

        scanned = _make_image(str(img_file.resolve()), style="landscape")

        def mock_read(path, suffix):
            if path == str(img_file.resolve()):
                return scanned
            return None

        entry = ConfigDatasetEntry(path=str(tmp_path), extras={"universe": "genshin"})
        images = resolve_dataset([entry], ".txt", read_image=mock_read)

        assert len(images) == 1
        assert _get_extras(images[0])["style"] == "landscape"
        assert _get_extras(images[0])["universe"] == "genshin"
        assert images[0]._dataset_extras == {"universe": "genshin"}

    def test_inline_images_only(self):
        entry = ConfigDatasetEntry(
            images=[
                _make_image("img1.png", name="alice"),
                _make_image("img2.png", name="bob"),
            ]
        )
        images = resolve_dataset([entry], ".txt", read_image=lambda p, s: None)

        assert len(images) == 2
        assert _get_extras(images[0])["name"] == "alice"
        assert _get_extras(images[1])["name"] == "bob"

    def test_inline_image_overrides_scanned_extras(self, tmp_path):
        img_file = tmp_path / "photo.png"
        img_file.write_bytes(b"fake")
        toml_file = tmp_path / "photo.toml"
        toml_file.write_text('style = "landscape"\n')

        # path entry with extras
        entry = ConfigDatasetEntry(
            path=str(tmp_path),
            extras={"style": "photo", "artist": "unknown"},
            images=[_make_image(str((tmp_path / "photo.png").resolve()), style="portrait")],
        )

        images = resolve_dataset([entry], ".txt")

        assert len(images) == 1
        # scanned image gets dataset extras, then inline overrides 'style'
        assert _get_extras(images[0])["style"] == "portrait"
        assert _get_extras(images[0])["artist"] == "unknown"

    def test_inline_image_applies_dataset_extras(self):
        entry = ConfigDatasetEntry(
            extras={"style": "anime"},
            images=[_make_image("new_img.png", name="alice")],
        )

        images = resolve_dataset([entry], ".txt", read_image=lambda p, s: None)

        assert len(images) == 1
        assert _get_extras(images[0])["name"] == "alice"
        assert _get_extras(images[0])["style"] == "anime"

    def test_inline_extras_override_dataset_extras(self):
        entry = ConfigDatasetEntry(
            extras={"style": "anime", "artist": "unknown"},
            images=[_make_image("img.png", style="photo")],
        )

        images = resolve_dataset([entry], ".txt", read_image=lambda p, s: None)

        assert len(images) == 1
        # per-image style overrides dataset style
        assert _get_extras(images[0])["style"] == "photo"
        # dataset artist is applied as default (not in per-image)
        assert _get_extras(images[0])["artist"] == "unknown"

    def test_multiple_entries(self):
        entry1 = ConfigDatasetEntry(
            extras={"style": "anime"},
            images=[_make_image("img1.png")],
        )
        entry2 = ConfigDatasetEntry(
            extras={"style": "photo"},
            images=[_make_image("img2.png")],
        )

        images = resolve_dataset([entry1, entry2], ".txt", read_image=lambda p, s: None)

        assert len(images) == 2
        assert _get_extras(images[0])["style"] == "anime"
        assert _get_extras(images[1])["style"] == "photo"

    def test_read_image_receives_caption_suffix(self):
        received = {}

        def mock_read(path, caption_suffix):
            received["path"] = path
            received["suffix"] = caption_suffix
            return None

        entry = ConfigDatasetEntry(images=[_make_image("img.png")])
        resolve_dataset([entry], ".caption", read_image=mock_read)

        assert received["suffix"] == ".caption"
