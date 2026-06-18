"""Tests for yadc.core.exporters — caption export backends."""

import json
import zipfile

import pytest

from yadc.core.dataset import DatasetImage
from yadc.core.exporters import get_backend, list_backends, run_export, run_export_zip
from yadc.core.exporters.utils import read_caption_source

# ---- helpers ----


def _make_png() -> bytes:
    """Create a minimal valid 1×1 PNG."""
    from io import BytesIO

    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", (1, 1), color="red").save(buf, format="png")
    buf.seek(0)
    return buf.getvalue()


@pytest.fixture
def tmp_images(tmp_path):
    """Create a set of fake images with caption and draft files."""
    images = []
    for name, caption_text, draft_text in [
        ("img001.png", "1girl, hatsune miku, vocaloid", "draft: miku on stage"),
        ("img002.png", "1girl, sakura, cherry blossoms", "draft: sakura in spring"),
        ("img003.png", "", ""),  # no caption
    ]:
        img_path = tmp_path / name
        img_path.write_bytes(_make_png())

        if caption_text:
            (tmp_path / (img_path.stem + ".txt")).write_text(caption_text)
        if draft_text:
            (tmp_path / (img_path.stem + ".test.draft~")).write_text(draft_text)

        images.append(DatasetImage(path=str(img_path), caption_suffix=".txt"))

    return tmp_path, images


@pytest.fixture
def tmp_images_multi_draft(tmp_path):
    """Create images with two drafts each."""
    images = []
    for name, caption_text in [
        ("img001.png", "1girl, hatsune miku, vocaloid"),
        ("img002.png", "1girl, sakura, cherry blossoms"),
    ]:
        img_path = tmp_path / name
        img_path.write_bytes(_make_png())
        (tmp_path / (img_path.stem + ".txt")).write_text(caption_text)
        (tmp_path / (img_path.stem + ".gemma.draft~")).write_text("gemma: anime girl")
        (tmp_path / (img_path.stem + ".qwen.draft~")).write_text("qwen: colorful scene")

        images.append(DatasetImage(path=str(img_path), caption_suffix=".txt"))

    return tmp_path, images


# ---- registry tests ----


class TestRegistry:
    def test_sd_scripts_registered(self):
        backends = list_backends()
        assert "sd-scripts" in backends
        assert backends["sd-scripts"].formats == ("json", "jsonl", "txt")

    def test_get_backend(self):
        b = get_backend("sd-scripts")
        assert b.name == "sd-scripts"

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            get_backend("nonexistent")


# ---- sd-scripts: txt format ----


class TestExportTxt:
    def test_export_caption_source(self, tmp_images):
        tmp_path, images = tmp_images
        out_dir = tmp_path / "output"
        out_dir.mkdir()

        count = run_export(
            "sd-scripts",
            images,
            fmt="txt",
            source="caption",
            output=out_dir,
            append=False,
            caption_extension=".caption",
        )

        assert count == 2
        assert (out_dir / "img001.caption").read_text() == "1girl, hatsune miku, vocaloid\n"
        assert (out_dir / "img002.caption").read_text() == "1girl, sakura, cherry blossoms\n"
        assert not (out_dir / "img003.caption").exists()

    def test_export_single_draft(self, tmp_images):
        tmp_path, images = tmp_images
        out_dir = tmp_path / "output"
        out_dir.mkdir()

        count = run_export(
            "sd-scripts",
            images,
            fmt="txt",
            source="draft",
            drafts=("test",),
            output=out_dir,
            append=False,
            caption_extension=".caption",
        )

        assert count == 2
        assert (out_dir / "img001.caption").read_text() == "draft: miku on stage\n"

    def test_export_caption_with_one_draft(self, tmp_images):
        tmp_path, images = tmp_images
        out_dir = tmp_path / "output"
        out_dir.mkdir()

        count = run_export(
            "sd-scripts",
            images,
            fmt="txt",
            source="caption",
            drafts=("test",),
            output=out_dir,
            append=False,
            caption_extension=".caption",
        )

        assert count == 2
        assert (out_dir / "img001.caption").read_text() == "1girl, hatsune miku, vocaloid\ndraft: miku on stage\n"
        assert (out_dir / "img002.caption").read_text() == "1girl, sakura, cherry blossoms\ndraft: sakura in spring\n"

    def test_export_caption_with_multiple_drafts(self, tmp_images_multi_draft):
        tmp_path, images = tmp_images_multi_draft
        out_dir = tmp_path / "output"
        out_dir.mkdir()

        count = run_export(
            "sd-scripts",
            images,
            fmt="txt",
            source="caption",
            drafts=("gemma", "qwen"),
            output=out_dir,
            append=False,
            caption_extension=".caption",
        )

        assert count == 2
        assert (out_dir / "img001.caption").read_text() == "1girl, hatsune miku, vocaloid\ngemma: anime girl\nqwen: colorful scene\n"

    def test_export_draft_with_draft(self, tmp_images_multi_draft):
        tmp_path, images = tmp_images_multi_draft
        out_dir = tmp_path / "output"
        out_dir.mkdir()

        count = run_export(
            "sd-scripts",
            images,
            fmt="txt",
            source="draft",
            drafts=("gemma", "qwen"),
            output=out_dir,
            append=False,
            caption_extension=".caption",
        )

        assert count == 2
        assert (out_dir / "img001.caption").read_text() == "gemma: anime girl\nqwen: colorful scene\n"

    def test_export_txt_alongside_images(self, tmp_images):
        tmp_path, images = tmp_images

        count = run_export(
            "sd-scripts",
            images,
            fmt="txt",
            source="caption",
            output=None,
            append=False,
            caption_extension=".caption",
        )

        assert count == 2
        assert (tmp_path / "img001.caption").read_text() == "1girl, hatsune miku, vocaloid\n"
        assert (tmp_path / "img001.txt").exists()

    def test_export_txt_append(self, tmp_images):
        tmp_path, images = tmp_images
        out_dir = tmp_path / "output"
        out_dir.mkdir()

        run_export(
            "sd-scripts",
            images[:1],
            fmt="txt",
            source="caption",
            output=out_dir,
            append=False,
            caption_extension=".caption",
        )
        run_export(
            "sd-scripts",
            images[1:2],
            fmt="txt",
            source="caption",
            output=out_dir,
            append=True,
            caption_extension=".caption",
        )

        assert (out_dir / "img001.caption").read_text() == "1girl, hatsune miku, vocaloid\n"
        assert (out_dir / "img002.caption").read_text() == "1girl, sakura, cherry blossoms\n"


# ---- sd-scripts: json format ----


class TestExportJson:
    def test_export_caption_source(self, tmp_images):
        tmp_path, images = tmp_images
        output = tmp_path / "metadata.json"

        count = run_export(
            "sd-scripts",
            images,
            fmt="json",
            source="caption",
            output=output,
            append=False,
            caption_extension=".caption",
        )

        assert count == 2
        data = json.loads(output.read_text())
        assert data["img001.png"]["caption"] == "1girl, hatsune miku, vocaloid"
        assert data["img002.png"]["caption"] == "1girl, sakura, cherry blossoms"
        assert "img003.png" not in data

    def test_export_single_draft(self, tmp_images):
        tmp_path, images = tmp_images
        output = tmp_path / "metadata.json"

        count = run_export(
            "sd-scripts",
            images,
            fmt="json",
            source="draft",
            drafts=("test",),
            output=output,
            append=False,
            caption_extension=".caption",
        )

        assert count == 2
        data = json.loads(output.read_text())
        assert data["img001.png"]["caption"] == "draft: miku on stage"

    def test_export_caption_with_one_draft(self, tmp_images):
        tmp_path, images = tmp_images
        output = tmp_path / "metadata.json"

        count = run_export(
            "sd-scripts",
            images,
            fmt="json",
            source="caption",
            drafts=("test",),
            output=output,
            append=False,
            caption_extension=".caption",
        )

        assert count == 2
        data = json.loads(output.read_text())
        assert data["img001.png"]["caption"] == "1girl, hatsune miku, vocaloid\ndraft: miku on stage"
        assert data["img002.png"]["caption"] == "1girl, sakura, cherry blossoms\ndraft: sakura in spring"

    def test_export_caption_with_multiple_drafts(self, tmp_images_multi_draft):
        tmp_path, images = tmp_images_multi_draft
        output = tmp_path / "metadata.json"

        count = run_export(
            "sd-scripts",
            images,
            fmt="json",
            source="caption",
            drafts=("gemma", "qwen"),
            output=output,
            append=False,
            caption_extension=".caption",
        )

        assert count == 2
        data = json.loads(output.read_text())
        assert data["img001.png"]["caption"] == "1girl, hatsune miku, vocaloid\ngemma: anime girl\nqwen: colorful scene"

    def test_export_draft_with_draft(self, tmp_images_multi_draft):
        tmp_path, images = tmp_images_multi_draft
        output = tmp_path / "metadata.json"

        count = run_export(
            "sd-scripts",
            images,
            fmt="json",
            source="draft",
            drafts=("gemma", "qwen"),
            output=output,
            append=False,
            caption_extension=".caption",
        )

        assert count == 2
        data = json.loads(output.read_text())
        assert data["img001.png"]["caption"] == "gemma: anime girl\nqwen: colorful scene"

    def test_export_json_append_merges(self, tmp_images):
        tmp_path, images = tmp_images
        output = tmp_path / "metadata.json"

        run_export(
            "sd-scripts",
            images[:1],
            fmt="json",
            source="caption",
            output=output,
            append=False,
            caption_extension=".caption",
        )
        count = run_export(
            "sd-scripts",
            images[1:2],
            fmt="json",
            source="caption",
            output=output,
            append=True,
            caption_extension=".caption",
        )

        assert count == 1
        data = json.loads(output.read_text())
        assert "img001.png" in data
        assert "img002.png" in data

    def test_export_json_append_overwrites_same_key(self, tmp_images):
        tmp_path, images = tmp_images
        output = tmp_path / "metadata.json"

        run_export(
            "sd-scripts",
            images[:1],
            fmt="json",
            source="caption",
            output=output,
            append=False,
            caption_extension=".caption",
        )
        run_export(
            "sd-scripts",
            images[:1],
            fmt="json",
            source="draft",
            drafts=("test",),
            output=output,
            append=True,
            caption_extension=".caption",
        )

        data = json.loads(output.read_text())
        assert data["img001.png"]["caption"] == "draft: miku on stage"


# ---- sd-scripts: jsonl format ----


class TestExportJsonl:
    def test_export_caption_source(self, tmp_images):
        tmp_path, images = tmp_images
        output = tmp_path / "metadata.jsonl"

        count = run_export(
            "sd-scripts",
            images,
            fmt="jsonl",
            source="caption",
            output=output,
            append=False,
            caption_extension=".caption",
        )

        assert count == 2
        lines = output.read_text().strip().split("\n")
        assert len(lines) == 2

        entry1 = json.loads(lines[0])
        assert entry1["image_path"] == "img001.png"
        assert entry1["caption"] == "1girl, hatsune miku, vocaloid"

        entry2 = json.loads(lines[1])
        assert entry2["image_path"] == "img002.png"

    def test_export_single_draft(self, tmp_images):
        tmp_path, images = tmp_images
        output = tmp_path / "metadata.jsonl"

        count = run_export(
            "sd-scripts",
            images,
            fmt="jsonl",
            source="draft",
            drafts=("test",),
            output=output,
            append=False,
            caption_extension=".caption",
        )

        assert count == 2
        lines = output.read_text().strip().split("\n")
        assert json.loads(lines[0])["caption"] == "draft: miku on stage"

    def test_export_caption_with_one_draft(self, tmp_images):
        tmp_path, images = tmp_images
        output = tmp_path / "metadata.jsonl"

        count = run_export(
            "sd-scripts",
            images,
            fmt="jsonl",
            source="caption",
            drafts=("test",),
            output=output,
            append=False,
            caption_extension=".caption",
        )

        assert count == 2
        lines = output.read_text().strip().split("\n")
        entry1 = json.loads(lines[0])
        assert entry1["caption"] == "1girl, hatsune miku, vocaloid\ndraft: miku on stage"
        entry2 = json.loads(lines[1])
        assert entry2["caption"] == "1girl, sakura, cherry blossoms\ndraft: sakura in spring"

    def test_export_caption_with_multiple_drafts(self, tmp_images_multi_draft):
        tmp_path, images = tmp_images_multi_draft
        output = tmp_path / "metadata.jsonl"

        count = run_export(
            "sd-scripts",
            images,
            fmt="jsonl",
            source="caption",
            drafts=("gemma", "qwen"),
            output=output,
            append=False,
            caption_extension=".caption",
        )

        assert count == 2
        lines = output.read_text().strip().split("\n")
        entry1 = json.loads(lines[0])
        assert entry1["caption"] == "1girl, hatsune miku, vocaloid\ngemma: anime girl\nqwen: colorful scene"

    def test_export_draft_with_draft(self, tmp_images_multi_draft):
        tmp_path, images = tmp_images_multi_draft
        output = tmp_path / "metadata.jsonl"

        count = run_export(
            "sd-scripts",
            images,
            fmt="jsonl",
            source="draft",
            drafts=("gemma", "qwen"),
            output=output,
            append=False,
            caption_extension=".caption",
        )

        assert count == 2
        lines = output.read_text().strip().split("\n")
        entry1 = json.loads(lines[0])
        assert entry1["caption"] == "gemma: anime girl\nqwen: colorful scene"

    def test_export_jsonl_append(self, tmp_images):
        tmp_path, images = tmp_images
        output = tmp_path / "metadata.jsonl"

        run_export(
            "sd-scripts",
            images[:1],
            fmt="jsonl",
            source="caption",
            output=output,
            append=False,
            caption_extension=".caption",
        )
        count = run_export(
            "sd-scripts",
            images[1:2],
            fmt="jsonl",
            source="caption",
            output=output,
            append=True,
            caption_extension=".caption",
        )

        assert count == 1
        lines = output.read_text().strip().split("\n")
        assert len(lines) == 2

    def test_export_jsonl_no_append_overwrites(self, tmp_images):
        tmp_path, images = tmp_images
        output = tmp_path / "metadata.jsonl"

        run_export(
            "sd-scripts",
            images[:1],
            fmt="jsonl",
            source="caption",
            output=output,
            append=False,
            caption_extension=".caption",
        )
        run_export(
            "sd-scripts",
            images[1:2],
            fmt="jsonl",
            source="caption",
            output=output,
            append=False,
            caption_extension=".caption",
        )

        lines = output.read_text().strip().split("\n")
        assert len(lines) == 1
        assert json.loads(lines[0])["image_path"] == "img002.png"


# ---- read_caption_source utility ----


class TestReadCaptionSource:
    def test_missing_draft_raises(self, tmp_images):
        tmp_path, images = tmp_images
        with pytest.raises(FileNotFoundError):
            read_caption_source(images[0], source="draft", drafts=("nonexistent",))

    def test_missing_caption_raises(self, tmp_images):
        tmp_path, images = tmp_images
        with pytest.raises(FileNotFoundError):
            read_caption_source(images[2], source="caption")

    def test_draft_without_drafts_raises(self, tmp_images):
        tmp_path, images = tmp_images
        with pytest.raises(ValueError, match="drafts is required"):
            read_caption_source(images[0], source="draft")

    def test_invalid_format_for_backend(self, tmp_images):
        tmp_path, images = tmp_images
        output = tmp_path / "metadata.jsonl"
        with pytest.raises(ValueError, match="does not support format"):
            run_export(
                "sd-scripts",
                images,
                fmt="xml",
                source="caption",
                output=output,
                append=False,
                caption_extension=".caption",
            )

    def test_caption_with_one_draft(self, tmp_images):
        tmp_path, images = tmp_images
        result = read_caption_source(images[0], source="caption", drafts=("test",))
        assert result == "1girl, hatsune miku, vocaloid\ndraft: miku on stage"

    def test_caption_with_multiple_drafts(self, tmp_images_multi_draft):
        tmp_path, images = tmp_images_multi_draft
        result = read_caption_source(images[0], source="caption", drafts=("gemma", "qwen"))
        assert result == "1girl, hatsune miku, vocaloid\ngemma: anime girl\nqwen: colorful scene"

    def test_draft_with_multiple_drafts(self, tmp_images_multi_draft):
        tmp_path, images = tmp_images_multi_draft
        result = read_caption_source(images[0], source="draft", drafts=("gemma", "qwen"))
        assert result == "gemma: anime girl\nqwen: colorful scene"

    def test_with_drafts_missing_draft_raises(self, tmp_images):
        tmp_path, images = tmp_images
        with pytest.raises(FileNotFoundError, match="Draft not found"):
            read_caption_source(images[0], source="caption", drafts=("nonexistent",))

    def test_with_drafts_missing_caption_raises(self, tmp_images):
        tmp_path, images = tmp_images
        with pytest.raises(FileNotFoundError, match="Caption not found"):
            read_caption_source(images[2], source="caption", drafts=("test",))

    def test_invalid_source_raises(self, tmp_images):
        tmp_path, images = tmp_images
        with pytest.raises(ValueError, match="source must be"):
            read_caption_source(images[0], source="bogus")


# ---- yadc backend: fixtures ----


@pytest.fixture
def tmp_images_full_sidecars(tmp_path):
    """Images carrying the full yadc sidecar set: caption, draft, metadata, backup, history."""
    images = []
    for name, caption_text in [
        ("img001.png", "1girl, hatsune miku, vocaloid"),
        ("img002.png", "1girl, sakura, cherry blossoms"),
        ("img003.png", ""),  # no caption, but still has the other sidecars
    ]:
        img_path = tmp_path / name
        img_path.write_bytes(_make_png())

        stem = img_path.stem
        if caption_text:
            (tmp_path / f"{stem}.txt").write_text(caption_text)
        (tmp_path / f"{stem}.gemma.draft~").write_text("gemma draft")
        (tmp_path / f"{stem}.toml").write_text('foo = "bar"\n')
        (tmp_path / f"{stem}.toml~").write_text('foo = "old"\n')
        (tmp_path / f"{stem}.history~").write_text("snapshot\n----------\n")

        images.append(DatasetImage(path=str(img_path), caption_suffix=".txt"))

    return tmp_path, images


# ---- yadc backend: registry ----


class TestYadcRegistry:
    def test_yadc_registered(self):
        backends = list_backends()
        assert "yadc" in backends
        assert backends["yadc"].formats == ("zip",)
        assert backends["yadc"].zip_only is True

    def test_get_backend(self):
        b = get_backend("yadc")
        assert b.name == "yadc"
        assert b.zip_only is True


# ---- yadc backend: zip ----


class TestYadcZip:
    def test_zip_includes_images_and_all_sidecars(self, tmp_images_full_sidecars):
        tmp_path, images = tmp_images_full_sidecars

        buf, count = run_export_zip(
            "yadc",
            images,
            fmt="zip",
            source="caption",
            base_dir=tmp_path,
        )

        assert count == 3  # all three images archived
        with zipfile.ZipFile(buf) as zf:
            names = set(zf.namelist())

        # images (always included)
        assert {"img001.png", "img002.png", "img003.png"} <= names
        # caption sidecars — only for images that actually have one
        assert {"img001.txt", "img002.txt"} <= names
        assert "img003.txt" not in names
        # drafts
        assert {"img001.gemma.draft~", "img002.gemma.draft~", "img003.gemma.draft~"} <= names
        # metadata, backup, history
        assert {"img001.toml", "img001.toml~", "img001.history~"} <= names

    def test_zip_preserves_sidecar_content(self, tmp_images_full_sidecars):
        tmp_path, images = tmp_images_full_sidecars
        buf, _ = run_export_zip("yadc", images, fmt="zip", source="caption", base_dir=tmp_path)
        with zipfile.ZipFile(buf) as zf:
            assert zf.read("img001.txt").decode() == "1girl, hatsune miku, vocaloid"
            assert zf.read("img001.toml").decode() == 'foo = "bar"\n'

    def test_zip_preserves_relative_paths(self, tmp_path):
        sub = tmp_path / "train"
        sub.mkdir()
        img = sub / "img001.png"
        img.write_bytes(_make_png())
        (sub / "img001.txt").write_text("caption")

        images = [DatasetImage(path=str(img), caption_suffix=".txt")]

        buf, _ = run_export_zip("yadc", images, fmt="zip", source="caption", base_dir=tmp_path)
        with zipfile.ZipFile(buf) as zf:
            names = set(zf.namelist())
        assert {"train/img001.png", "train/img001.txt"} <= names

    def test_zip_does_not_match_longer_filename(self, tmp_path):
        # img0010.* must not be swept in as a sidecar of the img001 image.
        img1 = tmp_path / "img001.png"
        img1.write_bytes(_make_png())
        (tmp_path / "img001.txt").write_text("caption one")
        img10 = tmp_path / "img0010.png"
        img10.write_bytes(_make_png())
        (tmp_path / "img0010.txt").write_text("caption ten")

        images = [DatasetImage(path=str(img1), caption_suffix=".txt")]

        buf, _ = run_export_zip("yadc", images, fmt="zip", source="caption", base_dir=tmp_path)
        with zipfile.ZipFile(buf) as zf:
            names = set(zf.namelist())
        assert {"img001.png", "img001.txt"} <= names
        assert "img0010.png" not in names
        assert "img0010.txt" not in names

    def test_run_raises_zip_only(self, tmp_images_full_sidecars):
        tmp_path, images = tmp_images_full_sidecars
        with pytest.raises(ValueError, match="zips only"):
            run_export(
                "yadc",
                images,
                fmt="zip",
                source="caption",
                output=tmp_path / "out",
            )
