"""Tests for yadc.core.exporters — caption export backends."""

import json
import struct
import zlib

import pytest

from yadc.core.dataset import DatasetImage
from yadc.core.exporters import get_backend, list_backends, run_export
from yadc.core.exporters.utils import read_caption_source

# ---- helpers ----


def _make_png() -> bytes:
    """Create a minimal valid 1×1 PNG."""
    header = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    ihdr_crc = zlib.crc32(b"IHDR" + ihdr_data) & 0xFFFFFFFF
    ihdr = struct.pack(">I", 13) + b"IHDR" + ihdr_data + struct.pack(">I", ihdr_crc)
    raw = zlib.compress(b"\x00\x00\x00\x00")
    idat_crc = zlib.crc32(b"IDAT" + raw) & 0xFFFFFFFF
    idat = struct.pack(">I", len(raw)) + b"IDAT" + raw + struct.pack(">I", idat_crc)
    iend_crc = zlib.crc32(b"IEND") & 0xFFFFFFFF
    iend = struct.pack(">I", 0) + b"IEND" + struct.pack(">I", iend_crc)
    return header + ihdr + idat + iend


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
            draft_name="",
            output=out_dir,
            append=False,
            caption_extension=".caption",
        )

        assert count == 2
        assert (out_dir / "img001.caption").read_text() == "1girl, hatsune miku, vocaloid\n"
        assert (out_dir / "img002.caption").read_text() == "1girl, sakura, cherry blossoms\n"
        assert not (out_dir / "img003.caption").exists()

    def test_export_draft_source(self, tmp_images):
        tmp_path, images = tmp_images
        out_dir = tmp_path / "output"
        out_dir.mkdir()

        count = run_export(
            "sd-scripts",
            images,
            fmt="txt",
            source="draft",
            draft_name="test",
            output=out_dir,
            append=False,
            caption_extension=".caption",
        )

        assert count == 2
        assert (out_dir / "img001.caption").read_text() == "draft: miku on stage\n"

    def test_export_txt_alongside_images(self, tmp_images):
        tmp_path, images = tmp_images

        count = run_export(
            "sd-scripts",
            images,
            fmt="txt",
            source="caption",
            draft_name="",
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
            draft_name="",
            output=out_dir,
            append=False,
            caption_extension=".caption",
        )
        run_export(
            "sd-scripts",
            images[1:2],
            fmt="txt",
            source="caption",
            draft_name="",
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
            draft_name="",
            output=output,
            append=False,
            caption_extension=".caption",
        )

        assert count == 2
        data = json.loads(output.read_text())
        assert data["img001.png"]["caption"] == "1girl, hatsune miku, vocaloid"
        assert data["img002.png"]["caption"] == "1girl, sakura, cherry blossoms"
        assert "img003.png" not in data

    def test_export_draft_source(self, tmp_images):
        tmp_path, images = tmp_images
        output = tmp_path / "metadata.json"

        count = run_export(
            "sd-scripts",
            images,
            fmt="json",
            source="draft",
            draft_name="test",
            output=output,
            append=False,
            caption_extension=".caption",
        )

        assert count == 2
        data = json.loads(output.read_text())
        assert data["img001.png"]["caption"] == "draft: miku on stage"

    def test_export_json_append_merges(self, tmp_images):
        tmp_path, images = tmp_images
        output = tmp_path / "metadata.json"

        run_export(
            "sd-scripts",
            images[:1],
            fmt="json",
            source="caption",
            draft_name="",
            output=output,
            append=False,
            caption_extension=".caption",
        )
        count = run_export(
            "sd-scripts",
            images[1:2],
            fmt="json",
            source="caption",
            draft_name="",
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
            draft_name="",
            output=output,
            append=False,
            caption_extension=".caption",
        )
        run_export(
            "sd-scripts",
            images[:1],
            fmt="json",
            source="draft",
            draft_name="test",
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
            draft_name="",
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

    def test_export_draft_source(self, tmp_images):
        tmp_path, images = tmp_images
        output = tmp_path / "metadata.jsonl"

        count = run_export(
            "sd-scripts",
            images,
            fmt="jsonl",
            source="draft",
            draft_name="test",
            output=output,
            append=False,
            caption_extension=".caption",
        )

        assert count == 2
        lines = output.read_text().strip().split("\n")
        assert json.loads(lines[0])["caption"] == "draft: miku on stage"

    def test_export_jsonl_append(self, tmp_images):
        tmp_path, images = tmp_images
        output = tmp_path / "metadata.jsonl"

        run_export(
            "sd-scripts",
            images[:1],
            fmt="jsonl",
            source="caption",
            draft_name="",
            output=output,
            append=False,
            caption_extension=".caption",
        )
        count = run_export(
            "sd-scripts",
            images[1:2],
            fmt="jsonl",
            source="caption",
            draft_name="",
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
            draft_name="",
            output=output,
            append=False,
            caption_extension=".caption",
        )
        run_export(
            "sd-scripts",
            images[1:2],
            fmt="jsonl",
            source="caption",
            draft_name="",
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
            read_caption_source(images[0], source="draft", draft_name="nonexistent")

    def test_missing_caption_raises(self, tmp_images):
        tmp_path, images = tmp_images
        with pytest.raises(FileNotFoundError):
            read_caption_source(images[2], source="caption")

    def test_draft_without_name_raises(self, tmp_images):
        tmp_path, images = tmp_images
        with pytest.raises(ValueError, match="draft_name is required"):
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
                draft_name="",
                output=output,
                append=False,
                caption_extension=".caption",
            )
