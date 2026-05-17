"""Tests for yadc.cli_draft — draft CLI commands."""

import struct
import zlib

import pytest
from click.testing import CliRunner

from yadc.cli_draft import draft

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


def _make_dataset_toml(path: str) -> str:
    return f"""[api]
url = "http://localhost:8080"
model_name = "test"

[settings]
max_tokens = 100

[prompt]
template = "test"

[[dataset]]
path = "{path}"
"""


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def tmp_images(tmp_path):
    """Create a set of fake images with caption files."""
    img1 = tmp_path / "img001.png"
    img1.write_bytes(_make_png())
    (tmp_path / "img001.txt").write_text("1girl, hatsune miku, vocaloid")

    img2 = tmp_path / "img002.png"
    img2.write_bytes(_make_png())
    (tmp_path / "img002.txt").write_text("1girl, sakura, cherry blossoms")

    img3 = tmp_path / "img003.png"
    img3.write_bytes(_make_png())
    # no caption for img003

    return tmp_path


@pytest.fixture
def tmp_images_with_drafts(tmp_images):
    """Add draft files to tmp_images."""
    (tmp_images / "img001.v1.draft~").write_text("draft: miku v1")
    (tmp_images / "img001.v2.draft~").write_text("draft: miku v2")
    (tmp_images / "img002.v1.draft~").write_text("draft: sakura v1")
    return tmp_images


# ---- draft save ----


class TestDraftSave:
    def test_save_basic(self, runner, tmp_images):
        dataset_toml = _make_dataset_toml(str(tmp_images))
        result = runner.invoke(draft, ["save", "--name", "v1", "-"], input=dataset_toml)
        assert result.exit_code == 0, result.output
        assert (tmp_images / "img001.v1.draft~").read_text() == "1girl, hatsune miku, vocaloid"
        assert (tmp_images / "img002.v1.draft~").read_text() == "1girl, sakura, cherry blossoms"
        assert not (tmp_images / "img003.v1.draft~").exists()

    def test_save_skip_missing(self, runner, tmp_images):
        dataset_toml = _make_dataset_toml(str(tmp_images))
        result = runner.invoke(draft, ["save", "--name", "v1", "--skip-missing", "-"], input=dataset_toml)
        assert result.exit_code == 0, result.output
        assert not (tmp_images / "img003.v1.draft~").exists()

    def test_save_no_skip_missing(self, runner, tmp_images):
        dataset_toml = _make_dataset_toml(str(tmp_images))
        result = runner.invoke(draft, ["save", "--name", "v1", "--no-skip-missing", "-"], input=dataset_toml)
        assert result.exit_code == 0, result.output
        assert not (tmp_images / "img003.v1.draft~").exists()

    def test_save_no_overwrite(self, runner, tmp_images):
        (tmp_images / "img001.v1.draft~").write_text("old draft")
        dataset_toml = _make_dataset_toml(str(tmp_images))
        result = runner.invoke(draft, ["save", "--name", "v1", "-"], input=dataset_toml)
        assert result.exit_code == 0, result.output
        assert (tmp_images / "img001.v1.draft~").read_text() == "old draft"

    def test_save_overwrite(self, runner, tmp_images):
        (tmp_images / "img001.v1.draft~").write_text("old draft")
        dataset_toml = _make_dataset_toml(str(tmp_images))
        result = runner.invoke(draft, ["save", "--name", "v1", "--overwrite", "-"], input=dataset_toml)
        assert result.exit_code == 0, result.output
        assert (tmp_images / "img001.v1.draft~").read_text() == "1girl, hatsune miku, vocaloid"

    def test_save_name_required(self, runner, tmp_images):
        dataset_toml = _make_dataset_toml(str(tmp_images))
        result = runner.invoke(draft, ["save", "-"], input=dataset_toml)
        assert result.exit_code != 0

    def test_save_empty_dataset(self, runner, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        dataset_toml = _make_dataset_toml(str(empty_dir))
        result = runner.invoke(draft, ["save", "--name", "v1", "-"], input=dataset_toml)
        assert result.exit_code == 0, result.output

    def test_save_preserves_existing_captions(self, runner, tmp_images):
        dataset_toml = _make_dataset_toml(str(tmp_images))
        result = runner.invoke(draft, ["save", "--name", "backup", "-"], input=dataset_toml)
        assert result.exit_code == 0, result.output
        assert (tmp_images / "img001.txt").read_text() == "1girl, hatsune miku, vocaloid"
        assert (tmp_images / "img002.txt").read_text() == "1girl, sakura, cherry blossoms"


# ---- draft list ----


class TestDraftList:
    def test_list_with_drafts(self, runner, tmp_images_with_drafts):
        dataset_toml = _make_dataset_toml(str(tmp_images_with_drafts))
        result = runner.invoke(draft, ["list", "-"], input=dataset_toml)
        assert result.exit_code == 0, result.output

    def test_list_no_drafts(self, runner, tmp_images):
        dataset_toml = _make_dataset_toml(str(tmp_images))
        result = runner.invoke(draft, ["list", "-"], input=dataset_toml)
        assert result.exit_code == 0, result.output


# ---- draft show ----


class TestDraftShow:
    def test_show_existing_draft(self, runner, tmp_images_with_drafts):
        dataset_toml = _make_dataset_toml(str(tmp_images_with_drafts))
        result = runner.invoke(draft, ["show", "--name", "v1", "-"], input=dataset_toml)
        assert result.exit_code == 0, result.output
        assert "draft: miku v1" in result.output
        assert "draft: sakura v1" in result.output

    def test_show_nonexistent_draft(self, runner, tmp_images):
        dataset_toml = _make_dataset_toml(str(tmp_images))
        result = runner.invoke(draft, ["show", "--name", "nonexistent", "-"], input=dataset_toml)
        assert result.exit_code == 0, result.output
        assert result.output.strip() == ""

    def test_show_name_required(self, runner, tmp_images):
        dataset_toml = _make_dataset_toml(str(tmp_images))
        result = runner.invoke(draft, ["show", "-"], input=dataset_toml)
        assert result.exit_code != 0


# ---- draft remove ----


class TestDraftRemove:
    def test_remove_existing_draft(self, runner, tmp_images_with_drafts):
        dataset_toml = _make_dataset_toml(str(tmp_images_with_drafts))
        result = runner.invoke(draft, ["remove", "--name", "v1", "-"], input=dataset_toml)
        assert result.exit_code == 0, result.output
        assert not (tmp_images_with_drafts / "img001.v1.draft~").exists()
        assert not (tmp_images_with_drafts / "img002.v1.draft~").exists()
        # v2 drafts untouched
        assert (tmp_images_with_drafts / "img001.v2.draft~").exists()

    def test_remove_nonexistent_draft(self, runner, tmp_images):
        dataset_toml = _make_dataset_toml(str(tmp_images))
        result = runner.invoke(draft, ["remove", "--name", "nonexistent", "-"], input=dataset_toml)
        assert result.exit_code == 0, result.output

    def test_remove_name_required(self, runner, tmp_images):
        dataset_toml = _make_dataset_toml(str(tmp_images))
        result = runner.invoke(draft, ["remove", "-"], input=dataset_toml)
        assert result.exit_code != 0
