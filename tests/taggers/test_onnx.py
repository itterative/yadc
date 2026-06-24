"""Tests for ``OnnxTagger`` — including the HuggingFace Hub download path.

Download behavior is exercised by mocking ``huggingface_hub.hf_hub_download``
so the tests don't need network access.
"""

from __future__ import annotations

import csv
from pathlib import Path
from unittest.mock import patch

import pytest

from yadc.taggers.onnx import OnnxTagger, load_labels

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_model_file(tmp_path: Path) -> Path:
    """Create a tiny placeholder for the ONNX model file. The actual content doesn't matter for these tests."""
    p = tmp_path / "model.onnx"
    p.write_bytes(b"fake-onnx-bytes")
    return p


@pytest.fixture
def fake_labels_csv(tmp_path: Path) -> Path:
    """A minimal SmilingWolf-style selected_tags.csv with rating/general/character categories."""
    p = tmp_path / "selected_tags.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["tag_id", "name", "category"])
        writer.writeheader()
        writer.writerow({"tag_id": 0, "name": "1girl", "category": 0})
        writer.writerow({"tag_id": 1, "name": "solo", "category": 0})
        writer.writerow({"tag_id": 2, "name": "rei_(ayanami)", "category": 4})
        writer.writerow({"tag_id": 3, "name": "safe", "category": 9})
        writer.writerow({"tag_id": 4, "name": "questionable", "category": 9})
    return p


@pytest.fixture
def fake_labels_txt(tmp_path: Path) -> Path:
    """A flat one-label-per-line file (no categorization)."""
    p = tmp_path / "tags.txt"
    p.write_text("1girl\nsolo\nsmile\n", encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# load_labels (pure function)
# ---------------------------------------------------------------------------


class TestLabels:
    def test_load_labels_csv_parses_categories(self, fake_labels_csv: Path) -> None:
        """CSV labels are split into rating/general/character categories by code."""
        names, categories = load_labels(fake_labels_csv)
        assert names == ["1girl", "solo", "rei_(ayanami)", "safe", "questionable"]
        assert categories == {
            "general": ["1girl", "solo"],
            "character": ["rei_(ayanami)"],
            "rating": ["safe", "questionable"],
        }

    def test_load_labels_txt_is_flat(self, fake_labels_txt: Path) -> None:
        """Plain .txt files produce a flat list with no categories."""
        names, categories = load_labels(fake_labels_txt)
        assert names == ["1girl", "solo", "smile"]
        assert categories == {}


# ---------------------------------------------------------------------------
# OnnxTagger.load_model — HuggingFace download path
# ---------------------------------------------------------------------------


class TestLoadModelHf:
    def test_load_model_downloads_from_hub(self, fake_model_file: Path, fake_labels_csv: Path) -> None:
        """When ``repo_id`` is set, ``load_model`` calls ``hf_hub_download`` for both files."""
        tagger = OnnxTagger(
            repo_id="SmilingWolf/wd-v1-4-vit-tagger-v2",
            repo_model_filename="model.onnx",
            repo_label_filename="selected_tags.csv",
        )

        with (
            patch(
                "huggingface_hub.hf_hub_download",
                side_effect=[str(fake_model_file), str(fake_labels_csv)],
            ) as mock_dl,
            patch.object(tagger, "_create_session"),
        ):
            tagger.load_model("/unused/local/path.onnx")

        assert mock_dl.call_count == 2
        repo_calls = [c for c in mock_dl.call_args_list if c.kwargs.get("repo_id") == "SmilingWolf/wd-v1-4-vit-tagger-v2"]
        filenames = sorted(c.kwargs["filename"] for c in repo_calls)
        assert filenames == ["model.onnx", "selected_tags.csv"]

    def test_load_model_populates_categories_from_downloaded_csv(self, fake_model_file: Path, fake_labels_csv: Path) -> None:
        """After a HF download, the tagger's categories match the downloaded CSV."""
        tagger = OnnxTagger(repo_id="SmilingWolf/wd-v1-4-vit-tagger-v2")

        with (
            patch(
                "huggingface_hub.hf_hub_download",
                side_effect=[str(fake_model_file), str(fake_labels_csv)],
            ),
            patch.object(tagger, "_create_session"),
        ):
            tagger.load_model("/unused")

        assert tagger._categories == {
            "general": ["1girl", "solo"],
            "character": ["rei_(ayanami)"],
            "rating": ["safe", "questionable"],
        }
        assert tagger._labels == ["1girl", "solo", "rei_(ayanami)", "safe", "questionable"]

    def test_load_model_ignores_local_path_when_repo_id_set(self, fake_model_file: Path, fake_labels_csv: Path) -> None:
        """When ``repo_id`` is set, the ``model_path`` argument to ``load_model`` is ignored."""
        tagger = OnnxTagger(repo_id="SmilingWolf/wd-v1-4-vit-tagger-v2")

        with (
            patch(
                "huggingface_hub.hf_hub_download",
                side_effect=[str(fake_model_file), str(fake_labels_csv)],
            ) as mock_dl,
            patch.object(tagger, "_create_session") as mock_session,
        ):
            tagger.load_model("/this/path/should/be/ignored.onnx")

        # The local path must not leak into the download call, and the
        # session must be created from the downloaded path instead.
        for call in mock_dl.call_args_list:
            assert "/this/path" not in str(call)
        assert mock_session.call_args.args[0] == str(fake_model_file)

    def test_load_model_propagates_download_error(self) -> None:
        """A download failure surfaces a clear error to the caller."""
        tagger = OnnxTagger(repo_id="nonexistent/repo")

        with patch(
            "huggingface_hub.hf_hub_download",
            side_effect=RuntimeError("repo not found"),
        ):
            with pytest.raises(RuntimeError, match="repo not found"):
                tagger.load_model("/unused")


# ---------------------------------------------------------------------------
# OnnxTagger.load_model — local path (existing behavior)
# ---------------------------------------------------------------------------


class TestLoadModelLocal:
    def test_load_model_uses_local_path_when_no_repo_id(self, fake_model_file: Path, fake_labels_csv: Path) -> None:
        """When ``repo_id`` is not set, the local ``model_path`` is used directly and no download happens."""
        tagger = OnnxTagger(label_path=fake_labels_csv)

        with patch("huggingface_hub.hf_hub_download") as mock_dl, patch.object(tagger, "_create_session") as mock_session:
            tagger.load_model(str(fake_model_file))

        mock_dl.assert_not_called()
        assert mock_session.call_args.args[0] == str(fake_model_file)
        assert tagger._categories["general"] == ["1girl", "solo"]


# ---------------------------------------------------------------------------
# OnnxTagger._build_result — uncategorized fallback
# ---------------------------------------------------------------------------


class TestResults:
    def test_build_result_is_uncategorized_when_labels_mismatch(self) -> None:
        """When the label count doesn't match the score count, synthesized index tags are uncategorized."""
        import numpy as np

        tagger = OnnxTagger()
        tagger._labels = ["1girl"]  # one label, but three scores
        tagger._categories = {"general": ["1girl"]}
        scores = np.array([0.9, 0.5, 0.2], dtype=np.float32)

        result = tagger._build_result(scores)

        assert result.tags == {"0": pytest.approx(0.9), "1": pytest.approx(0.5), "2": pytest.approx(0.2)}
        assert result.categories == {}

    def test_build_result_is_uncategorized_when_no_labels(self) -> None:
        """With no labels at all, synthesized index tags are uncategorized."""
        import numpy as np

        tagger = OnnxTagger()
        scores = np.array([0.9, 0.5], dtype=np.float32)

        result = tagger._build_result(scores)

        assert result.tags == {"0": pytest.approx(0.9), "1": pytest.approx(0.5)}
        assert result.categories == {}
