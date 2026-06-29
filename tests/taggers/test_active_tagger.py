"""Tests for the ``ActiveTagger`` persisted-selection schema.

Pure-model tests; no service or DB.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from yadc.api.modules.tagger_catalog import ActiveTagger


class TestActiveTagger:
    # --- happy path ---

    def test_hf_kind_round_trips_through_json(self):
        """The HF selection survives model_dump -> model_validate as the same model."""
        original = ActiveTagger(
            kind="hf",
            repo_id="SmilingWolf/wd-eva02-large-tagger-v3",
            preproc_profile="wd-tagger",
        )
        restored = ActiveTagger.model_validate(original.model_dump())

        assert restored == original

    def test_local_kind_round_trips_through_json(self):
        """The local selection survives model_dump -> model_validate, including
        optional ``label_path`` / ``default_size``."""
        original = ActiveTagger(
            kind="local",
            model_path="/path/to/model.onnx",
            label_path="/path/to/selected_tags.csv",
            preproc_profile="timm",
            default_size=512,
        )
        restored = ActiveTagger.model_validate(original.model_dump())

        assert restored == original
        assert restored.default_size == 512

    def test_hf_source_label_uses_repo_id(self):
        """HF identity is the standard ``hf:<repo_id>`` SSE format."""
        selection = ActiveTagger(
            kind="hf",
            repo_id="SmilingWolf/wd-eva02-large-tagger-v3",
        )
        assert selection.source_label == "hf:SmilingWolf/wd-eva02-large-tagger-v3"

    def test_local_source_label_uses_model_path(self):
        """Local identity is the standard ``local:<path>`` SSE format."""
        selection = ActiveTagger(
            kind="local",
            model_path="/models/animetimm/model.onnx",
        )
        assert selection.source_label == "local:/models/animetimm/model.onnx"

    def test_frozen_prevents_mutation(self):
        """The selection is frozen so persisted-on-disk and in-process copies can't diverge."""
        selection = ActiveTagger(kind="hf", repo_id="any/repo")
        with pytest.raises(ValidationError):
            selection.repo_id = "other/repo"  # type: ignore[misc]

    # --- rejection cases ---

    def test_hf_kind_requires_repo_id(self):
        """``kind='hf'`` + empty ``repo_id`` is rejected — the subprocess would
        otherwise silently fall back to the empty ``model_path`` and fail."""
        with pytest.raises(ValidationError, match="repo_id"):
            ActiveTagger(kind="hf", repo_id="")

    def test_hf_kind_requires_repo_id_stripped(self):
        """Whitespace-only ``repo_id`` is treated as empty."""
        with pytest.raises(ValidationError, match="repo_id"):
            ActiveTagger(kind="hf", repo_id="   ")

    def test_local_kind_requires_model_path(self):
        """``kind='local'`` + empty ``model_path`` is rejected for the same reason."""
        with pytest.raises(ValidationError, match="model_path"):
            ActiveTagger(kind="local", model_path="")

    def test_local_kind_requires_model_path_stripped(self):
        """Whitespace-only ``model_path`` is treated as empty."""
        with pytest.raises(ValidationError, match="model_path"):
            ActiveTagger(kind="local", model_path="   ")

    def test_unknown_kind_rejected(self):
        """The discriminated kind only accepts the two known string literals."""
        with pytest.raises(ValidationError):
            ActiveTagger.model_validate(
                {
                    "kind": "weird",
                    "repo_id": "x/y",
                    "model_path": "",
                    "label_path": "",
                    "preproc_profile": "wd-tagger",
                    "default_size": 0,
                }
            )
