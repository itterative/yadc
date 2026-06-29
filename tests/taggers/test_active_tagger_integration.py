"""Tests for the Phase-2 integration: ``ActiveTagger`` actually drives
the subprocess spawn + cache key + source label.

Covers:

- :meth:`TaggingService._source_label` reads ``active_tagger`` first,
  falls back to ``Configuration`` fields.
- :meth:`TaggingService._subprocess_spawn_args` builds the correct
  kwargs from a persisted selection (HF / local) and from the legacy
  Configuration path; rejects unknown preproc profiles.
- :meth:`TaggingService._tag_result_key` derives ``model_id`` from
  ``active_tagger.source_label`` so swapping the model invalidates
  cold-path hits naturally.
- :attr:`TaggingService.is_configured` returns ``True`` when either
  source has a model.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from yadc.api.configuration import Configuration
from yadc.api.modules.tagger_catalog import ActiveTagger
from yadc.api.services.tagging import TaggingService
from yadc.taggers.onnx_preprocess import TIMM_PROFILE, WD_TAGGER_PROFILE

# ---------------------------------------------------------------------------
# _source_label
# ---------------------------------------------------------------------------


class TestSourceLabelDrivenByActiveTagger:
    def test_uses_active_tagger_source_label_when_set(self, service: TaggingService):
        """A persisted selection takes precedence over the flat ``Configuration``."""
        service.set_active_tagger(
            ActiveTagger(
                kind="hf",
                repo_id="SmilingWolf/wd-vit-tagger-v3",
                preproc_profile="wd-tagger",
            )
        )
        assert service._source_label() == "hf:SmilingWolf/wd-vit-tagger-v3"

    def test_local_active_tagger_overrides_configured_model_path(self, service: TaggingService):
        """``active_tagger`` overrides the flat ``tagger_model_path`` field.

        Uses the ``service`` fixture (which configures a local-path
        tagger via ``test_configuration``), then swaps it for an HF
        repo via ``set_active_tagger``. The source label follows the
        swapped selection, not the configured local path.
        """
        assert service._source_label() == "local:/fake/model.onnx"  # legacy before swap
        service.set_active_tagger(
            ActiveTagger(
                kind="local",
                model_path="/models/animetimm/model.onnx",
                preproc_profile="timm",
            )
        )
        assert service._source_label() == "local:/models/animetimm/model.onnx"

    def test_no_active_tagger_falls_back_to_hf_repo_id(self, test_configuration: Configuration, service: TaggingService):
        """Legacy setups (no swap yet) read ``tagger_repo_id`` directly."""
        test_configuration.tagger_repo_id = "SmilingWolf/wd-eva02-large-tagger-v3"
        test_configuration.tagger_model_path = ""
        assert service._source_label() == "hf:SmilingWolf/wd-eva02-large-tagger-v3"

    def test_no_active_tagger_falls_back_to_model_path(self, test_configuration: Configuration, service: TaggingService):
        """Legacy local-path setups emit ``local:<path>`` when active_tagger is None."""
        test_configuration.tagger_repo_id = ""
        test_configuration.tagger_model_path = "/fake/model.onnx"
        assert service._source_label() == "local:/fake/model.onnx"

    def test_no_active_tagger_no_config_returns_empty(self, test_configuration: Configuration, service: TaggingService):
        """A fully unconfigured service emits the empty-source SSE payload.

        The shared ``test_configuration`` fixture now sets all
        tagger fields to empty defaults, so a vanilla service built
        from it is unconfigured.
        """
        test_configuration.tagger_repo_id = ""
        test_configuration.tagger_model_path = ""
        assert service._source_label() == ""


# ---------------------------------------------------------------------------
# _subprocess_spawn_args
# ---------------------------------------------------------------------------


class TestSubprocessSpawnArgs:
    def test_hf_active_tagger_produces_repo_kwargs(self, service: TaggingService):
        """An HF ActiveTagger yields the HF download kwargs; model_path is empty."""
        service.set_active_tagger(
            ActiveTagger(
                kind="hf",
                repo_id="SmilingWolf/wd-vit-tagger-v3",
                repo_model_filename="model.onnx",
                repo_label_filename="selected_tags.csv",
                preproc_profile="wd-tagger",
            )
        )

        kwargs, model_path = service._subprocess_spawn_args()

        assert model_path == ""
        assert kwargs["repo_id"] == "SmilingWolf/wd-vit-tagger-v3"
        assert kwargs["repo_model_filename"] == "model.onnx"
        assert kwargs["repo_label_filename"] == "selected_tags.csv"
        assert kwargs["preproc_profile"] == WD_TAGGER_PROFILE
        assert "label_path" not in kwargs

    def test_local_active_tagger_uses_model_path_and_explicit_label(self, service: TaggingService):
        """A local ActiveTagger with an explicit ``label_path`` uses it verbatim."""
        service.set_active_tagger(
            ActiveTagger(
                kind="local",
                model_path="/models/animetimm/model.onnx",
                label_path="/models/animetimm/selected_tags.csv",
                preproc_profile="timm",
                default_size=512,
            )
        )

        kwargs, model_path = service._subprocess_spawn_args()

        assert model_path == "/models/animetimm/model.onnx"
        assert kwargs["label_path"] == "/models/animetimm/selected_tags.csv"
        assert kwargs["preproc_profile"] == TIMM_PROFILE
        assert kwargs["default_size"] == 512

    def test_local_active_tagger_without_label_path_falls_back_to_auto_discovery(self, service: TaggingService, tmp_path: Path):
        """When ``label_path`` is empty, the auto-discovery helper locates
        ``<model_dir>/selected_tags.csv`` (matching the legacy fallback)."""
        labels_csv = tmp_path / "selected_tags.csv"
        labels_csv.write_text("name,category\n1girl,0\n")

        service.set_active_tagger(
            ActiveTagger(
                kind="local",
                model_path=str(tmp_path / "model.onnx"),
                # label_path intentionally left blank
                preproc_profile="timm",
            )
        )

        kwargs, _model_path = service._subprocess_spawn_args()
        assert kwargs["label_path"] == str(labels_csv)

    def test_default_size_zero_does_not_set_kwarg(self, service: TaggingService):
        """A ``default_size`` of 0 mirrors the legacy "let the profile decide" semantics
        and is not forwarded to the subprocess (matching the pre-Phase-2 behaviour)."""
        service.set_active_tagger(
            ActiveTagger(
                kind="hf",
                repo_id="SmilingWolf/wd-vit-tagger-v3",
                # preproc_profile default ('wd-tagger') + default_size=0
            )
        )
        kwargs, _ = service._subprocess_spawn_args()

        assert "default_size" not in kwargs

    def test_fallback_to_legacy_config_for_hf(self, test_configuration: Configuration, service: TaggingService):
        """No active_tagger + legacy HF config produces the historical HF kwargs."""
        test_configuration.tagger_repo_id = "SmilingWolf/wd-eva02-large-tagger-v3"
        test_configuration.tagger_model_path = ""
        test_configuration.tagger_preproc_profile = "wd-tagger"

        kwargs, model_path = service._subprocess_spawn_args()

        assert model_path == ""  # HF ignores model_path
        assert kwargs["repo_id"] == "SmilingWolf/wd-eva02-large-tagger-v3"
        assert kwargs["preproc_profile"] == WD_TAGGER_PROFILE

    def test_fallback_to_legacy_config_for_local(self, test_configuration: Configuration, service: TaggingService, tmp_path: Path):
        """No active_tagger + legacy local-path config uses the configured ``label_path``."""
        labels_csv = tmp_path / "selected_tags.csv"
        labels_csv.write_text("name,category\n1girl,0\n")
        test_configuration.tagger_repo_id = ""
        test_configuration.tagger_model_path = str(tmp_path / "model.onnx")
        test_configuration.tagger_label_path = str(labels_csv)

        kwargs, model_path = service._subprocess_spawn_args()

        assert model_path == str(tmp_path / "model.onnx")
        assert kwargs["label_path"] == str(labels_csv)

    def test_unknown_preproc_profile_in_active_tagger_raises_at_spawn(self, service: TaggingService):
        """A typo in the persisted selection surfaces as ``RuntimeError`` at spawn,
        so the user gets a clear error rather than a silent wd-tagger fallback.

        ``ActiveTagger`` doesn't enforce profile validity at
        construction (the name lives in user-editable settings); the
        rejection happens in :func:`get_profile` when the subprocess is
        about to spawn.
        """
        service.set_active_tagger(
            ActiveTagger(
                kind="hf",
                repo_id="SmilingWolf/wd-vit-tagger-v3",
                preproc_profile="not-a-real-profile",
            )
        )
        with pytest.raises(RuntimeError, match="tagger_preproc_profile"):
            service._subprocess_spawn_args()

    def test_unknown_preproc_profile_in_fallback_raises(self, test_configuration: Configuration, service: TaggingService):
        """Same protection for the legacy Configuration path."""
        test_configuration.tagger_repo_id = "SmilingWolf/wd-vit-tagger-v3"
        test_configuration.tagger_preproc_profile = "not-a-real-profile"

        with pytest.raises(RuntimeError, match="tagger_preproc_profile"):
            service._subprocess_spawn_args()


# ---------------------------------------------------------------------------
# _tag_result_key
# ---------------------------------------------------------------------------


class TestTagResultKeyUsesActiveTagger:
    def test_active_tagger_source_label_becomes_model_id(self, service: TaggingService):
        """The cache key's ``model_id`` mirrors the SSE event's ``source`` label, so
        swapping the model on the same image produces a different bucket."""
        from yadc.api.services.tagging import TaggingThresholds, bucket_threshold

        service.set_active_tagger(
            ActiveTagger(
                kind="hf",
                repo_id="SmilingWolf/wd-vit-tagger-v3",
            )
        )
        thresholds = TaggingThresholds(
            rating=0.0,
            general=bucket_threshold(0.35),
            character=bucket_threshold(0.85),
        )

        key = service._tag_result_key("ds", 1, thresholds)

        assert key.model_id == "hf:SmilingWolf/wd-vit-tagger-v3"

    def test_no_active_tagger_falls_back_to_legacy_model_id(self, test_configuration: Configuration, service: TaggingService):
        """Legacy setups continue to use the historical ``repo_id|model_path`` model_id."""
        from yadc.api.services.tagging import TaggingThresholds, bucket_threshold

        test_configuration.tagger_model_path = "/fake/model.onnx"
        test_configuration.tagger_repo_id = ""
        thresholds = TaggingThresholds(
            rating=0.0,
            general=bucket_threshold(0.35),
            character=bucket_threshold(0.85),
        )
        key = service._tag_result_key("ds", 1, thresholds)

        # Legacy derived from Configuration fallback.
        assert key.model_id == "/fake/model.onnx"

    def test_swapping_active_tagger_invalidates_cache(self, service: TaggingService):
        """Different active selections hash to different model_ids, so cached results
        from the old model can't be served after a swap."""
        from yadc.api.services.tagging import TaggingThresholds, bucket_threshold

        thresholds = TaggingThresholds(
            rating=0.0,
            general=bucket_threshold(0.35),
            character=bucket_threshold(0.85),
        )
        service.set_active_tagger(
            ActiveTagger(
                kind="hf",
                repo_id="SmilingWolf/wd-vit-tagger-v3",
            )
        )
        key_a = service._tag_result_key("ds", 1, thresholds)

        service.set_active_tagger(
            ActiveTagger(
                kind="hf",
                repo_id="SmilingWolf/wd-eva02-large-tagger-v3",
            )
        )
        key_b = service._tag_result_key("ds", 1, thresholds)

        assert key_a.model_id != key_b.model_id
        assert hash(key_a) != hash(key_b)


# ---------------------------------------------------------------------------
# is_configured
# ---------------------------------------------------------------------------


class TestIsConfigured:
    def test_active_tagger_alone_is_configured(self, service: TaggingService):
        """An active_tagger means a swap has happened, which is sufficient."""
        service.set_active_tagger(
            ActiveTagger(
                kind="hf",
                repo_id="SmilingWolf/wd-vit-tagger-v3",
            )
        )
        assert service.is_configured is True

    def test_configured_via_legacy_config_without_active_tagger(self, test_configuration: Configuration, service: TaggingService):
        """Legacy setups are configured when the flat fields have a model."""
        test_configuration.tagger_repo_id = "SmilingWolf/wd-vit-tagger-v3"
        assert service.is_configured is True

    def test_not_configured_when_both_empty(self, test_configuration: Configuration, service: TaggingService):
        """A brand-new install with neither active_tagger nor flat fields is unconfigured.

        The shared ``test_configuration`` fixture now sets all
        tagger fields to empty defaults.
        """
        test_configuration.tagger_repo_id = ""
        test_configuration.tagger_model_path = ""
        assert service.is_configured is False
