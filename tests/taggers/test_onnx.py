"""Tests for ``OnnxTagger`` — including the HuggingFace Hub download path.

Download behavior is exercised by mocking ``huggingface_hub.hf_hub_download``
so the tests don't need network access.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from yadc.taggers.onnx import OnnxTagger, load_labels
from yadc.taggers.onnx_preprocess import TIMM_PROFILE, WD_TAGGER_PROFILE, PreprocProfile

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@dataclass
class _FakeInputMeta:
    """Minimal stand-in for ``onnxruntime.InferenceSession`` inputs."""

    name: str
    shape: list[Any]


@dataclass
class _FakeOutputMeta:
    """Minimal stand-in for ``onnxruntime.InferenceSession`` outputs."""

    name: str


class _FakeSession:
    """Session double with ``get_inputs`` / ``get_outputs`` / ``run`` matching ORT's API surface."""

    def __init__(self, input_meta: _FakeInputMeta, output_name: str = "logits") -> None:
        self._inputs = [input_meta]
        self._outputs = [_FakeOutputMeta(name=output_name)]
        self.run_calls: list[tuple[list[str], dict[str, Any]]] = []

    def get_inputs(self) -> list[_FakeInputMeta]:
        return self._inputs

    def get_outputs(self) -> list[_FakeOutputMeta]:
        return self._outputs

    def run(self, output_names: list[str], inputs: dict[str, Any]) -> list[Any]:
        # Record the call (for assertions) and return a (1, 3) zeros
        # array — enough for any test fixture.
        self.run_calls.append((list(output_names), dict(inputs)))
        import numpy as np

        return [np.zeros((1, 3), dtype=np.float32)]


class _SessionPatch:
    """Context manager that patches ``OnnxTagger._create_session`` with a fake.

    Use ``with patch_session(...) as fake:`` to bind the fake session for
    post-load assertions, or ``with patch_session(...):`` if you only need
    the side effect.
    """

    def __init__(self, tagger: OnnxTagger, input_shape: list[Any], input_name: str, output_name: str) -> None:
        self._tagger = tagger
        self._fake = _FakeSession(_FakeInputMeta(name=input_name, shape=input_shape), output_name)
        self._cm: Any = None

    def __enter__(self) -> _FakeSession:
        def _install(model_path: str, **kwargs: Any) -> None:  # noqa: ANN401
            self._tagger._session = self._fake  # type: ignore[assignment]

        self._cm = patch.object(self._tagger, "_create_session", side_effect=_install)
        self._cm.__enter__()
        return self._fake

    def __exit__(self, *exc: Any) -> None:
        if self._cm is not None:
            self._cm.__exit__(*exc)


def patch_session(
    tagger: OnnxTagger,
    *,
    input_shape: list[Any],
    input_name: str = "image",
    output_name: str = "logits",
) -> _SessionPatch:
    """Patch ``_create_session`` on ``tagger`` to install a fake ONNX session.

    Returns a :class:`_SessionPatch` context manager. Use either ``with
    patch_session(...):`` or ``with patch_session(...) as fake:`` depending
    on whether you need to assert against the installed fake session.
    """
    return _SessionPatch(tagger, input_shape, input_name, output_name)


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
            patch_session(tagger, input_shape=["batch", 448, 448, 3]),
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
            patch_session(tagger, input_shape=["batch", 448, 448, 3]),
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
            patch_session(tagger, input_shape=["batch", 448, 448, 3]) as fake_sess,
        ):
            tagger.load_model("/this/path/should/be/ignored.onnx")

        # The local path must not leak into the download call, and the
        # session must be created from the downloaded path instead.
        for call in mock_dl.call_args_list:
            assert "/this/path" not in str(call)
        assert tagger._session is fake_sess
        assert tagger._input_height == 448 and tagger._input_width == 448 and tagger._layout == "nhwc"

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

        with patch("huggingface_hub.hf_hub_download") as mock_dl, patch_session(tagger, input_shape=["batch", 448, 448, 3]):
            tagger.load_model(str(fake_model_file))

        mock_dl.assert_not_called()
        assert tagger._session is not None
        assert tagger._categories["general"] == ["1girl", "solo"]
        assert tagger._input_height == 448 and tagger._input_width == 448 and tagger._layout == "nhwc"


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


# ---------------------------------------------------------------------------
# PreprocProfile — named profiles and lookup
# ---------------------------------------------------------------------------


class TestPreprocProfile:
    def test_wd_tagger_profile_defaults(self) -> None:
        """WD_TAGGER_PROFILE matches the SmilingWolf convention used by historical code."""
        assert WD_TAGGER_PROFILE.channel_order == "bgr"
        assert WD_TAGGER_PROFILE.normalize == "none"
        assert WD_TAGGER_PROFILE.default_input_size == 448
        assert WD_TAGGER_PROFILE.apply_sigmoid is False

    def test_timm_profile_defaults(self) -> None:
        """TIMM_PROFILE matches the standard PyTorch / timm ImageNet preprocessing (animetimm convnext)."""
        assert TIMM_PROFILE.channel_order == "rgb"
        assert TIMM_PROFILE.normalize == "imagenet"
        assert TIMM_PROFILE.default_input_size == 512
        assert TIMM_PROFILE.apply_sigmoid is True

    def test_get_profile_resolves_canonical_and_aliases(self) -> None:
        """Both canonical names and aliases resolve; unknown names raise ``ValueError``."""
        from yadc.taggers.onnx_preprocess import get_profile

        assert get_profile("wd-tagger") is WD_TAGGER_PROFILE
        assert get_profile("wdtagger") is WD_TAGGER_PROFILE
        assert get_profile("timm") is TIMM_PROFILE
        assert get_profile("pytorch") is TIMM_PROFILE
        assert get_profile("wd-tagger  ") is WD_TAGGER_PROFILE  # whitespace-tolerant

        with pytest.raises(ValueError, match="Unknown preproc profile"):
            get_profile("nope")

    def test_list_profiles_is_canonical(self) -> None:
        """``list_profiles`` returns the canonical names (deduplicated)."""
        from yadc.taggers.onnx_preprocess import list_profiles

        names = list_profiles()
        assert names == ["wd-tagger", "timm"]

    def test_constructor_default_is_wd_tagger(self) -> None:
        """Constructing :class:`OnnxTagger` without arguments gives the wd-tagger profile."""
        tagger = OnnxTagger()
        assert tagger._profile is WD_TAGGER_PROFILE

    def test_constructor_default_size_overrides_profile(self) -> None:
        """``default_size`` overrides the profile's ``default_input_size`` (without mutating it)."""
        tagger = OnnxTagger(preproc_profile=TIMM_PROFILE, default_size=512)
        assert tagger._profile.default_input_size == 512
        # Shared module-level profile is untouched.
        assert TIMM_PROFILE.default_input_size == 512

        override = OnnxTagger(preproc_profile=TIMM_PROFILE, default_size=384)
        assert override._profile.default_input_size == 384
        assert TIMM_PROFILE.default_input_size == 512


# ---------------------------------------------------------------------------
# onnx_preprocess.resolve_input — layout detection
# ---------------------------------------------------------------------------


class TestResolveInput:
    def test_nchw_with_concrete_dims(self) -> None:
        """An input shape with channels in dim 1 + concrete H/W resolves to NCHW + (H, W)."""
        from yadc.taggers.onnx_preprocess import resolve_input

        h, w, layout = resolve_input(["batch", 3, 512, 512], TIMM_PROFILE)
        assert (h, w, layout) == (512, 512, "nchw")

    def test_nhwc_with_concrete_dims(self) -> None:
        """An input shape with channels in dim 3 + concrete H/W resolves to NHWC + (H, W)."""
        from yadc.taggers.onnx_preprocess import resolve_input

        h, w, layout = resolve_input(["batch", 448, 448, 3], WD_TAGGER_PROFILE)
        assert (h, w, layout) == (448, 448, "nhwc")

    def test_dynamic_nchw_falls_back_to_profile_size(self) -> None:
        """Dynamic NCHW shape with symbolic H/W uses the profile's ``default_input_size`` and keeps NCHW layout."""
        from yadc.taggers.onnx_preprocess import resolve_input

        h, w, layout = resolve_input(["batch", 3, "H", "W"], TIMM_PROFILE)
        assert (h, w, layout) == (512, 512, "nchw")

    def test_dynamic_nhwc_falls_back_to_profile_size(self) -> None:
        """Dynamic NHWC shape with symbolic H/W uses the profile's ``default_input_size`` and keeps NHWC layout."""
        from yadc.taggers.onnx_preprocess import resolve_input

        h, w, layout = resolve_input(["batch", "H", "W", 3], WD_TAGGER_PROFILE)
        assert (h, w, layout) == (448, 448, "nhwc")

    def test_unknown_shape_with_no_fallback_raises(self) -> None:
        """A shape that can't be parsed + ``default_input_size=0`` raises ``ValueError``."""
        from yadc.taggers.onnx_preprocess import resolve_input

        empty = PreprocProfile(default_input_size=0)
        # No shape at all → can't determine layout.
        with pytest.raises(ValueError, match="Could not detect"):
            resolve_input(None, empty)
        # Rank != 4 → also can't determine layout.
        with pytest.raises(ValueError, match="Could not detect"):
            resolve_input([3], empty)
        # Rank-4 but no channel signal anywhere → can't determine layout.
        with pytest.raises(ValueError, match="Could not detect"):
            resolve_input(["a", "b", "c", "d"], empty)

    def test_fully_symbolic_nchw_via_dim_names(self) -> None:
        """A fully-symbolic shape whose dim names hint at the layout resolves correctly (animetimm ConvNeXt case)."""
        from yadc.taggers.onnx_preprocess import resolve_input

        # Real animetimm ConvNeXt export: ['batch_size', 'num_channels', 'height', 'width']
        h, w, layout = resolve_input(["batch_size", "num_channels", "height", "width"], TIMM_PROFILE)
        assert (h, w, layout) == (512, 512, "nchw")

    def test_concrete_h_w_with_symbolic_channel(self) -> None:
        """A shape with concrete H/W + symbolic channels still resolves to the right layout via the dim name."""
        from yadc.taggers.onnx_preprocess import resolve_input

        h, w, layout = resolve_input(["batch_size", "num_channels", 512, 512], TIMM_PROFILE)
        assert (h, w, layout) == (512, 512, "nchw")


# ---------------------------------------------------------------------------
# onnx_preprocess.prepare_image — per-profile output
# ---------------------------------------------------------------------------


class TestPrepareImage:
    def _solid_color_png(self, color: tuple[int, int, int], size: tuple[int, int]) -> bytes:
        from PIL import Image

        img = Image.new("RGB", size, color=color)
        import io as _io

        buf = _io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def test_wd_tagger_outputs_nhwc_bgr_no_normalize(self) -> None:
        """The wd-tagger profile produces an NHWC float32 tensor with channel-flipped BGR + raw [0, 255] values."""
        import numpy as np

        from yadc.taggers.onnx_preprocess import prepare_image

        img_bytes = self._solid_color_png((100, 150, 200), (448, 448))
        out = prepare_image(img_bytes, 448, 448, "nhwc", WD_TAGGER_PROFILE)

        assert out.shape == (1, 448, 448, 3)
        assert out.dtype == np.float32
        # Channel order check: original is RGB(100, 150, 200), BGR flip -> (200, 150, 100).
        assert out[0, 100, 100, :].tolist() == [pytest.approx(200), pytest.approx(150), pytest.approx(100)]
        # No normalization: raw [0, 255] range.
        assert out.min() == 100.0
        assert out.max() == 200.0

    def test_timm_outputs_nchw_rgb_imagenet_normalized(self) -> None:
        """The timm profile produces an NCHW float32 tensor with RGB + ImageNet normalization."""
        import numpy as np

        from yadc.taggers.onnx_preprocess import prepare_image

        img_bytes = self._solid_color_png((100, 150, 200), (512, 512))
        out = prepare_image(img_bytes, 512, 512, "nchw", TIMM_PROFILE)

        assert out.shape == (1, 3, 512, 512)
        assert out.dtype == np.float32
        # Pixel lies in the body of the (square) image \u2014 no padding involved.
        # ImageNet normalization: (x/255 - mean) / std for each channel.
        expected = [
            (100 / 255 - 0.485) / 0.229,
            (150 / 255 - 0.456) / 0.224,
            (200 / 255 - 0.406) / 0.225,
        ]
        actual = out[0, :, 100, 100].tolist()
        assert actual == [pytest.approx(e, abs=1e-5) for e in expected]

    def test_timm_with_rgba_input_uses_white_canvas(self) -> None:
        """Transparent input gets composited on a white canvas before timm preprocessing runs."""
        import numpy as np
        from PIL import Image

        from yadc.taggers.onnx_preprocess import prepare_image

        rgba = Image.new("RGBA", (512, 512), color=(100, 150, 200, 128))
        import io as _io

        buf = _io.BytesIO()
        rgba.save(buf, format="PNG")
        out = prepare_image(buf.getvalue(), 512, 512, "nchw", TIMM_PROFILE)

        # The composited pixel isn't pure (100, 150, 200) because alpha=128
        # mixes with white; we only assert the shape + dtype and that the
        # output is in the expected normalization range.
        assert out.shape == (1, 3, 512, 512)
        assert out.dtype == np.float32
        # ImageNet normalization yields values in [-2.1, 2.6] roughly.
        assert float(out.min()) > -2.5 and float(out.max()) < 3.0

    def test_nonsquare_target_pads_with_white(self) -> None:
        """Non-square targets fit-then-pad with white (covers timm ``pad_to_size`` semantics)."""

        from yadc.taggers.onnx_preprocess import prepare_image

        # Wide image, taller target \u2014 scale to height, pad width.
        img_bytes = self._solid_color_png((50, 100, 150), (400, 200))
        wide_profile = PreprocProfile(channel_order="rgb", normalize="none", default_input_size=0)
        out = prepare_image(img_bytes, 224, 224, "nhwc", wide_profile)

        assert out.shape == (1, 224, 224, 3)
        # Top-left corner of a non-square image padded to 224x224 should be white.
        assert out[0, 0, 0, 0] == pytest.approx(255.0)
        assert out[0, 0, 0, 1] == pytest.approx(255.0)
        assert out[0, 0, 0, 2] == pytest.approx(255.0)

    def test_padding_initially_white(self) -> None:
        """The pad canvas is initialized to white before paste \u2014 non-square inputs get padded with white, not black."""
        from yadc.taggers.onnx_preprocess import prepare_image

        # Wide-aspect image padded to a square target forces the pad step.
        img_bytes = self._solid_color_png((0, 0, 0), (400, 100))
        out = prepare_image(img_bytes, 200, 200, "nhwc", WD_TAGGER_PROFILE)
        # Image is scaled to height=200, then padded horizontally to width=200.
        # Pad has 100px on each side (after centering). Top corners are pure padding -> white.
        assert out[0, 0, 0, :].tolist() == [pytest.approx(255.0)] * 3
        assert out[0, 0, 199, :].tolist() == [pytest.approx(255.0)] * 3
        # Middle-left pixel is in the image body \u2014 should be black (after BGR flip).
        assert out[0, 100, 0, 0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# OnnxTagger integration with profiles
# ---------------------------------------------------------------------------


class TestOnnxTaggerWithProfiles:
    def test_timm_profile_resolves_nchw_layout(self, fake_model_file: Path, fake_labels_csv: Path) -> None:
        """Loading a TIMM-profile tagger with an NCHW dynamic-shape model resolves the right contract."""
        tagger = OnnxTagger(preproc_profile=TIMM_PROFILE)
        with (
            patch(
                "huggingface_hub.hf_hub_download",
                side_effect=[str(fake_model_file), str(fake_labels_csv)],
            ),
            patch_session(tagger, input_shape=["batch", 3, "H", "W"]),
        ):
            tagger.load_model("/unused")

        assert tagger._input_height == 512
        assert tagger._input_width == 512
        assert tagger._layout == "nchw"
        # Shared TIMM_PROFILE wasn't overridden (no default_size was given),
        # so the tagger just references it directly.
        assert tagger._profile is TIMM_PROFILE
        assert tagger._profile.channel_order == "rgb"
        assert tagger._profile.normalize == "imagenet"

    def test_predict_uses_profile(self, fake_model_file: Path, fake_labels_csv: Path) -> None:
        """``predict`` forwards the resolved (height, width, layout, profile) to ``prepare_image``."""
        import io as _io

        from PIL import Image

        tagger = OnnxTagger(preproc_profile=TIMM_PROFILE)
        # Non-square source so the prep step actually pads (exercises white-canvas).
        img = Image.new("RGB", (300, 200), color=(50, 100, 150))
        buf = _io.BytesIO()
        img.save(buf, format="PNG")
        image_bytes = buf.getvalue()

        with (
            patch(
                "huggingface_hub.hf_hub_download",
                side_effect=[str(fake_model_file), str(fake_labels_csv)],
            ),
            patch_session(tagger, input_shape=["batch", 3, 512, 512]) as fake_sess,
        ):
            tagger.load_model("/unused")
            tagger.predict(image_bytes)

        # The session received the preprocessed tensor with the right name and shape.
        assert len(fake_sess.run_calls) == 1
        output_names, inputs = fake_sess.run_calls[0]
        assert output_names == ["logits"]
        # Input name comes from the fake session; key is "image".
        tensor = inputs["image"]
        assert tensor.shape == (1, 3, 512, 512)
        # ImageNet normalization + white pad: red of (50,100,150) ~= -1.26;
        # blue channel of white pad (1-0.406)/0.225 ~ 2.64.
        assert -1.3 < float(tensor.min()) < -1.2
        assert 2.6 < float(tensor.max()) < 2.7

    def test_predict_applies_sigmoid_when_profile_says_so(self, fake_model_file: Path, fake_labels_csv: Path) -> None:
        """When ``profile.apply_sigmoid`` is True, ``predict`` sigmoid-converts the raw model output."""
        import io as _io

        import numpy as np
        from PIL import Image

        tagger = OnnxTagger(preproc_profile=TIMM_PROFILE)
        img = Image.new("RGB", (100, 100), color=(50, 50, 50))
        buf = _io.BytesIO()
        img.save(buf, format="PNG")
        image_bytes = buf.getvalue()

        with (
            patch(
                "huggingface_hub.hf_hub_download",
                side_effect=[str(fake_model_file), str(fake_labels_csv)],
            ),
            patch_session(tagger, input_shape=["batch_size", "num_channels", "height", "width"]) as fake_sess,
        ):
            tagger.load_model("/unused")
            # Override the fake session's run to return logits.
            raw_scores = np.array([-3.0, 0.0, 2.0, 5.0, -1.0, 1.0, 3.0, -2.0, 4.0, 0.5], dtype=np.float32)
            fake_sess.run = lambda output_names, inputs: [raw_scores.reshape(1, -1)]  # type: ignore[method-assign]
            result = tagger.predict(image_bytes)

        # All tag scores must be in [0, 1] \u2014 sigmoid was applied (raw_scores had values up to 5.0).
        assert all(0.0 <= score <= 1.0 for score in result.tags.values())
        # Spot-check: sigmoid(0) = 0.5, sigmoid(2) ~ 0.881.
        scores = list(result.tags.values())
        assert pytest.approx(scores[1]) == 0.5
        assert pytest.approx(scores[2], abs=1e-4) == 1 / (1 + np.exp(-2.0))

    def test_predict_does_not_sigmoid_when_profile_off(self, fake_model_file: Path, fake_labels_csv: Path) -> None:
        """When ``profile.apply_sigmoid`` is False (WD-tagger), raw scores flow through unchanged."""
        import io as _io

        import numpy as np
        from PIL import Image

        tagger = OnnxTagger(preproc_profile=WD_TAGGER_PROFILE)
        img = Image.new("RGB", (100, 100), color=(50, 50, 50))
        buf = _io.BytesIO()
        img.save(buf, format="PNG")
        image_bytes = buf.getvalue()

        with (
            patch(
                "huggingface_hub.hf_hub_download",
                side_effect=[str(fake_model_file), str(fake_labels_csv)],
            ),
            patch_session(tagger, input_shape=["batch", 448, 448, 3]) as fake_sess,
        ):
            tagger.load_model("/unused")
            raw_scores = np.array([1.93, 1.28, 1.15, 0.67], dtype=np.float32)
            fake_sess.run = lambda output_names, inputs: [raw_scores.reshape(1, -1)]  # type: ignore[method-assign]
            result = tagger.predict(image_bytes)

        # No sigmoid applied: values stay > 1, matching raw_scores exactly.
        scores = list(result.tags.values())
        assert scores == [pytest.approx(1.93), pytest.approx(1.28), pytest.approx(1.15), pytest.approx(0.67)]  # type: ignore[list-item]  # exact float comparison


# ---------------------------------------------------------------------------
# OnnxTagger.load_model — HF sidecar downloads (best-effort)
# ---------------------------------------------------------------------------
#
# For models that exceed protobuf's 2 GB size limit, ONNX exports the
# weights to a separate ``model.onnx_data`` file alongside the main
# ``model.onnx``. ``OnnxTagger`` accepts a ``repo_sidecar_filenames``
# list and downloads each entry after the model + labels. Each
# download is wrapped in a try/except for ``EntryNotFoundError`` so
# a missing file just means the model loads as a single file.


class TestOnnxTaggerSidecars:
    def test_sidecars_downloaded_best_effort(self, fake_model_file: Path, fake_labels_csv: Path):
        """When ``repo_sidecar_filenames`` is set, each entry is fetched after
        the main model + label downloads. ``EntryNotFoundError`` on the
        sidecar is logged at warning but doesn't fail the load."""
        from huggingface_hub.errors import EntryNotFoundError

        tagger = OnnxTagger(
            repo_id="some/large-repo",
            repo_sidecar_filenames=["model.onnx_data"],
        )

        def _fake_download(repo_id, filename, **_):
            if filename == "model.onnx":
                return str(fake_model_file)
            if filename == "selected_tags.csv":
                return str(fake_labels_csv)
            if filename == "model.onnx_data":
                raise EntryNotFoundError("not in repo")
            raise AssertionError(f"unexpected download: {filename}")

        with (
            patch("huggingface_hub.hf_hub_download", side_effect=_fake_download) as mock_dl,
            patch_session(tagger, input_shape=["batch", 448, 448, 3]),
        ):
            tagger.load_model("/unused")

        filenames = [c.kwargs["filename"] for c in mock_dl.call_args_list]
        assert filenames == ["model.onnx", "selected_tags.csv", "model.onnx_data"]

    def test_no_sidecar_download_when_list_empty(self, fake_model_file: Path, fake_labels_csv: Path):
        """Default behavior — no sidecar downloads when ``repo_sidecar_filenames`` is empty/None."""
        tagger = OnnxTagger(repo_id="some/repo")

        with (
            patch(
                "huggingface_hub.hf_hub_download",
                side_effect=[str(fake_model_file), str(fake_labels_csv)],
            ) as mock_dl,
            patch_session(tagger, input_shape=["batch", 448, 448, 3]),
        ):
            tagger.load_model("/unused")

        filenames = sorted(c.kwargs["filename"] for c in mock_dl.call_args_list)
        assert filenames == ["model.onnx", "selected_tags.csv"]

    def test_sidecar_present_download_succeeds(self, fake_model_file: Path, fake_labels_csv: Path):
        """When the sidecar IS in the repo, it gets downloaded normally."""
        sidecar_file = fake_model_file.parent / "model.onnx_data"
        sidecar_file.write_bytes(b"external-data")

        tagger = OnnxTagger(
            repo_id="some/repo",
            repo_sidecar_filenames=["model.onnx_data"],
        )

        def _fake_download(repo_id, filename, **_):
            if filename == "model.onnx":
                return str(fake_model_file)
            if filename == "selected_tags.csv":
                return str(fake_labels_csv)
            return str(sidecar_file)

        with patch("huggingface_hub.hf_hub_download", side_effect=_fake_download) as mock_dl, patch_session(tagger, input_shape=["batch", 448, 448, 3]):
            tagger.load_model("/unused")

        filenames = sorted(c.kwargs["filename"] for c in mock_dl.call_args_list)
        assert filenames == ["model.onnx", "model.onnx_data", "selected_tags.csv"]


class TestLoadPerTagThresholds:
    def _write_csv(self, tmp_path: Path, header: list[str], rows: list[list[str]]) -> Path:
        path = tmp_path / "selected_tags.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
        return path

    def test_reads_supported_columns(self, tmp_path: Path):
        from yadc.taggers.onnx_preprocess import _load_per_tag_thresholds

        path = self._write_csv(
            tmp_path,
            ["name", "category", "best_threshold", "best_recall"],
            [["1girl", "0", "0.35", "0.5"], ["solo", "0", "0.4", "0.6"]],
        )
        result = _load_per_tag_thresholds(path)
        assert result is not None
        assert result["best_threshold"] == {"1girl": 0.35, "solo": 0.4}
        assert result["best_recall"] == {"1girl": 0.5, "solo": 0.6}

    def test_returns_none_without_threshold_columns(self, tmp_path: Path):
        from yadc.taggers.onnx_preprocess import _load_per_tag_thresholds

        path = self._write_csv(tmp_path, ["name", "category"], [["1girl", "0"]])
        assert _load_per_tag_thresholds(path) is None

    def test_skips_invalid_values(self, tmp_path: Path):
        from yadc.taggers.onnx_preprocess import _load_per_tag_thresholds

        path = self._write_csv(
            tmp_path,
            ["name", "category", "best_threshold"],
            [["good", "0", "0.5"], ["bad-float", "0", "abc"], ["bad-range", "0", "1.5"], ["", "0", "0.5"]],
        )
        result = _load_per_tag_thresholds(path)
        assert result is not None
        assert result["best_threshold"] == {"good": 0.5}

    def test_ignores_best_f1_column(self, tmp_path: Path):
        from yadc.taggers.onnx_preprocess import _load_per_tag_thresholds

        path = self._write_csv(tmp_path, ["name", "category", "best_f1"], [["1girl", "0", "0.7"]])
        assert _load_per_tag_thresholds(path) is None


class TestApplyThresholdsPerTag:
    def _result(self):
        from yadc.taggers.base import TaggerResult

        return TaggerResult(
            tags={"1girl": 0.4, "solo": 0.3},
            categories={"general": ["1girl", "solo"]},
        )

    def test_per_tag_overrides_category(self):
        from yadc.taggers.onnx import apply_thresholds

        out = apply_thresholds(
            self._result(),
            general_threshold=0.9,
            per_tag_thresholds={"1girl": 0.35},
        )
        assert "1girl" in out.tags
        assert "solo" not in out.tags

    def test_missing_per_tag_falls_back_to_category(self):
        from yadc.taggers.onnx import apply_thresholds

        out = apply_thresholds(
            self._result(),
            general_threshold=0.35,
            per_tag_thresholds={"1girl": 0.35},
        )
        assert "solo" not in out.tags

    def test_zero_per_tag_keeps_tag(self):
        from yadc.taggers.onnx import apply_thresholds

        out = apply_thresholds(
            self._result(),
            general_threshold=0.9,
            per_tag_thresholds={"solo": 0.0},
        )
        assert "solo" in out.tags
