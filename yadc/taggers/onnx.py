"""ONNX Runtime tagger — runs image classification via ONNX models.

Supports GPU (CUDA) when available, falls back to CPU. The default
label format is the SmilingWolf / WD ``selected_tags.csv`` (with a
``category`` column: 9 = rating, 0 = general, 4 = character) — that
matches the well-known wd-tagger family (``wd-v1-4-*``,
``wd-swinv2-tagger-v3``, etc.). A plain one-label-per-line ``.txt``
file is also supported for flat (uncategorized) models.

When ``repo_id`` is set, the model and label file are downloaded from
HuggingFace Hub via :func:`huggingface_hub.hf_hub_download` on
``load_model()``. The download happens in the worker process so the
main API process doesn't need network access.

The preprocessing pipeline (canvas → pad → resize → normalize → layout)
lives in :mod:`yadc.taggers.onnx_preprocess` and is selectable via a
:class:`PreprocProfile`. The default profile
(:data:`WD_TAGGER_PROFILE`) matches the wd-tagger convention
(NHWC + BGR, /255 baked into the graph). For PyTorch / timm exports
(e.g. ``animetimm/convnextv2``), pass :data:`TIMM_PROFILE` (NCHW +
RGB + ImageNet normalization) instead.
"""

from __future__ import annotations

import csv
import io
import logging
from pathlib import Path
from typing import Any

import numpy as np
from typing_extensions import override

from yadc.taggers.base import Tagger, TaggerResult
from yadc.taggers.onnx_preprocess import (
    TIMM_PROFILE,
    WD_TAGGER_PROFILE,
    Layout,
    PreprocProfile,
    log_profile_info,
    prepare_image,
    resolve_input,
)

BytesIO = io.BytesIO

logger = logging.getLogger(__name__)

# SmilingWolf / WD selected_tags.csv category codes.
# See https://huggingface.co/SmilingWolf/wd-v1-4-vit-tagger-v2/blob/main/selected_tags.csv
CATEGORY_RATING = 9
CATEGORY_GENERAL = 0
CATEGORY_CHARACTER = 4

# Default names used for the three SmilingWolf categories, in the order
# we want them emitted in the result.
DEFAULT_CATEGORY_NAMES: dict[int, str] = {
    CATEGORY_RATING: "rating",
    CATEGORY_GENERAL: "general",
    CATEGORY_CHARACTER: "character",
}


def _load_labels_csv(path: Path) -> tuple[list[str], dict[str, list[str]]]:
    """Load a SmilingWolf-style ``selected_tags.csv`` and split it into categories.

    Returns ``(all_tag_names, categories)`` where ``all_tag_names`` is
    the ordered list of every tag and ``categories`` maps the category
    name (``"rating"`` / ``"general"`` / ``"character"``) to its subset.
    Tags whose category code isn't recognized are dropped from
    ``categories`` but still included in ``all_tag_names``.
    """
    names: list[str] = []
    by_category: dict[int, list[str]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("name")
            if not name:
                continue
            names.append(name)
            cat_raw = row.get("category")
            if cat_raw is None:
                continue
            try:
                cat = int(cat_raw)
            except ValueError:
                continue
            by_category.setdefault(cat, []).append(name)
    categories: dict[str, list[str]] = {}
    for code, names_in_cat in by_category.items():
        cat_name = DEFAULT_CATEGORY_NAMES.get(code)
        if cat_name is not None:
            categories[cat_name] = names_in_cat
    return names, categories


def _load_labels_txt(path: Path) -> tuple[list[str], dict[str, list[str]]]:
    """Load a flat one-label-per-line text file. No categorization."""
    text = path.read_text(encoding="utf-8")
    names = [line for line in text.splitlines() if line]
    return names, {}


def load_labels(path: str | Path) -> tuple[list[str], dict[str, list[str]]]:
    """Load tag labels from a file.

    Dispatches by extension: ``.csv`` → SmilingWolf format, anything
    else (``.txt`` typically) → flat one-label-per-line.
    """
    p = Path(path)
    if p.suffix.lower() == ".csv":
        return _load_labels_csv(p)
    return _load_labels_txt(p)


# Re-export so external callers that already imported from this module
# (``from yadc.taggers.onnx import TIMM_PROFILE``) keep working.
__all__ = [
    "CATEGORY_CHARACTER",
    "CATEGORY_GENERAL",
    "CATEGORY_RATING",
    "DEFAULT_CATEGORY_NAMES",
    "OnnxTagger",
    "PreprocProfile",
    "TIMM_PROFILE",
    "WD_TAGGER_PROFILE",
    "apply_thresholds",
    "load_labels",
]


class OnnxTagger(Tagger):
    """Image tagger backed by an ONNX Runtime model.

    Preprocessing is delegated to :mod:`yadc.taggers.onnx_preprocess`
    via a :class:`PreprocProfile`. The default
    (:data:`WD_TAGGER_PROFILE`) matches the wd-tagger convention
    (NHWC + BGR, /255 baked into the graph). Use
    :data:`TIMM_PROFILE` for PyTorch / timm exports whose graphs run
    on raw NCHW RGB ImageNet-normalized input (animetimm ConvNeXt
    and friends).

    ``load_model`` logs the resolved preprocessing contract so the
    user can verify the pipeline before the first inference.

    Args:
        label_path: Local path to the labels file. Ignored when
            ``repo_id`` is set.
        repo_id: HuggingFace repo to download the model + labels
            from. When set, the model + label files are fetched on
            ``load_model`` and the ``model_path`` argument to
            ``load_model`` is ignored.
        repo_model_filename: Filename within the HF repo for the model.
        repo_label_filename: Filename within the HF repo for the
            labels.
        repo_sidecar_filenames: Extra files in the HF repo to fetch
            alongside the model (best-effort). Typical case: the ONNX
            external-data file (``model.onnx_data``) for models that
            exceed protobuf's 2 GB size limit. Each download is
            wrapped in a try/except for ``EntryNotFoundError`` — a
            missing sidecar just means the model loads as a single
            file.
        preproc_profile: The preprocessing profile to use. ``None`` →
            :data:`WD_TAGGER_PROFILE`.
        default_size: Override ``profile.default_input_size`` when
            the model's input shape has symbolic H/W dims. ``0`` →
            inherit the profile's default.
    """

    def __init__(
        self,
        label_path: str | Path | None = None,
        repo_id: str | None = None,
        repo_model_filename: str = "model.onnx",
        repo_label_filename: str = "selected_tags.csv",
        repo_sidecar_filenames: list[str] | None = None,
        preproc_profile: PreprocProfile | None = None,
        default_size: int = 0,
    ) -> None:
        self._label_path: str | Path | None = label_path
        # HuggingFace Hub download configuration. When ``repo_id`` is
        # set, ``load_model()`` downloads the model + label from the
        # repo and ignores the ``model_path`` argument.
        self._repo_id: str | None = repo_id or None
        self._repo_model_filename: str = repo_model_filename
        self._repo_label_filename: str = repo_label_filename
        self._repo_sidecar_filenames: list[str] = list(repo_sidecar_filenames or [])
        # Preprocessing profile — applied in ``predict()`` to the
        # input image. The default matches the wd-tagger convention;
        # TIMM_PROFILE fits PyTorch / timm exports that don't bake
        # preprocessing into the graph.
        self._profile: PreprocProfile = preproc_profile or WD_TAGGER_PROFILE
        # If a positive ``default_size`` was given, build a copy of the
        # profile with the size override so callers can pass e.g.
        # ``TIMM_PROFILE`` plus an explicit size without mutating the
        # shared module-level constant.
        if default_size > 0:
            self._profile = self._profile._replace(default_input_size=default_size)
        self._session: Any = None
        self._labels: list[str] = []
        self._categories: dict[str, list[str]] = {}
        # Resolved input height/width and layout — populated by
        # ``load_model`` once the session is built, then read by every
        # ``predict()`` call. Keeping these on the instance means the
        # (slow) shape inspection runs once instead of per image.
        self._input_height: int = 0
        self._input_width: int = 0
        self._layout: Layout = "nhwc"  # overwritten by load_model
        self._input_name: str = ""
        self._output_name: str = ""

    # ---- lifecycle --------------------------------------------------------

    @override
    def load_model(self, model_path: str, **kwargs: Any) -> None:  # noqa: ANN401
        # If a HuggingFace repo is configured, download the model and
        # label file from it and use the cached paths. Done before
        # touching onnxruntime so a download failure surfaces a
        # clear network/HF error rather than an opaque ORT error.
        if self._repo_id:
            from huggingface_hub import hf_hub_download
            from huggingface_hub.errors import EntryNotFoundError

            downloaded_model = hf_hub_download(
                repo_id=self._repo_id,
                filename=self._repo_model_filename,
            )
            downloaded_label = hf_hub_download(
                repo_id=self._repo_id,
                filename=self._repo_label_filename,
            )
            model_path = downloaded_model
            self._labels, self._categories = load_labels(downloaded_label)
            # Sidecar downloads — typically the ONNX external-data file
            # for models that exceed protobuf's 2 GB size limit. Each
            # one is best-effort: ``EntryNotFoundError`` just means the
            # repo doesn't carry that file, which is fine (most models
            # don't need any). Onnxruntime finds a sibling
            # ``<model>.onnx_data`` next to the ``.onnx`` automatically,
            # so downloading it into the same cache directory is all we
            # need. Logged at warning so operators can see which sidecars
            # were requested but not present.
            for sidecar in self._repo_sidecar_filenames:
                try:
                    hf_hub_download(repo_id=self._repo_id, filename=sidecar)
                except EntryNotFoundError:
                    logger.warning(
                        "Tagger sidecar not present in HF repo. [repo=%s, sidecar=%s]",
                        self._repo_id,
                        sidecar,
                    )

        # Build the ONNX session (extracted into a helper so tests
        # can mock just this step). ``onnxruntime`` is imported lazily
        # here so the main API process — which imports this module to
        # reference :class:`OnnxTagger` and ``apply_thresholds`` but
        # never builds a session itself — doesn't pay the import cost
        # (and the dependency) until the worker process actually needs it.
        self._create_session(model_path, **kwargs)

        # Load labels from a local file (only when we didn't already
        # load them from a HF download above).
        if self._repo_id is None and self._label_path is not None:
            self._labels, self._categories = load_labels(self._label_path)

        # Resolve and cache the preprocessing contract once so
        # ``predict()`` doesn't re-parse the shape on every image.
        input_meta = self._session.get_inputs()[0]
        self._input_height, self._input_width, self._layout = resolve_input(input_meta.shape, self._profile)
        self._input_name = input_meta.name
        self._output_name = self._session.get_outputs()[0].name

        log_profile_info(self._profile, self._input_height, self._input_width, self._layout)

    def _create_session(self, model_path: str, **kwargs: Any) -> None:  # noqa: ANN401
        """Build the ONNX InferenceSession. Extracted so tests can mock the ORT call.

        ``onnxruntime`` is imported lazily so the module can be imported
        in the main API process (e.g. to reference :class:`OnnxTagger`
        or ``apply_thresholds``) without paying the dependency cost;
        only the worker that actually runs inference needs the package.
        """
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError("onnxruntime is required for OnnxTagger (pip install onnxruntime-gpu)") from exc

        # Determine execution provider — GPU preferred.
        providers: list[str] = []
        if "CUDAExecutionProvider" in ort.get_available_providers():
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = kwargs.get("intra_op_num_threads", 1)
        session_options.inter_op_num_threads = kwargs.get("inter_op_num_threads", 1)

        self._session = ort.InferenceSession(
            model_path,
            sess_options=session_options,
            providers=providers,
        )

    @override
    def unload_model(self) -> None:
        self._session = None
        self._labels = []
        self._categories = {}
        self._input_height = 0
        self._input_width = 0

    # ---- prediction -------------------------------------------------------

    @override
    def predict(self, image_bytes: bytes) -> TaggerResult:
        if self._session is None:
            raise RuntimeError("Model not loaded — call load_model() first")

        tensor = prepare_image(
            image_bytes,
            self._input_height,
            self._input_width,
            self._layout,
            self._profile,
        )
        logger.debug(
            "Tagger preprocessed tensor: shape=%s dtype=%s min=%.4f max=%.4f mean=%.4f",
            tuple(tensor.shape),
            tensor.dtype,
            float(tensor.min()),
            float(tensor.max()),
            float(tensor.mean()),
        )
        preds = self._session.run([self._output_name], {self._input_name: tensor})[0]
        scores = np.asarray(preds[0], dtype=np.float32)
        if self._profile.apply_sigmoid:
            scores = 1.0 / (1.0 + np.exp(-scores))

        return self._build_result(scores)

    # ---- helpers ----------------------------------------------------------

    def _build_result(self, scores: np.ndarray) -> TaggerResult:
        """Map raw sigmoid/score outputs to a :class:`TaggerResult`.

        Sigmoid isn't applied here because wd-tagger models already
        return post-sigmoid probabilities. If a future model returns
        logits, the caller (or a model-specific subclass) should
        normalize them before reaching this method.
        """
        if self._labels and scores.ndim == 1 and len(self._labels) == scores.shape[0]:
            tags = {name: float(scores[i]) for i, name in enumerate(self._labels)}
            return TaggerResult(tags=tags, categories=dict(self._categories))
        # No labels, or the label count doesn't match the score count —
        # synthesize integer-index tags and treat them as uncategorized
        # (empty categories) rather than carrying stale category lists.
        tags = {str(i): float(scores[i]) for i in range(scores.shape[0])}
        return TaggerResult(tags=tags, categories={})


def apply_thresholds(
    result: TaggerResult,
    *,
    rating_threshold: float = 0.0,
    general_threshold: float = 0.35,
    character_threshold: float = 0.85,
    per_tag_thresholds: dict[str, float] | None = None,
) -> TaggerResult:
    """Return a new :class:`TaggerResult` with low-score tags dropped per category.

    Tags not assigned to any category are kept as-is. Tags in a
    category with no threshold entry are kept as-is. Set a threshold
    to ``0.0`` to keep all tags in that category.
    """
    thresholds = {
        "rating": rating_threshold,
        "general": general_threshold,
        "character": character_threshold,
    }
    if not result.categories:
        return result

    drop: set[str] = set()
    for cat_name, cat_tags in result.categories.items():
        cat_thr = thresholds.get(cat_name, 0.0)
        for tag in cat_tags:
            thr = 0.0
            if per_tag_thresholds and tag in per_tag_thresholds:
                thr = per_tag_thresholds[tag]
            elif cat_thr > 0:
                thr = cat_thr
            if thr <= 0:
                continue
            score = result.tags.get(tag, 0.0)
            if score < thr:
                drop.add(tag)

    if not drop:
        return result

    filtered_tags = {k: v for k, v in result.tags.items() if k not in drop}
    filtered_categories = {cat: [t for t in tags if t not in drop] for cat, tags in result.categories.items()}
    return TaggerResult(tags=filtered_tags, categories=filtered_categories)
