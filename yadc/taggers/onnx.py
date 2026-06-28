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
"""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from typing_extensions import override

from yadc.taggers.base import Tagger, TaggerResult

BytesIO = io.BytesIO

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


class OnnxTagger(Tagger):
    """Image tagger backed by an ONNX Runtime model.

    The preprocessing pipeline matches the wd-tagger convention used by
    the SmilingWolf model family:

    1. White-canvas composite for RGBA / palette / transparency modes.
    2. Pad to a square with white.
    3. Resize to the model's expected input size (square, taken from
       the ONNX graph).
    4. Cast to ``float32`` (no explicit /255 — the model graph bakes
       in its own normalization).
    5. RGB → BGR channel flip (models are trained on BGR).

    GPU (CUDA) is used when available; otherwise the CPU provider is
    used.  ``intra_op_num_threads`` / ``inter_op_num_threads`` are
    exposed via ``load_model(..., intra_op_num_threads=N)``.
    """

    def __init__(
        self,
        label_path: str | Path | None = None,
        repo_id: str | None = None,
        repo_model_filename: str = "model.onnx",
        repo_label_filename: str = "selected_tags.csv",
    ) -> None:
        self._label_path: str | Path | None = label_path
        # HuggingFace Hub download configuration. When ``repo_id`` is
        # set, ``load_model()`` downloads the model + label from the
        # repo and ignores the ``model_path`` argument.
        self._repo_id: str | None = repo_id or None
        self._repo_model_filename: str = repo_model_filename
        self._repo_label_filename: str = repo_label_filename
        self._session: Any = None
        self._labels: list[str] = []
        self._categories: dict[str, list[str]] = {}

    # ---- lifecycle --------------------------------------------------------

    @override
    def load_model(self, model_path: str, **kwargs: Any) -> None:  # noqa: ANN401
        # If a HuggingFace repo is configured, download the model and
        # label file from it and use the cached paths. Done before
        # touching onnxruntime so a download failure surfaces a
        # clear network/HF error rather than an opaque ORT error.
        if self._repo_id:
            from huggingface_hub import hf_hub_download

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

    # ---- prediction -------------------------------------------------------

    @override
    def predict(self, image_bytes: bytes) -> TaggerResult:
        if self._session is None:
            raise RuntimeError("Model not loaded — call load_model() first")

        # ONNX input shape: NHWC for wd-tagger models (or NCHW). We use
        # the height/width from the graph, with a fallback to 448 if
        # the shape is symbolic.
        input_meta = self._session.get_inputs()[0]
        height, width = self._resolve_input_hw(input_meta.shape)
        if height <= 0 or width <= 0:
            # Fall back to a sane default for SmilingWolf models.
            height = width = 448

        image = self._prepare_image(image_bytes, height, width)
        input_name = input_meta.name
        output_name = self._session.get_outputs()[0].name
        preds = self._session.run([output_name], {input_name: image})[0]
        scores = np.asarray(preds[0], dtype=np.float32)

        return self._build_result(scores)

    # ---- helpers ----------------------------------------------------------

    @staticmethod
    def _resolve_input_hw(shape: Any) -> tuple[int, int]:
        """Resolve (height, width) from an ONNX input shape.

        wd-tagger models use NHWC shapes like ``['batch', 448, 448, 3]``;
        other backbones use NCHW ``['batch', 3, H, W]``. We accept both
        and return (h, w). Symbolic / None dims fall back to 0 so the
        caller can pick a default.
        """
        if shape is None or len(shape) < 3:
            return 0, 0
        # NHWC
        if len(shape) == 4 and shape[1] not in (None, "batch", "batch_size") and shape[3] in (1, 3, 4):
            try:
                return int(shape[1]), int(shape[2])
            except (TypeError, ValueError):
                return 0, 0
        # NCHW
        if len(shape) == 4 and shape[1] in (1, 3, 4):
            try:
                return int(shape[2]), int(shape[3])
            except (TypeError, ValueError):
                return 0, 0
        return 0, 0

    @staticmethod
    def _prepare_image(image_bytes: bytes, height: int, width: int) -> np.ndarray:
        """Apply wd-tagger-style preprocessing and return an NCHW float32 tensor.

        Pipeline: white canvas for RGBA → pad to square → resize →
        float32 → BGR → NHWC (wd-tagger convention; the graph does its
        own NHWC→NCHW transpose if needed).
        """
        img = Image.open(BytesIO(image_bytes))

        # White-canvas composite for non-RGB modes (RGBA, palette, LA).
        # Mirrors the wd-tagger reference.
        canvas = Image.new("RGBA", img.size, (255, 255, 255))
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        canvas.alpha_composite(img)
        img = canvas.convert("RGB")

        # Pad to a square with white.
        max_dim = max(img.size)
        pad_left = (max_dim - img.size[0]) // 2
        pad_top = (max_dim - img.size[1]) // 2
        padded = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
        padded.paste(img, (pad_left, pad_top))

        # Resize to the model's input size.
        if max_dim != height or max_dim != width:
            padded = padded.resize((width, height), Image.Resampling.BICUBIC)

        # Cast to float32 and flip RGB → BGR. No explicit /255 — wd-tagger
        # models include the 0-255 → 0-1 normalization in the graph.
        arr = np.asarray(padded, dtype=np.float32)
        arr = arr[:, :, ::-1]
        return np.expand_dims(arr, axis=0)

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
        thr = thresholds.get(cat_name, 0.0)
        if thr <= 0:
            continue
        for tag in cat_tags:
            score = result.tags.get(tag, 0.0)
            if score < thr:
                drop.add(tag)

    if not drop:
        return result

    filtered_tags = {k: v for k, v in result.tags.items() if k not in drop}
    filtered_categories = {cat: [t for t in tags if t not in drop] for cat, tags in result.categories.items()}
    return TaggerResult(tags=filtered_tags, categories=filtered_categories)
