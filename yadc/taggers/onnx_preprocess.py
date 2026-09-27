"""Preprocessing pipelines for ONNX image taggers.

The wd-tagger SmilingWolf models include preprocessing baked into the
graph — the host only needs to provide a normalized BGR NHWC tensor and
the model handles the rest. Other PyTorch / timm exports (e.g.
``animetimm/convnextv2``) don't — they expect a NCHW RGB image normalized
with ImageNet statistics directly from the host.

This module isolates that variability:

- :class:`PreprocProfile` — a named tuple with the three knobs a host-side
  pipeline can vary: ``layout`` (NHWC vs NCHW), ``channel_order`` (RGB vs
  BGR), ``normalize`` (none vs ImageNet). The model's expected layout is
  always auto-detected from its input metadata; the profile supplies the
  other two.

- :data:`WD_TAGGER_PROFILE` — the original SmilingWolf pipeline. Default
  for backward compatibility.

- :data:`TIMM_PROFILE` — standard PyTorch / timm pipeline: NCHW + RGB +
  ImageNet. Used by ``animetimm/convnextv2`` and most ConvNeXt-based
  tagging exports.

- :func:`resolve_input` — parses the model's input metadata and returns
  ``(height, width, layout)``. Falls back to ``default_size`` for
  symbolic shapes (the convnextv2 export declares ``['batch', 3, 'H',
  'W']``).

- :func:`prepare_image` — runs the wd-tagger-style canvas → pad → resize
  pipeline (or pad-only for non-square targets) and applies the profile.
"""

from __future__ import annotations

import csv
import io
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, NamedTuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

BytesIO = io.BytesIO

# Tensor layout — channels-last (NHWC, wd-tagger convention) or
# channels-first (NCHW, PyTorch / timm convention).
Layout = Literal["nhwc", "nchw"]
# Image channel order — RGB (PyTorch native) or BGR (wd-tagger, where the
# graph flips internally).
ChannelOrder = Literal["rgb", "bgr"]
# Normalization scheme — "none" expects the model graph to bake /255 in
# (wd-tagger); "imagenet" /255 then (x - mean) / std with the canonical
# ImageNet statistics.
Normalize = Literal["none", "imagenet"]

# Canonical ImageNet mean/std — used by timm and most PyTorch image
# classification preprocessing pipelines (the same values the convnextv2
# export expects).
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


class PreprocProfile(NamedTuple):
    """Preprocessing profile for an ONNX tagger.

    Attributes:
        channel_order: ``"rgb"`` or ``"bgr"`` — the order the model expects
            on the *input tensor's* channels axis. ``"bgr"`` causes a
            channel flip before normalization/layout transforms.
        normalize: ``"none"`` leaves the input in [0, 255] (the wd-tagger
            pipeline bakes /255 into the graph); ``"imagenet"`` applies
            the standard ImageNet normalization (divide by 255, subtract
            mean, divide by std).
        default_input_size: Fallback square side length when the model's
            input shape has symbolic H/W dims (e.g. ``['batch', 3, 'H',
            'W']``). ``0`` means "no fallback" — error out for fully
            dynamic shapes so the user has to set this explicitly.
        apply_sigmoid: ``True`` applies ``sigmoid`` to the model output
            before it reaches the caller. ``False`` returns the model's
            raw output (wd-tagger models bake the sigmoid into the
            graph; PyTorch / timm classifier exports typically emit
            logits and expect the host to apply sigmoid).
    """

    channel_order: ChannelOrder = "bgr"
    normalize: Normalize = "none"
    default_input_size: int = 0
    apply_sigmoid: bool = False


#: Original SmilingWolf / wd-tagger pipeline. The ONNX graph is expected
#: to have /255 and NHWC→NCHW baked in, so the host just delivers a
#: normalized BGR NHWC tensor. The graph also returns post-sigmoid
#: probabilities, so no host-side sigmoid is applied.
WD_TAGGER_PROFILE = PreprocProfile(
    channel_order="bgr",
    normalize="none",
    default_input_size=448,
    apply_sigmoid=False,
)

#: Standard PyTorch / timm pipeline (e.g. animetimm/convnextv2). Host
#: produces an NCHW float32 tensor normalized with ImageNet statistics;
#: the model graph runs a bare inference and emits raw logits, which
#: the host converts to probabilities with sigmoid.
TIMM_PROFILE = PreprocProfile(
    channel_order="rgb",
    normalize="imagenet",
    default_input_size=512,
    apply_sigmoid=True,
)


_PROFILES: dict[str, PreprocProfile] = {
    "wd-tagger": WD_TAGGER_PROFILE,
    "wdtagger": WD_TAGGER_PROFILE,  # alias without the hyphen
    "timm": TIMM_PROFILE,
    "pytorch": TIMM_PROFILE,  # alias
}


def get_profile(name: str) -> PreprocProfile:
    """Look up a named preprocessing profile.

    Accepts short aliases (``"wdtagger"``, ``"pytorch"``) in addition to
    the canonical ``"wd-tagger"`` and ``"timm"`` names.

    Raises:
        ValueError: if ``name`` isn't a known profile.
    """
    key = name.strip().lower()
    try:
        return _PROFILES[key]
    except KeyError:
        known = sorted({k for k in _PROFILES})
        raise ValueError(f"Unknown preproc profile: {name!r}. Known profiles: {known}") from None


def list_profiles() -> list[str]:
    """Return the canonical profile names (deduplicated)."""
    return ["wd-tagger", "timm"]


_CHANNEL_DIM_NAMES = ("num_channels", "channel", "channels", "_c", "/c", " c")
_SPATIAL_H_DIM_NAMES = ("height", "_h", "/h", " h")
_SPATIAL_W_DIM_NAMES = ("width", "_w", "/w", " w")
_BATCH_DIM_NAMES = ("batch_size", "batch", "batch_dim")


def _shape_dim_as_int(value: object) -> int:
    """Convert an ONNX shape entry to an int, treating symbolic dims as 0.

    ``onnxruntime`` shape entries are either ``int`` (concrete),
    ``str`` (symbolic — typically ``"H"`` / ``"W"`` / ``"batch"`` for
    rank-4 dim tags) or ``None``. Anything else raises ``ValueError``.
    """
    if isinstance(value, bool):
        # ``bool`` is a subclass of ``int`` in Python; treat separately
        # to avoid accidentally accepting ``True`` / ``False`` as channel counts.
        raise ValueError(f"Unexpected shape entry: {value!r}")
    if isinstance(value, int):
        return value
    return 0


def _dim_name_matches(value: object, candidates: tuple[str, ...]) -> bool:
    """Return ``True`` when ``value`` is a symbolic dim whose name matches the candidate set.

    Substring match is case-insensitive. ``"channel"`` matches
    ``"num_channels"`` and ``"in_channels"``; ``"height"`` matches
    ``"image_height"`` and ``"img_h"``. Pure-int values never match
    (those carry their own channel-or-spatial info via the (1, 3, 4)
    membership test).
    """
    if not isinstance(value, str):
        return False
    lower = value.strip().lower()
    return any(candidate in lower for candidate in candidates)


def resolve_input(
    input_meta_shape: Sequence[object] | None,
    profile: PreprocProfile,
) -> tuple[int, int, Layout]:
    """Derive (height, width, layout) from an ONNX input metadata shape.

    Detection order:

    1. If any dim is a concrete channel count (1, 3, or 4) — at dim 1 or
       dim 3 — pick that layout immediately. This handles both static
       shapes (``[1, 3, 512, 512]``, ``[1, 448, 448, 3]``) and
       partially-symbolic shapes (``[1, 3, 'H', 'W']``,
       ``[1, 'H', 'W', 3]``).
    2. Otherwise inspect symbolic dim *names*. ``"num_channels"`` /
       ``"channel"`` / etc. in dim 1 → NCHW; in dim 3 → NHWC. Same for
       ``"height"`` / ``"width"``. Covers fully-symbolic exports like
       the animetimm ConvNeXt (``['batch_size', 'num_channels',
       'height', 'width']``).
    3. Otherwise fall back to ``profile.default_input_size`` for the side
       length and assume the profile's expected layout (NHWC for
       WD-tagger; this is the documented default).

    Args:
        input_meta_shape: The shape attribute from
            ``onnxruntime.InferenceSession.get_inputs()[0].shape``.
        profile: The active preprocessing profile (used only for the
            fallback side length).

    Returns:
        ``(height, width, layout)``.

    Raises:
        ValueError: if the shape can't be parsed at all (rank != 4, or
            no channel signal in concrete or symbolic form) and no
            fallback size is available.
    """
    layout: Layout | None = None
    height: int = 0
    width: int = 0

    if input_meta_shape is not None and len(input_meta_shape) == 4:
        raw: list[object] = list(input_meta_shape)
        # Convert each dim to an int up-front. Symbolic entries become 0
        # so the membership-in-(1,3,4) test only matches concrete channels.
        try:
            shape = [_shape_dim_as_int(d) for d in raw]
        except ValueError:
            shape = [0, 0, 0, 0]

        # Detect NCHW: concrete channels in dim 1, or named channels in
        # dim 1 (works for fully-symbolic shapes too). Then similarly
        # for NHWC: concrete channels in dim 3, or named channels in
        # dim 3. Concrete signals trump named ones when they disagree
        # (e.g. a fully-concrete ``[1, 3, 512, 512]`` shouldn't be
        # overridden by accidental substring matches in dim 3 names).
        ch1_concrete = shape[1] in (1, 3, 4)
        ch3_concrete = shape[3] in (1, 3, 4)
        ch1_named = _dim_name_matches(raw[1], _CHANNEL_DIM_NAMES)
        ch3_named = _dim_name_matches(raw[3], _CHANNEL_DIM_NAMES)

        if ch1_concrete or (ch1_named and not ch3_concrete and not ch3_named):
            layout = "nchw"
            height = shape[2]
            width = shape[3]
        elif ch3_concrete or (ch3_named and not ch1_named):
            layout = "nhwc"
            height = shape[1]
            width = shape[2]
        elif ch1_named and ch3_named:
            # Both dims look like channels — unusual, default to NCHW
            # (the timm / PyTorch idiom).
            layout = "nchw"
            height = shape[2]
            width = shape[3]  # type: ignore[assignment]  # shape[2/3] unused; purely typed hint for mypy

    # --- (3) Fallback for symbolic H/W ---
    if (height <= 0 or width <= 0) and profile.default_input_size > 0:
        height = width = profile.default_input_size
    elif (height <= 0 or width <= 0) and layout is not None and profile.default_input_size == 0:
        # We know the layout but not the size, and there's no fallback \u2014 error.
        raise ValueError(
            f"Could not resolve H/W from ONNX input shape {input_meta_shape!r} and the "
            f"active preproc profile has no default_input_size; set tagger_default_size or "
            f"pick a profile whose default_input_size matches the model.",
        )

    if layout is None:
        raise ValueError(
            f"Could not detect ONNX input layout from shape {input_meta_shape!r}; provide an explicit preprocessing profile or fix the model export.",
        )
    if height <= 0 or width <= 0:
        raise ValueError(
            f"Could not resolve ONNX input H/W from {input_meta_shape!r} and the active "
            f"preproc profile has no default_input_size; set tagger_default_size or pick "
            f"a profile whose default_input_size matches the model.",
        )

    return height, width, layout


def prepare_image(
    image_bytes: bytes,
    height: int,
    width: int,
    layout: Layout,
    profile: PreprocProfile,
) -> np.ndarray:
    """Preprocess an image and return a batched float32 tensor.

    Pipeline:

    1. White-canvas composite for RGBA / palette / transparency modes.
    2. Fit (preserve aspect ratio) then pad with white to ``(width, height)``.
       Square inputs short-circuit the pad step.
    3. Cast to ``float32``.
    4. Channel flip if ``profile.channel_order == "bgr"``.
    5. Normalization:

       - ``"none"`` — leave values in [0, 255].
       - ``"imagenet"`` — /255, subtract ImageNet mean, divide by std.

    6. Layout transpose: NCHW host tensors get ``(H, W, C) -> (C, H, W)``.
       NHWC host tensors stay as-is.
    7. ``np.expand_dims`` for the batch axis.

    Args:
        image_bytes: Raw encoded image (JPEG/PNG/WebP/...).
        height: Target height.
        width: Target width.
        layout: ``"nhwc"`` or ``"nchw"`` — controls the final layout
            transpose.
        profile: The active preprocessing profile.
    """
    img = Image.open(BytesIO(image_bytes))

    # White-canvas composite for non-RGB modes (RGBA, palette, LA).
    canvas = Image.new("RGBA", img.size, (255, 255, 255))
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    canvas.alpha_composite(img)
    img = canvas.convert("RGB")

    # Fit (preserve aspect ratio) into (width, height), then center-pad
    # the remainder with white. Equivalent to ``resize`` + ``pad_to_size``
    # in timm pipelines. Square targets + square inputs short-circuit both
    # the resize and the pad steps.
    if img.size != (width, height):
        scale = max(width, height) / max(img.size)
        new_size = (
            max(1, int(round(img.size[0] * scale))),
            max(1, int(round(img.size[1] * scale))),
        )
        img = img.resize(new_size, Image.Resampling.BICUBIC)
        if img.size != (width, height):
            padded = Image.new("RGB", (width, height), (255, 255, 255))
            pad_left = (width - img.size[0]) // 2
            pad_top = (height - img.size[1]) // 2
            padded.paste(img, (pad_left, pad_top))
            img = padded

    arr = np.asarray(img, dtype=np.float32)

    # Channel order.
    if profile.channel_order == "bgr":
        arr = arr[:, :, ::-1]

    # Normalization.
    if profile.normalize == "imagenet":
        # /255 first, then (x - mean) / std. Express mean/std in the
        # [0, 255] domain so the divide-by-1.0 broadcast matches a
        # pre-/255 array.
        arr = arr / 255.0
        mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)
        std = np.asarray(IMAGENET_STD, dtype=np.float32)
        arr = (arr - mean) / std

    # Layout — transpose only when NCHW is requested.
    if layout == "nchw":
        arr = arr.transpose(2, 0, 1)

    return np.expand_dims(arr, axis=0)


def log_profile_info(
    profile: PreprocProfile,
    height: int,
    width: int,
    layout: Layout,
) -> None:
    """Emit a one-line log describing the resolved preproc contract.

    Called from :meth:`OnnxTagger.load_model` so the user can confirm
    the detected layout / size / profile before the first inference.
    """
    logger.info(
        "Tagger preprocessing contract: profile.channel_order=%s, profile.normalize=%s, layout=%s, size=%dx%d, profile.default_input_size=%d",
        profile.channel_order,
        profile.normalize,
        layout,
        height,
        width,
        profile.default_input_size,
    )


PER_TAG_THRESHOLD_COLUMNS = frozenset({"best_threshold", "best_recall"})


def _load_per_tag_thresholds(path: Path) -> dict[str, dict[str, float]] | None:
    """Extract per-tag threshold columns from a selected_tags.csv.

    Returns {column_name: {tag_name: threshold_value}} when threshold
    columns are present, None otherwise. Supported columns: best_threshold,
    best_recall. Values that aren't valid floats between 0 and 1 are skipped.
    """
    result: dict[str, dict[str, float]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return None
        available = PER_TAG_THRESHOLD_COLUMNS.intersection(reader.fieldnames)
        if not available:
            return None
        for col in available:
            result[col] = {}
        for row in reader:
            name = row.get("name")
            if not name:
                continue
            for col in available:
                raw = row.get(col, "")
                try:
                    val = float(raw)
                except (ValueError, TypeError):
                    continue
                if 0.0 <= val <= 1.0:
                    result[col][name] = val
    return result if result else None
