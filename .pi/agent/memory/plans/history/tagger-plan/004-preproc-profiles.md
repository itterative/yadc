---
date: 2026-06-29
---
# Preprocessing profiles + sigmoid (animetimm ConvNeXt support)

**Context:** The wd-tagger pipeline hardcoded NHWC + BGR + no normalization on
the host, expecting the graph to bake /255 + NHWC→NCHW + sigmoid in. The user
tried to load a pytorch→onnx export of `animetimm/convnextv2_huge.dbv4-full`
and got ORT errors — `Input channels C: 448 kernel channels: 3`. The yadc
fallback for symbolic H/W was 448 even though the model is 512; the layout
detection also fell back to NHWC (wrong for ConvNeXt, which expects NCHW).

**Decision:**

- **`PreprocProfile` NamedTuple** in a new `yadc/taggers/onnx_preprocess.py`
  with `channel_order`, `normalize`, `default_input_size`, and
  `apply_sigmoid`. Two built-in profiles:
  - `WD_TAGGER_PROFILE` (default) — channel_order=`bgr`, normalize=`none`,
    default_input_size=448, apply_sigmoid=False. Exact match to the
    SmilingWolf wd-tagger convention used historically.
  - `TIMM_PROFILE` — channel_order=`rgb`, normalize=`imagenet`,
    default_input_size=512, apply_sigmoid=True. Standard PyTorch / timm
    pipeline used by e.g. animetimm ConvNeXt.

- **Layout auto-detection** in `resolve_input` works in three layers:
  concrete channel count at dim 1 or 3 (the unambiguous signal),
  symbolic dim names (`"num_channels"` / `"channels"` / etc. -> NCHW vs
  NHWC), with concrete values winning over name hints when they disagree.
  Verified end-to-end against the actual animetimm export, which declares
  shape `['batch_size', 'num_channels', 'height', 'width']` — fully
  symbolic, only the dim names hint at the layout.

- **Sigmoid is profile-driven** rather than off-by-default. `TIMM_PROFILE`
  applies `1 / (1 + exp(-x))` after `session.run` because timm classifier
  exports emit logits; `WD_TAGGER_PROFILE` skips it (the graph already
  applies sigmoid).

- **Configuration:**
  - `tagger_preproc_profile: str = "wd-tagger"` (backward-compatible
    default).
  - `tagger_default_input_size: int = 0` — used when the user wants to
    override the profile's built-in fallback size (e.g. for a non-default
    timm variant).
  - Both validated at startup via `get_profile()` so a typo in config
    surfaces as a clean `ValueError` rather than a hidden silent default.

- **CLI:** `--preproc-profile {wd-tagger,timm,pytorch}` +
  `--default-size N` on `yadc tagger start` / `tag`. Click validates the
  profile name through `Choice`, so a CLI typo is rejected before any
  subprocess spawns.

- **Diagnostic logs** (as requested):
  - INFO on `load_model`:
    `Tagger preprocessing contract: profile.channel_order=rgb,
    profile.normalize=imagenet, layout=nchw, size=512x512,
    profile.default_input_size=512`
  - DEBUG on every `predict`:
    `Tagger preprocessed tensor: shape=(1, 3, 512, 512)
    dtype=float32 min=-0.7479 max=2.6400 mean=1.0363`

- **Verified end-to-end** against the actual animetimm model on a solid-blue
  PNG. After `TIMM_PROFILE` + sigmoid:
  `[monochrome 0.873, solo 0.782, 1girl 0.760, sensitive 0.661, lineart
  0.660, smile 0.586, long_hair 0.519, dress 0.513, breasts 0.372,
  general 0.316]` — `monochrome` correctly tops the list for a single-color
  image; thresholds and categorization work the same as wd-tagger output.

**Rationale:**

- Separate module (`onnx_preprocess.py`) keeps the layered contract
  visible without growing `onnx.py` — per the user's request to split it
  out, and matches the existing module split (`base` / `onnx` / `server`
  / `client`).
- Layout auto-detection is layered (concrete -> name -> fallback) so any
  well-formed ONNX input shape resolves without a config knob. Channel
  order, normalization, and sigmoid aren't detectable from the graph
  alone, so they live on the profile.
- `WD_TAGGER_PROFILE` is left as the default so existing wd-tagger users
  keep working unchanged; new users select `TIMM_PROFILE` (or another)
  via `Configuration` / CLI.

**Files touched:**
- `yadc/taggers/onnx_preprocess.py` — new module: `PreprocProfile`,
  `WD_TAGGER_PROFILE`, `TIMM_PROFILE`, `resolve_input`, `prepare_image`,
  `log_profile_info`, `get_profile`, `list_profiles`.
- `yadc/taggers/onnx.py` — `OnnxTagger.__init__` now takes
  `preproc_profile` + `default_size`; `_prepare_image` /
  `_resolve_input_hw` removed (delegated to `onnx_preprocess`); `predict`
  applies sigmoid when `profile.apply_sigmoid` is true and emits a DEBUG
  line with the preprocessed tensor stats.
- `yadc/api/configuration.py` — added `tagger_preproc_profile` and
  `tagger_default_input_size` to `Configuration`.
- `yadc/api/services/tagging.py` — passes the new fields through to
  `OnnxTagger` (`_ensure_running_locked`); pre-validates the profile name
  so config typos surface at service-startup time.
- `yadc/cli_tagger.py` — `--preproc-profile` + `--default-size` on `start`
  and `tag`; `_profile_kwargs` helper centralizes profile→kwargs.
- `tests/taggers/test_onnx.py` — covers profile lookup, layout detection
  (concrete + symbolic dim names), per-profile preprocessing output,
  sigmoid-on/off paths, and the integration of the resolved contract with
  `predict`.
- `.pi/agent/memory/docs/tagger-architecture.md` — updated
  "OnnxTagger preprocessing" + Configuration table.
