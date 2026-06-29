"""Image-tagger model catalog and the persisted ``ActiveTagger`` selection.

The user picks a model (SmilingWolf HF repo or a local ONNX path) from
the WebUI; the selection is persisted via :class:`SettingsService`
under the key ``tagger.active_model`` as the JSON form of
:class:`ActiveTagger`. At runtime, :class:`TaggingService` constructs
the subprocess from the persisted selection — see ``tagger-plan``
for the wider design.

The catalog constants (``KNOWN_TAGGER_MODELS``, ``LOCAL_FILE_ENTRY``)
live in this module so adding a new curated model is a one-file
change; the active-selection schema is colocated because the catalog
entries produce ``ActiveTagger`` payloads directly.
"""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

type ActiveTaggerKind = Literal["hf", "local"]


class TaggerModelSummary(BaseModel):
    """One row in the curated catalog returned by ``GET /api/tagger/models``.

    ``id`` is the value the picker sends back to ``POST /api/tagger/swap``
    (``repo_id`` for ``kind="hf"``, or the special
    :data:`LOCAL_FILE_ENTRY.id`` to trigger the local-file prompt).
    ``display`` / ``params`` are surface-only metadata shown next to
    the dropdown entry — they never affect subprocess construction.

    ``default_preproc_profile`` / ``default_size`` ride along so the
    picker doesn't need its own per-row profile controls for curated
    HF models: each row's profile is baked in. The Local sentinel
    exposes these as user-editable fields in the picker because the
    profile for a local model is caller-specific (animetimm needs
    ``"timm"``, anything else might need ``"wd-tagger"``).

    ``sidecars`` is **internal server-side config**: extra files in
    the HF repo to download alongside the main ``model.onnx`` on a
    best-effort basis. The typical case is the ONNX external-data
    file (``model.onnx_data``) for models that exceed protobuf's 2 GB
    size limit. Not exposed via the picker API — each download
    attempt is wrapped in a try/except, so a missing sidecar just
    means the model loads as a single file. The ``exclude=True``
    flag keeps the field out of ``model_dump`` / JSON serialization
    so endpoint code doesn't have to remember to strip it.
    """

    id: str
    display: str
    params: str
    default_preproc_profile: str = "wd-tagger"
    default_size: int = 0
    sidecars: list[str] = Field(default_factory=list, exclude=True)

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)


# Curated HF repos. Custom HF repo_ids are intentionally out of
# scope for v1; if a user wants a fork they swap to local and point
# at the file. SmilingWolf rows use the wd-tagger profile (the
# SmilingWolf export convention) with the model's native input size.
# The animetimm ConvNeXt row uses the ``timm`` profile (PyTorch /
# timm convention) and the ONNX external-data sidecar because the
# graph exceeds protobuf's 2 GB size limit. The picker hides the
# profile/size controls for all curated rows so the user can't pick
# a mismatched combination.
KNOWN_TAGGER_MODELS: list[TaggerModelSummary] = [
    TaggerModelSummary(
        id="itterative/convnextv2_huge.dbv4-full-onnx",
        display="animetimm ConvNeXtV2 Huge",
        params="large · SOTA accuracy · timm profile",
        default_preproc_profile="timm",
        sidecars=["model.onnx_data"],
    ),
    TaggerModelSummary(
        id="SmilingWolf/wd-eva02-large-tagger-v3",
        display="WD EVA02 Large v3",
        params="large · highest accuracy",
    ),
    TaggerModelSummary(
        id="SmilingWolf/wd-swinv2-tagger-v3",
        display="WD SwinV2 v3",
        params="large · strong generalist",
    ),
    TaggerModelSummary(
        id="SmilingWolf/wd-vit-tagger-v3",
        display="WD ViT v3",
        params="base · fast",
    ),
    TaggerModelSummary(
        id="SmilingWolf/wd-vit-large-tagger-v3",
        display="WD ViT Large v3",
        params="large · balanced",
    ),
]

# Sentinel returned alongside ``KNOWN_TAGGER_MODELS`` so the picker
# can offer a "Local file…" entry. The actual path + profile + size
# are supplied by the user at swap time — see the SettingsDialog. The
# ``default_preproc_profile`` here is only the initial picker value
# when the user has no active selection to seed from; an existing
# animetimm selection (profile ``"timm"``) takes priority.
LOCAL_FILE_ENTRY: TaggerModelSummary = TaggerModelSummary(
    id="__local__",
    display="Local file…",
    params="ONNX, sigmoid-output, single-input — best with wd-tagger profile",
)


class ActiveTagger(BaseModel):
    """The user's persisted active tagger selection.

    Round-trips through JSON via :meth:`BaseModel.model_dump` /
    :meth:`BaseModel.model_validate` (used by ``SettingsService``).
    ``kind`` discriminates the two upload modes: ``"hf"`` downloads the
    model + labels from a HuggingFace repo in the worker process,
    ``"local"`` reads them from local paths.

    Preproc profile + default size ride along with the model because
    the choice is model-specific (animetimm ConvNeXt needs ``"timm"``,
    SmilingWolf WD models use ``"wd-tagger"``). They can be set
    independently if needed later — the dataclass leaves room.
    """

    kind: ActiveTaggerKind
    repo_id: str = ""
    repo_model_filename: str = "model.onnx"
    repo_label_filename: str = "selected_tags.csv"
    model_path: str = ""
    label_path: str = ""
    preproc_profile: str = "wd-tagger"
    default_size: int = 0

    @model_validator(mode="after")
    def _validate_consistency(self) -> ActiveTagger:
        """Reject impossible combinations (e.g. ``kind="hf"`` + empty ``repo_id``).

        Otherwise the subprocess would silently fall back to the
        empty ``model_path`` and fail to load. The settings table
        round-trip treats the parse error as "unconfigured".
        """
        if self.kind == "hf" and not self.repo_id.strip():
            raise ValueError("kind='hf' requires a non-empty repo_id")
        if self.kind == "local" and not self.model_path.strip():
            raise ValueError("kind='local' requires a non-empty model_path")
        return self

    @property
    def source_label(self) -> str:
        """Server-side identity string used in ``TaggerStatusEvent.source``.

        Mirrors the existing Configuration-derived format so the SSE
        event carries the same shape the frontend already consumes
        (``hf:<repo_id>`` / ``local:<path>``). A profile/size swap with
        the same model identity therefore does not change this label —
        cache invalidation across profile changes is a separate concern
        tracked in the plan.
        """
        if self.kind == "hf":
            return f"hf:{self.repo_id}"
        return f"local:{self.model_path}"

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
