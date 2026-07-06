"""``/api/datasets/<name>/...`` tagging endpoints — single image, batch job, save, and result cache.

Routes:

- ``POST /datasets/<name>/images/<int:image_id>/tag`` — tag a single
  image **synchronously** and return the result (used by the interactive
  Tag button in the image detail UI; the result populates the prune grid
  immediately, no job/SSE round-trip). Also writes to the in-memory
  result LRU (see :class:`TaggingService._tag_results`).
- ``GET /datasets/<name>/images/<int:image_id>/tag`` — read the cached
  tag result for an image without re-running the model. Used by the
  Tags tab on a fresh page load to surface a result produced by an
  earlier POST or batch job. Accepts the same threshold +
  ``replace_underscores`` overrides as POST so the read lands in the
  same cache bucket as the original write. Returns ``404`` when the
  entry was never cached, was evicted by LRU, or was tagged under
  different settings.
- ``POST /datasets/<name>/tag`` — start a **batch** tagging job
  (``image_ids`` omitted → whole dataset; ``[id]`` → single image via
  the job path). Returns ``202`` + ``TagJobInfo``; progress flows via
  ``TagJobStatusEvent`` / ``ImageTaggedEvent`` / ``ImageTagErrorEvent``.
  Per-image results are written to the same LRU as the single-image
  endpoint, so a subsequent GET on any of those images is a cache hit.
- ``DELETE /datasets/<name>/tag`` — graceful stop.
- ``GET /datasets/<name>/tag`` — current job status.
- ``GET /datasets/<name>/tag/jobs`` — all tracked job snapshots.
- ``POST /datasets/<name>/images/<int:image_id>/tags`` — write
  **user-pruned** tags to a draft / extras without re-running the model.
- ``POST /datasets/<name>/images/<int:image_id>/tags/preview`` —
  preview what the save would write, without touching disk.
- ``POST /tagging/cancel`` — cancel a batch job and/or interrupt an
  in-flight single-image tag (graceful-first, kill-as-fallback).
- ``POST /tagging/preview`` — context-free format preview; no dataset
  or image required (used by settings panels).
- ``GET /tagging/suggest`` — autocomplete for the Tags tab's custom-tag
  input (query + limit).
- ``GET /tagging/suggest/variant`` — return the active + default tag-
  suggestion catalog variant and the full variant list.
- ``PUT /tagging/suggest/variant`` — switch the active variant (persists
  + reloads).
- ``GET /tagging/highlights`` / ``PUT /tagging/highlights`` — global
  tier curation (starred / desired / undesired + category overrides)
  stored under the ``settings`` KV table at
  ``tagger.tag_highlights``.
- ``GET /datasets/<name>/tag/policy`` / ``PUT .../tag/policy`` —
  per-dataset always-add / banned policy, stored under the
  ``dataset_settings`` table (key ``policy_always_add`` /
  ``policy_banned``).

The tagger is a server-side background process (started lazily on the
first request when ``Configuration.tagger_repo_id`` /
``tagger_model_path`` is set). The controller resolves image paths
through :class:`DatasetService` and delegates to :class:`TaggingService`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pydantic
from quart import Response, jsonify, request

from yadc.taggers.onnx_preprocess import list_profiles

from ..configuration import Configuration
from ..modules.dataset_watcher import SELF_JOB_ID
from ..modules.logging_factory import LoggingFactory
from ..modules.tagger_catalog import (
    KNOWN_TAGGER_MODELS,
    LOCAL_FILE_ENTRY,
    ActiveTagger,
)
from ..services.dataset_jobs import DatasetBusyError
from ..services.datasets import DatasetService
from ..services.tag_highlights_service import TaggedEntry, TagHighlights, TagHighlightsService
from ..services.tag_policy_service import StoredPolicy, TagPolicyService
from ..services.tag_suggestions import CatalogVariant, TagSuggestionsService, list_variants, resolve_variant
from ..services.tagging import (
    TaggerBusyError,
    TaggerSwapInProgressError,
    TaggingService,
    TaggingThresholds,
    TagJobOptions,
    TagSaveOptions,
)
from . import controller
from .blueprints import ApiBlueprint
from .utils_json import ErrorCode, jsonify_dataclass, jsonify_error, validate_body


class TagImageBody(pydantic.BaseModel):
    """Optional per-request threshold overrides. All fields default to ``None`` (use server config).

    The always-add / banned policy is not carried on the wire — the
    backend resolves it per-dataset from :class:`TagPolicyService`.
    """

    rating_threshold: float | None = None
    general_threshold: float | None = None
    character_threshold: float | None = None
    replace_underscores: bool | None = None


class TagCustomizationsBody(pydantic.BaseModel):
    """Body shape mirroring :class:`TagCustomizations` — the user's edits.

    Optional on the preview request; persisted onto the cached result
    so navigation restores the selection. Absent / ``null`` leaves the
    cache entry untouched.
    """

    disabled: list[str]
    custom_tags: dict[str, list[str]] = pydantic.Field(default_factory=dict)


class SaveTagsBody(pydantic.BaseModel):
    """Body for ``POST .../images/<id>/tags`` — write pruned tags.

    ``tags`` / ``categories`` echo the :class:`TaggerResult` shape the
    single-image ``tag`` endpoint returns, so the frontend can pass back
    exactly what the user kept after pruning. ``save`` selects the
    destination; ``source`` is a frontend label echoed in no events here
    (the write is synchronous, not job-driven). ``customizations`` is
    only honored by the preview route (it persists the selection to the
    cache); the save route ignores it.
    """

    tags: dict[str, float]
    categories: dict[str, list[str]]
    save: TagSaveOptions = pydantic.Field(default_factory=TagSaveOptions)
    customizations: TagCustomizationsBody | None = None


class CancelBody(pydantic.BaseModel):
    """Body for ``POST /tagging/cancel`` — cancel a job / interrupt the tagger.

    ``job_id`` omitted → the single-image case (no job to stop; cancel
    escalates straight to killing the subprocess if a request is
    mid-inference).
    """

    job_id: str | None = None


class SuggestionVariantBody(pydantic.BaseModel):
    """Body for ``PUT /tagging/suggest/variant`` — switch catalog variant.

    ``variant`` is a ``CatalogVariant`` value (``anima`` / ``illustrious`` /
    ``noobaixl``); the controller coerces it to the enum so the service
    never sees a raw string.
    """

    variant: str


class TagPolicyBody(pydantic.BaseModel):
    """Body for ``PUT /datasets/<name>/tag/policy`` — replace the always-add / banned lists.

    Each entry is the canonical ``{name, canonical_form}`` shape
    (``canonical_form: True`` for catalog picks — the frontend
    applies the user's ``replace_underscores`` preference; ``False``
    for free-text entries or kaomojis whose literal identity is
    pinned). Stored and returned verbatim; the backend validator
    auto-flips the flag to ``False`` for kaomojis regardless of
    what the client sent.
    """

    always_add: list[TaggedEntry]
    banned: list[TaggedEntry]


class TagHighlightsBody(pydantic.BaseModel):
    """Body for ``PUT /tagging/highlights`` — replace the global tag tiers in full.

    Each tier entry is the canonical ``{name, canonical_form}``
    shape. Stored and returned verbatim; the backend validator
    auto-flips ``canonical_form`` to ``False`` for kaomojis.
    """

    starred: list[TaggedEntry]
    desired: list[TaggedEntry]
    undesired: list[TaggedEntry]
    category_overrides: dict[str, str] = pydantic.Field(default_factory=dict)


@controller
def api_tagging(
    app: ApiBlueprint,
    tagging: TaggingService,
    datasets: DatasetService,
    tag_suggestions: TagSuggestionsService,
    tag_highlights: TagHighlightsService,
    tag_policy: TagPolicyService,
    configuration: Configuration,
    logging: LoggingFactory,
):
    _logger = logging.get_logger(__name__)

    @app.post("/datasets/<name>/images/<int:image_id>/tag")
    async def tag_image(name: str, image_id: int):  # pyright: ignore[reportUnusedFunction]
        """Tag a single image in a dataset (synchronous).

        Optional JSON body fields (all default to the server's
        configured values):

            rating_threshold, general_threshold, character_threshold, replace_underscores

        Returns a :class:`TaggerResult` JSON: ``{"tags": {tag: score}, "categories": {cat: [tag, ...]}}``.
        """
        if not tagging.is_configured:
            return jsonify_error("Tagger is not configured on this server", status=503, code=ErrorCode.SERVICE_UNAVAILABLE)

        # Resolve the image before any subprocess work — the tagger
        # service is told only about images we know exist on disk.
        image_info = datasets.get_image(name, image_id)
        if image_info is None:
            return jsonify_error(
                f"Image not found in dataset '{name}': id={image_id}",
                status=404,
                code=ErrorCode.NOT_FOUND,
            )
        image_path = Path(image_info.path)
        if not image_path.exists():
            return jsonify_error(
                f"Image file no longer exists on disk: {image_path}",
                status=404,
                code=ErrorCode.NOT_FOUND,
            )

        body = validate_body(TagImageBody, await request.get_json(silent=True))

        # Frontend-supplied label (e.g. a model name from a dropdown),
        # echoed in the dispatched SSE events so other tabs / clients
        # can display what the requester used. Falls back to the
        # server-configured model when not provided.
        source = request.args.get("source") or None

        thresholds: TaggingThresholds | None = None
        if any(v is not None for v in (body.rating_threshold, body.general_threshold, body.character_threshold)):
            thresholds = TaggingThresholds(
                rating=body.rating_threshold if body.rating_threshold is not None else configuration.tagger_rating_threshold,
                general=body.general_threshold if body.general_threshold is not None else configuration.tagger_general_threshold,
                character=body.character_threshold if body.character_threshold is not None else configuration.tagger_character_threshold,
            )

        try:
            result = await tagging.tag_single_image_async(
                name,
                image_info,
                thresholds=thresholds,
                replace_underscores=body.replace_underscores,
                source=source,
            )
        except DatasetBusyError as exc:
            return jsonify_error(str(exc), status=409, code=ErrorCode.CONFLICT)
        except RuntimeError as exc:
            return jsonify_error(str(exc), status=503, code=ErrorCode.SERVICE_UNAVAILABLE)

        return jsonify_dataclass(result), 200

    @app.post("/datasets/<name>/tag")
    async def start_tag_job(name: str):  # pyright: ignore[reportUnusedFunction]
        """Start a batch tagging job for a dataset.

        Optional JSON body (``TagJobOptions``): ``image_ids``, per-category
        threshold overrides, ``save`` (``TagSaveOptions``), ``source``.
        Returns ``202`` + ``TagJobInfo``.
        """
        if not tagging.is_configured:
            return jsonify_error("Tagger is not configured on this server", status=503, code=ErrorCode.SERVICE_UNAVAILABLE)

        options = validate_body(TagJobOptions, await request.get_json(silent=True))
        try:
            info = await tagging.start_tag_job_async(name, options)
        except ValueError as exc:
            return jsonify_error(str(exc), status=409, code=ErrorCode.CONFLICT)
        except RuntimeError as exc:
            return jsonify_error(str(exc), status=503, code=ErrorCode.SERVICE_UNAVAILABLE)
        return jsonify_dataclass(info), 202

    @app.delete("/datasets/<name>/tag")
    async def stop_tag_job(name: str):  # pyright: ignore[reportUnusedFunction]
        """Request cancellation of a running tagging job (graceful + escalate).

        Thin delegate over ``POST /tagging/cancel``: looks up the
        dataset's current job id and hands it to ``cancel_async``, so a
        job stuck mid-inference escalates to killing the subprocess.
        """
        job_id = tagging.job_id_for_dataset(name)
        if job_id is None:
            return jsonify_error("No running tagging job for this dataset", status=404, code=ErrorCode.NOT_FOUND)
        result = await tagging.cancel_async(job_id)
        return jsonify_dataclass(result)

    @app.post("/tagging/cancel")
    async def cancel_tagging():  # pyright: ignore[reportUnusedFunction]
        """Cancel a tagging job and/or interrupt an in-flight tag.

        Body ``{"job_id": "..."}`` cancels that batch job (graceful via
        its stop_event, escalating to a subprocess kill if an inference
        is still in flight). Omitting ``job_id`` is the single-image
        case — cancel goes straight to the kill escalation.
        """
        body = validate_body(CancelBody, await request.get_json(silent=True))
        result = await tagging.cancel_async(body.job_id)
        return jsonify_dataclass(result)

    @app.get("/datasets/<name>/tag")
    async def get_tag_job_status(name: str):  # pyright: ignore[reportUnusedFunction]
        """Get the current tagging job status for a dataset."""
        return jsonify_dataclass(await tagging.get_tag_job_status_async(name))

    @app.get("/datasets/<name>/tag/jobs")
    async def list_tag_jobs(name: str):  # pyright: ignore[reportUnusedFunction]
        """List snapshots of every tracked tagging job for a dataset."""
        jobs = await tagging.list_tag_jobs_async()
        # ``list_tag_jobs_async`` returns every dataset's jobs; filter to
        # the requested one for a scoped view.
        scoped = [j for j in jobs if j.dataset_name == name]
        return jsonify_dataclass(scoped)

    # --- per-dataset tag policy (always-add / banned) -----------------
    #
    # Stored server-side per dataset. The frontend mirror store caches
    # the same object so the UI is reactive; mutators update locally +
    # PUT. The GET also covers the "fresh tab on a different dataset"
    # case where the previous mirror isn't applicable.

    @app.get("/datasets/<name>/tag/policy")
    def get_dataset_tag_policy(name: str):  # pyright: ignore[reportUnusedFunction]
        """Return the always-add / banned policy for ``name``.

        Each entry comes back in its canonical ``{name, canonical_form}``
        shape (display is the frontend's render concern; the
        ``canonical_form`` flag tells the frontend whether to apply
        the user's ``replace_underscores`` preference). Returns the
        persisted policy (both lists may be empty) for a registered
        dataset, and ``404`` when the dataset isn't registered with
        the yadc DB.
        """
        if datasets.get_dataset(name) is None:
            return jsonify_error(f"Dataset '{name}' not found", status=404, code=ErrorCode.NOT_FOUND)
        policy = tag_policy.get(name)
        return jsonify(
            {
                "always_add": [e.model_dump() for e in policy.always_add],
                "banned": [e.model_dump() for e in policy.banned],
            }
        )

    @app.put("/datasets/<name>/tag/policy")
    async def put_dataset_tag_policy(name: str):  # pyright: ignore[reportUnusedFunction]
        """Persist the always-add / banned policy for ``name``.

        Body is the full policy shape (both lists required, both
        overwritten verbatim). The PUT response echoes the persisted
        canonical ``{name, canonical_form}`` shape. ``404`` when the
        dataset isn't registered; ``400`` when the body fails
        validation.
        """
        if datasets.get_dataset(name) is None:
            return jsonify_error(f"Dataset '{name}' not found", status=404, code=ErrorCode.NOT_FOUND)
        body = validate_body(TagPolicyBody, await request.get_json(silent=True))
        policy = StoredPolicy(always_add=list(body.always_add), banned=list(body.banned))
        stored = tag_policy.set(name, policy)
        if not stored:
            # Race: dataset was unregistered between the GET check
            # above and the SET. Surface as 404 for symmetry.
            return jsonify_error(f"Dataset '{name}' not found", status=404, code=ErrorCode.NOT_FOUND)
        return jsonify(
            {
                "always_add": [e.model_dump() for e in policy.always_add],
                "banned": [e.model_dump() for e in policy.banned],
            }
        )

    @app.get("/datasets/<name>/images/<int:image_id>/tag")
    async def get_cached_tag_result(name: str, image_id: int):  # pyright: ignore[reportUnusedFunction]
        """Read the cached tag result for an image, if any.

        Same URL as :func:`tag_image` (POST): POST runs the model and
        populates the cache, GET reads the cache without running it.
        Used by the interactive Tags tab on a fresh page load to
        surface a result that was produced by an earlier batch job or
        single-image tag. The cache key fingerprint includes thresholds
        (and model); ``replace_underscores`` and the always-add / banned
        policy are read-time transforms applied to the cached value, so
        they don't need to match the original write to hit the slot.

        Policy is resolved server-side from :class:`TagPolicyService`
        for the dataset — this endpoint no longer accepts
        ``always_add`` / ``banned`` query params (the policy move took
        them off the wire). Toggling a policy tag is reflected on the
        next read without re-tagging.

        Returns ``404`` when no entry matches (never tagged, evicted
        by LRU, or tagged under a different model / settings).
        """
        image_info = datasets.get_image(name, image_id)
        if image_info is None:
            return jsonify_error(
                f"Image not found in dataset '{name}': id={image_id}",
                status=404,
                code=ErrorCode.NOT_FOUND,
            )

        # Optional query-string overrides; ``None`` means "use config".
        def _float_arg(name: str) -> float | None:
            raw = request.args.get(name)
            return float(raw) if raw is not None else None

        rating = _float_arg("rating_threshold")
        general = _float_arg("general_threshold")
        character = _float_arg("character_threshold")
        replace_raw = request.args.get("replace_underscores")
        replace = None if replace_raw is None else replace_raw.lower() in ("true", "1", "yes")

        thresholds = (
            TaggingThresholds(
                rating=rating if rating is not None else configuration.tagger_rating_threshold,
                general=general if general is not None else configuration.tagger_general_threshold,
                character=character if character is not None else configuration.tagger_character_threshold,
            )
            if any(v is not None for v in (rating, general, character))
            else None
        )

        # Policy now lives on the server (per-dataset, ``TagPolicyService``)
        # so the cached result is re-applied against the active policy on
        # the read; no per-request override is accepted here.
        result = await tagging.get_tag_result(
            name,
            image_id,
            thresholds=thresholds,
            replace_underscores=replace,
        )
        if result is None:
            return jsonify_error("No cached tag result for this image", status=404, code=ErrorCode.NOT_FOUND)
        return jsonify_dataclass(result)

    @app.post("/datasets/<name>/images/<int:image_id>/tags")
    async def save_tags(name: str, image_id: int):  # pyright: ignore[reportUnusedFunction]
        """Write user-pruned tags for an image to a draft / extras.

        Does NOT re-run the model — the body carries the exact
        ``{tags, categories}`` the user kept after pruning. Used by the
        interactive Tags tab's save bar.
        """
        image_info = datasets.get_image(name, image_id)
        if image_info is None:
            return jsonify_error(
                f"Image not found in dataset '{name}': id={image_id}",
                status=404,
                code=ErrorCode.NOT_FOUND,
            )

        body = validate_body(SaveTagsBody, await request.get_json(silent=True))
        from yadc.taggers import TaggerResult

        result = TaggerResult(tags=dict(body.tags), categories={k: list(v) for k, v in body.categories.items()})
        # ``source`` identifies the originating tab (``"ui:<uuid>"`` from a
        # frontend tab) so the resulting ``DatasetChangedEvent`` can be
        # suppressed by the originating tab. Mirrors the caption / draft /
        # extras write endpoints.
        try:
            await tagging.save_tags(name, image_info, result, body.save, source=request.args.get("source", SELF_JOB_ID))
        except ValueError as exc:
            # ``save_tags`` → ``update_extras`` validates TOML; a malformed merge surfaces here.
            return jsonify_error(str(exc), status=400, code=ErrorCode.BAD_REQUEST)
        return jsonify({"status": "ok"})

    @app.post("/datasets/<name>/images/<int:image_id>/tags/preview")
    async def preview_tags(name: str, image_id: int):  # pyright: ignore[reportUnusedFunction]
        """Preview what the save would write, without touching disk.

        Same body as ``POST .../tags``. Returns ``{"content": "..."}`` —
        formatter text for ``draft``, the ``[tags]`` TOML sub-table for
        ``extras``, empty for ``none``.

        Doubles as the persistence path for the user's interactive
        selection: when ``customizations`` is present it is written onto
        the cached result (same key as the model output) so navigating
        away from the image and back restores the selection. The same
        threshold / ``replace_underscores`` query overrides as the GET
        route select the cache bucket to write against.
        """
        # Image existence isn't strictly required (no disk write), but
        # keep the check for a consistent 404 with the save route.
        image_info = datasets.get_image(name, image_id)
        if image_info is None:
            return jsonify_error(
                f"Image not found in dataset '{name}': id={image_id}",
                status=404,
                code=ErrorCode.NOT_FOUND,
            )

        body = validate_body(SaveTagsBody, await request.get_json(silent=True))
        from yadc.taggers import TagCustomizations, TaggerResult

        result = TaggerResult(tags=dict(body.tags), categories={k: list(v) for k, v in body.categories.items()})
        content = tagging.preview_save_tags(result, body.save)

        # Persist the selection onto the cached result (best-effort;
        # misses silently when the image was never tagged / evicted).
        if body.customizations is not None:

            def _float_arg(arg: str) -> float | None:
                raw = request.args.get(arg)
                return float(raw) if raw is not None else None

            rating = _float_arg("rating_threshold")
            general = _float_arg("general_threshold")
            character = _float_arg("character_threshold")
            thresholds = (
                TaggingThresholds(
                    rating=rating if rating is not None else configuration.tagger_rating_threshold,
                    general=general if general is not None else configuration.tagger_general_threshold,
                    character=character if character is not None else configuration.tagger_character_threshold,
                )
                if any(v is not None for v in (rating, general, character))
                else None
            )
            await tagging.set_tag_customizations(
                name,
                image_id,
                TagCustomizations(
                    disabled=list(body.customizations.disabled),
                    custom_tags={k: list(v) for k, v in body.customizations.custom_tags.items()},
                ),
                thresholds=thresholds,
            )
        return jsonify({"content": content})

    @app.post("/tagging/preview")
    async def preview_tag_formats():  # pyright: ignore[reportUnusedFunction]
        """Context-free format preview — no dataset or image required.

        Same body as :func:`preview_tags`. Used by settings panels that want
        to show what a save would produce from fixed mock data, without an
        image context. Returns ``{"content": "..."}``.
        """
        body = validate_body(SaveTagsBody, await request.get_json(silent=True))
        from yadc.taggers import TaggerResult

        result = TaggerResult(tags=dict(body.tags), categories={k: list(v) for k, v in body.categories.items()})
        content = tagging.preview_save_tags(result, body.save)
        return jsonify({"content": content})

    @app.get("/tagging/suggest")
    async def suggest_tags():  # pyright: ignore[reportUnusedFunction]
        """Autocomplete backend for the Tags tab's custom-tag input.

        Query-string params (no body): ``q`` (1–100 chars after trim;
        empty → ``[]``) and optional ``limit`` (1–50, default 20).

        Each suggestion's ``name`` is the canonical catalog identity
        (underscored, e.g. ``speech_bubble``); the dropdown applies the
        user's ``replace_underscores`` preference at render time.
        Matching runs on the canonical forms.

        Returns ``{"query", "suggestions": [{name, category}, ...]}``.
        Each suggestion carries its catalog category (the danbooru
        taxonomy: ``general`` / ``artist`` / ``copyright`` / ``character``
        / ``meta``) so the dropdown can render a category badge.

        Ranking is fzf-style fuzzy subsequence scoring (see
        :mod:`yadc.utils.fuzzy`) multiplied by ``log(post_count)`` so
        popular tags surface above equally-matching obscure ones. The
        catalog is downloaded from BetaDoggo's danbooru-tag-list releases
        and parsed once into memory by :class:`TagSuggestionsService`
        (currently pinned to NoobAIXL; variants are selectable later).

        The matcher is async and yields every few thousand iterations so
        the 140k+ catalog doesn't monopolize the event loop. The query
        echoed in the response is the trimmed input (not the normalized
        form) so the frontend can show a "no matches for …" hint.

        The active catalog variant is switchable — see
        ``GET`` / ``PUT /tagging/suggest/variant``.
        """
        raw_q = (request.args.get("q") or "").strip()
        # Length cap: 100 chars on the wire matches the same ceiling
        # ``suggest`` enforces on the matcher's side.
        if len(raw_q) > 100:
            return jsonify_error("Query too long (max 100 characters)", status=400, code=ErrorCode.BAD_REQUEST)

        raw_limit = request.args.get("limit", "20")
        try:
            limit = int(raw_limit)
        except ValueError:
            return jsonify_error("limit must be an integer", status=400, code=ErrorCode.BAD_REQUEST)

        t0 = time.perf_counter()
        suggestions = await tag_suggestions.suggest(raw_q, limit=limit)
        # Perf signal: how long the suggest pass actually took for
        # this query. Captured here (not inside the service) so the
        # log line lives next to the route, with the wire-level query
        # string and the final hit count visible together.
        elapsed_ms = (time.perf_counter() - t0) * 1000
        _logger.debug(
            "Tag suggest completed in %.1fms [query=%r, hits=%d, limit=%d]",
            elapsed_ms,
            raw_q,
            len(suggestions),
            limit,
        )
        # Re-shape tuples → dicts at the wire boundary so the matcher
        # can keep its efficient ``(name, category)`` tuple return type
        # while the JSON shape stays idiomatic for the frontend:
        # ``[{"name": ..., "category": ...}, ...]``.
        return jsonify(
            {
                "query": raw_q,
                "suggestions": [
                    {"name": name, "category": category} for name, category in suggestions
                ],
            }
        )

    @app.get("/tagging/suggest/variant")
    async def get_suggestion_variant():  # pyright: ignore[reportUnusedFunction]
        """Return the active tag-suggestion variant + the full variant list.

        Single call to populate the picker: the current ``variant`` (what's
        loaded), the ``default`` (from ``Configuration``), and ``variants``
        (every known variant with its display label).
        """
        active = tag_suggestions.variant
        default = resolve_variant(configuration.tagger_suggestion_variant)
        return jsonify(
            {
                "variant": active.value,
                "default": default.value,
                "variants": [{"value": v.value, "label": label} for v, label in list_variants()],
            }
        )

    @app.put("/tagging/suggest/variant")
    async def set_suggestion_variant():  # pyright: ignore[reportUnusedFunction]
        """Switch the active tag-suggestion variant.

        Body: ``{"variant": "noobaixl"}``. Coerced to :class:`CatalogVariant`
        (400 on an unknown value). The service persists the selection,
        drops its cached catalog, and kicks a background reload of the new
        variant — the HTTP response returns once the selection is recorded,
        without waiting for the (possibly network-bound) reload. Returns the
        same shape as :func:`get_suggestion_variant` so the caller can render
        immediately.
        """
        body = validate_body(SuggestionVariantBody, await request.get_json(silent=True))
        try:
            variant = CatalogVariant(body.variant.strip().lower())
        except ValueError:
            return jsonify_error(
                f"Unknown tag suggestion variant: {body.variant!r}",
                status=400,
                code=ErrorCode.BAD_REQUEST,
            )
        await tag_suggestions.set_variant(variant)
        default = resolve_variant(configuration.tagger_suggestion_variant)
        return jsonify(
            {
                "variant": variant.value,
                "default": default.value,
                "variants": [{"value": v.value, "label": label} for v, label in list_variants()],
            }
        )

    # --- global tag highlights (starred / desired / undesired tiers) ---
    #
    # Single row per yadc install. Lives under ``settings/tagger.tag_highlights``
    # (the same KV pattern as ``tagger.suggestion_variant``). The frontend
    # mirror store loads this on mount and PUTs the full shape on every
    # mutator so the persisted blob is always a complete representation of
    # the tiers — partial updates aren't worth the merge semantics here.

    @app.get("/tagging/highlights")
    def get_tag_highlights():  # pyright: ignore[reportUnusedFunction]
        """Return the persisted tag highlights (or defaults for a fresh install).

        Each tier entry is returned in its canonical ``{name,
        canonical_form}`` shape; ``category_overrides`` keys are
        canonical too. The frontend applies the user's
        ``replace_underscores`` preference for entries with
        ``canonical_form: True`` and renders verbatim otherwise.
        Missing / corrupted rows fall back to empty tiers in
        :class:`TagHighlightsService`.
        """
        value: TagHighlights = tag_highlights.get()
        return jsonify(
            {
                "starred": [e.model_dump() for e in value.starred],
                "desired": [e.model_dump() for e in value.desired],
                "undesired": [e.model_dump() for e in value.undesired],
                "category_overrides": dict(value.category_overrides),
            }
        )

    @app.put("/tagging/highlights")
    async def put_tag_highlights():  # pyright: ignore[reportUnusedFunction]
        """Replace the persisted tag highlights with the request body.

        Body is the full shape (all three tier lists + category
        overrides); the server overwrites the stored blob verbatim.
        The PUT response echoes the persisted canonical shape
        (``{name, canonical_form}`` per entry) so the frontend
        mirror stays in sync. 400 on Pydantic validation failure
        (e.g. wrong types).
        """
        body = validate_body(TagHighlightsBody, await request.get_json(silent=True))
        value = TagHighlights(
            starred=list(body.starred),
            desired=list(body.desired),
            undesired=list(body.undesired),
            category_overrides=dict(body.category_overrides),
        )
        tag_highlights.set(value)
        return jsonify(
            {
                "starred": [e.model_dump() for e in value.starred],
                "desired": [e.model_dump() for e in value.desired],
                "undesired": [e.model_dump() for e in value.undesired],
                "category_overrides": dict(value.category_overrides),
            }
        )

    # --- tagger model swap -----------------------------------------------
    #
    # The picker UI calls these three routes. ``GET /api/tagger/active``
    # surfaces the current selection + subprocess liveness so the panel
    # can show "Currently running: <source>". ``POST /api/tagger/swap``
    # drains any in-flight single-image call (bounded by
    # ``tagger_response_timeout_seconds``) before tearing down the old
    # subprocess and respawning with the new kwargs. ``GET
    # /api/tagger/models`` returns the curated catalog used to populate
    # the dropdown.

    @app.get("/tagger/active")
    async def get_active_tagger():  # pyright: ignore[reportUnusedFunction]
        """Return the active tagger selection and whether the subprocess is up.

        Response shape:

        ```json
        {
          "active": {"kind": "hf", "repo_id": "...", ..., "source": "hf:..."} | null,
          "is_available": true | false
        }
        ```

        ``active`` reflects both the persisted :class:`ActiveTagger`
        (SettingsService) and the flat ``Configuration`` fallback —
        the latter is synthesized so legacy setups using
        ``Configuration.tagger_model_path`` directly still see the
        right model in the picker without a no-op swap.
        ``is_available`` is whether the subprocess is currently up;
        the picker surfaces this for a "Currently running: …" line.
        ``active`` is ``null`` only when neither source has a model.
        """
        active = tagging.effective_active_tagger
        if active is None:
            return jsonify({"active": None, "is_available": False})
        payload = active.model_dump()
        payload["source"] = active.source_label
        return jsonify({"active": payload, "is_available": tagging.is_available})

    @app.post("/tagger/swap")
    async def swap_tagger():  # pyright: ignore[reportUnusedFunction]
        """Swap the active tagger model end-to-end.

        Body is the full :class:`ActiveTagger` schema (validated via
        ``model_validate``). Status codes:

        - 202 + the new active payload — swap accepted; the drain+respawn
          (including any first-run model download) runs in the background
          and is reported via the ``tagger_status`` SSE stream.
        - 400 — body failed validation (e.g. ``kind="hf"`` without ``repo_id``).
        - 409 — a batch tagging job is running; ask the user to stop it first.
        - 429 — another swap is in flight; response carries
          ``retry_after_s`` and the standard ``Retry-After`` header.

        The swap itself can take up to ``tagger_startup_timeout_seconds``
        when the model must be downloaded; because it runs detached from
        the request, the client follows the SSE stream rather than holding
        the connection open.
        """
        raw_body = await request.get_json(silent=True)
        try:
            selection = ActiveTagger.model_validate(raw_body)
        except pydantic.ValidationError:
            return jsonify_error(
                "Invalid swap body",
                status=400,
                code=ErrorCode.BAD_REQUEST,
            )

        try:
            selection = await tagging.swap_active_model(selection)
        except TaggerBusyError as exc:
            return jsonify_error(str(exc), status=409, code=ErrorCode.CONFLICT)
        except TaggerSwapInProgressError as exc:
            payload = {
                "error": str(exc),
                "retry_after_s": exc.retry_after_s,
                "code": ErrorCode.TOO_MANY_REQUESTS.value,
            }
            return Response(
                json.dumps(payload),
                status=429,
                mimetype="application/json",
                headers={"Retry-After": str(int(exc.retry_after_s))},
            )

        payload = selection.model_dump()
        payload["source"] = selection.source_label
        # 202 Accepted: validation passed and the drain+respawn runs in the
        # background (:meth:`TaggingService._swap_background`). A slow
        # first-run model download happens detached from this request; the
        # picker follows the ``tagger_status`` SSE stream to completion.
        return jsonify({"active": payload, "is_available": tagging.is_available}), 202

    @app.get("/tagger/models")
    async def list_tagger_models():  # pyright: ignore[reportUnusedFunction]
        """Return the curated tagger model catalog used by the picker dropdown.

        Static for v1 — SmilingWolf HF repos plus a "Local file…"
        sentinel. The ``profiles`` list mirrors
        :func:`yadc.taggers.onnx_preprocess.list_profiles` so the
        picker's profile dropdown stays in sync with the backend's
        supported set. ``sidecars`` is excluded from the response —
        it's internal server-side download config, not picker UI.
        Adding a new curated entry is a one-line change in
        :mod:`yadc.api.modules.tagger_catalog`.
        """
        return jsonify(
            {
                "models": [m.model_dump() for m in KNOWN_TAGGER_MODELS],
                "local": LOCAL_FILE_ENTRY.model_dump(),
                "profiles": list_profiles(),
            }
        )
