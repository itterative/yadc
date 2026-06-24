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

The tagger is a server-side background process (started lazily on the
first request when ``Configuration.tagger_repo_id`` /
``tagger_model_path`` is set). The controller resolves image paths
through :class:`DatasetService` and delegates to :class:`TaggingService`.
"""

from __future__ import annotations

from pathlib import Path

import pydantic
from quart import jsonify, request

from ..configuration import Configuration
from ..modules.logging_factory import LoggingFactory
from ..services.dataset_jobs import DatasetBusyError
from ..services.datasets import DatasetService
from ..services.tagging import TaggingService, TaggingThresholds, TagJobOptions, TagSaveOptions
from . import controller
from .blueprints import ApiBlueprint
from .utils_json import ErrorCode, jsonify_dataclass, jsonify_error, validate_body


class TagImageBody(pydantic.BaseModel):
    """Optional per-request threshold overrides. All fields default to ``None`` (use server config)."""

    rating_threshold: float | None = None
    general_threshold: float | None = None
    character_threshold: float | None = None
    replace_underscores: bool | None = None


class SaveTagsBody(pydantic.BaseModel):
    """Body for ``POST .../images/<id>/tags`` — write pruned tags.

    ``tags`` / ``categories`` echo the :class:`TaggerResult` shape the
    single-image ``tag`` endpoint returns, so the frontend can pass back
    exactly what the user kept after pruning. ``save`` selects the
    destination; ``source`` is a frontend label echoed in no events here
    (the write is synchronous, not job-driven).
    """

    tags: dict[str, float]
    categories: dict[str, list[str]]
    save: TagSaveOptions = pydantic.Field(default_factory=TagSaveOptions)


class CancelBody(pydantic.BaseModel):
    """Body for ``POST /tagging/cancel`` — cancel a job / interrupt the tagger.

    ``job_id`` omitted → the single-image case (no job to stop; cancel
    escalates straight to killing the subprocess if a request is
    mid-inference).
    """

    job_id: str | None = None


@controller
def api_tagging(
    app: ApiBlueprint,
    tagging: TaggingService,
    datasets: DatasetService,
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

    @app.get("/datasets/<name>/images/<int:image_id>/tag")
    async def get_cached_tag_result(name: str, image_id: int):  # pyright: ignore[reportUnusedFunction]
        """Read the cached tag result for an image, if any.

        Same URL as :func:`tag_image` (POST): POST runs the model and
        populates the cache, GET reads the cache without running it.
        Used by the interactive Tags tab on a fresh page load to
        surface a result that was produced by an earlier batch job or
        single-image tag. The cache key fingerprint includes thresholds
        and ``replace_underscores``; callers pass the same values
        they'd use for POST so the read lands in the same bucket as
        the original write. Any field omitted falls back to the server
        config (matching the POST behaviour).

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
        try:
            await tagging.save_tags(name, image_info, result, body.save, source="tagger")
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
        from yadc.taggers import TaggerResult

        result = TaggerResult(tags=dict(body.tags), categories={k: list(v) for k, v in body.categories.items()})
        content = tagging.preview_save_tags(result, body.save)
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
