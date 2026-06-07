"""``/api/datasets/<name>/caption`` endpoints — start/stop/status, refine, per-image.

This controller is a thin HTTP layer; all state lives in
``CaptioningService``. Routes:

- ``POST /datasets/<name>/caption`` — start a job. The optional JSON
  body matches ``CaptionJobOptions`` and overrides config / env defaults
  for ``api_url``, ``api_token``, ``api_model_name``, ``env``,
  ``prompt_template``, ``prompt_name``, ``max_tokens``,
  ``image_quality``, ``overwrite``, ``draft``, ``reasoning``,
  ``reasoning_effort``, ``reasoning_exclude_output``, ``password``.
  Pre-flights ``cmd_envs.load_env`` to surface ``PasswordRequiredError``
  as a 403 ``PASSWORD_REQUIRED`` *before* spawning the background job.
- ``POST /datasets/<name>/caption/stop`` — request a graceful stop.
- ``GET /datasets/<name>/caption/jobs`` — list the in-memory job log.
- ``GET /datasets/<name>/caption/status`` — current job state.
- ``POST /datasets/<name>/images/<id>/caption`` — caption a single
  image (used by the UI's "caption one" button).
- ``POST /datasets/<name>/images/<id>/refine`` — refine the caption
  via an interactive reply round (uses ``RefineOptions``); result
  cached up to ``Configuration.refine_result_buffer_size`` entries.

All errors funnel through ``utils_json.jsonify_error`` with the right
``ErrorCode`` (``NOT_FOUND``, ``CONFLICT``, ``PASSWORD_REQUIRED``, ...).
"""

from quart import jsonify, request

from yadc.cmd import envs as cmd_envs
from yadc.core.captioner import ROLE_ASSISTANT, ROLE_USER, ReplyRound
from yadc.core.captioning import CaptionJobOptions

from ..modules.logging_factory import LoggingFactory
from ..services.captioning import CaptioningService, JobInfo, RefineOptions
from . import controller
from .blueprints import ApiBlueprint
from .utils_json import ErrorCode, jsonify_dataclass, jsonify_error, validate_body


@controller
def api_captioning(app: ApiBlueprint, logging: LoggingFactory, captioning: CaptioningService):
    _logger = logging.get_logger(__name__)

    @app.post("/datasets/<name>/caption")
    async def start_captioning(name: str):  # pyright: ignore[reportUnusedFunction]
        """Start a captioning run.

        Optional JSON body fields (all override config / env defaults):
            api_url, api_token, api_model_name, env,
            prompt_template, prompt_name, max_tokens, image_quality,
            overwrite, draft, reasoning, reasoning_effort, reasoning_exclude_output,
            password
        """
        options = validate_body(CaptionJobOptions, await request.get_json(silent=True))

        try:
            # Pre-flight env load to catch password-required errors before starting a background job.
            cmd_envs.load_env(options.env, password=options.password)
        except cmd_envs.PasswordRequiredError:
            return jsonify_error(
                "Password required to decrypt environment settings",
                status=403,
                code=ErrorCode.PASSWORD_REQUIRED,
            )

        try:
            info: JobInfo = await captioning.start_job_async(name, options)
        except ValueError as e:
            return jsonify_error(str(e), status=409, code=ErrorCode.CONFLICT)

        return jsonify_dataclass(info), 202

    @app.delete("/datasets/<name>/caption")
    async def stop_captioning(name: str):  # pyright: ignore[reportUnusedFunction]
        """Stop a running captioning run."""
        stopped = await captioning.stop_job_async(name)
        if not stopped:
            return jsonify_error("No running captioning job for this dataset", status=404, code=ErrorCode.NOT_FOUND)
        return jsonify({"status": "stopping"})

    @app.get("/datasets/<name>/caption")
    async def get_captioning_status(name: str):  # pyright: ignore[reportUnusedFunction]
        """Get the current captioning status for a dataset."""
        info = await captioning.get_status_async(name)
        return jsonify_dataclass(info), 200

    @app.post("/datasets/<name>/images/<int:image_id>/caption")
    async def caption_single_image(name: str, image_id: int):  # pyright: ignore[reportUnusedFunction]
        """Start a single-image captioning job.

        Accepts the same JSON body fields as the batch endpoint.
        Returns job status immediately (202 Accepted); listen to SSE events
        or poll GET /datasets/<name>/caption for completion.
        """
        options = validate_body(CaptionJobOptions, await request.get_json(silent=True))

        try:
            cmd_envs.load_env(options.env, password=options.password)
        except cmd_envs.PasswordRequiredError:
            return jsonify_error(
                "Password required to decrypt environment settings",
                status=403,
                code=ErrorCode.PASSWORD_REQUIRED,
            )

        try:
            single_opts = options.model_copy(update={"image_ids": [image_id]})
            info: JobInfo = await captioning.start_job_async(name, single_opts)
        except ValueError as e:
            return jsonify_error(str(e), status=409, code=ErrorCode.CONFLICT)

        return jsonify_dataclass(info), 202

    @app.post("/datasets/<name>/images/<int:image_id>/refine")
    async def refine_caption(name: str, image_id: int):  # pyright: ignore[reportUnusedFunction]
        """Refine a caption by sending feedback to the model.

        Accepts the same JSON body fields as the single-image endpoint,
        plus:
            feedback (str, required): User feedback for refinement.
            refine_caption (str, optional): The caption to refine. If not provided,
                the current caption on disk is used.

        Returns job status immediately (202 Accepted); listen to SSE events
        or poll GET /datasets/<name>/caption for completion.
        """
        options = validate_body(CaptionJobOptions, await request.get_json(silent=True))

        if not options.feedback.strip():
            return jsonify_error("'feedback' is required and cannot be empty", status=400, code=ErrorCode.BAD_REQUEST)

        caption = options.refine_caption.strip()
        if not caption:
            # Read the current caption from disk
            caption_data = captioning._dataset_service.get_caption(name, image_id)
            if caption_data is None:
                return jsonify_error("Image not found", status=404, code=ErrorCode.NOT_FOUND)
            caption = (caption_data.get("caption") or "").strip()
            if not caption:
                return jsonify_error("No caption to refine — generate one first", status=400, code=ErrorCode.BAD_REQUEST)

        try:
            cmd_envs.load_env(options.env, password=options.password)
        except cmd_envs.PasswordRequiredError:
            return jsonify_error(
                "Password required to decrypt environment settings",
                status=403,
                code=ErrorCode.PASSWORD_REQUIRED,
            )

        extra_messages = [
            ReplyRound(role=ROLE_ASSISTANT, content=caption),
            ReplyRound(role=ROLE_USER, content=options.feedback.strip()),
        ]

        # Determine whether we're refining a caption or a draft
        refine_source = "caption"
        refine_draft_name = ""
        if options.draft:
            refine_source = "draft"
            refine_draft_name = options.draft

        try:
            single_opts = options.model_copy(update={"image_ids": [image_id]})
            info: JobInfo = await captioning.start_job_async(
                name,
                single_opts,
                refine=RefineOptions(
                    extra_messages=extra_messages,
                    refine_source=refine_source,
                    refine_draft_name=refine_draft_name,
                ),
            )
        except ValueError as e:
            return jsonify_error(str(e), status=409, code=ErrorCode.CONFLICT)

        return jsonify_dataclass(info), 202

    @app.get("/datasets/<name>/images/<int:image_id>/refine")
    async def get_refine_result(name: str, image_id: int):  # pyright: ignore[reportUnusedFunction]
        """Get the latest dry-run refine result for an image."""
        source = request.args.get("source", "caption")
        draft_name = request.args.get("draft_name", "")
        if source not in ("caption", "draft"):
            return jsonify_error("Invalid source — must be 'caption' or 'draft'", status=400, code=ErrorCode.BAD_REQUEST)
        result = await captioning.get_refine_result(name, image_id, source=source, draft_name=draft_name)
        if result is None:
            return jsonify_error("No refine result available", status=404, code=ErrorCode.NOT_FOUND)
        return jsonify({"caption": result})
