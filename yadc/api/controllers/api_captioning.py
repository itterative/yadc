from typing import Any

import pydantic
from quart import jsonify, request

from yadc.cmd import envs as cmd_envs

from ..modules.logging_factory import LoggingFactory
from ..services.captioning import CaptioningService, CaptionJobOptions, JobInfo
from . import controller
from .blueprints import ApiBlueprint
from .models_errors import APIErrorDetail
from .utils_json import ErrorCode, jsonify_dataclass, jsonify_error


@controller
def api_captioning(app: ApiBlueprint, logging: LoggingFactory, captioning: CaptioningService):
    _logger = logging.get_logger(__name__)

    @app.post("/datasets/<name>/caption")
    async def start_captioning(name: str):  # pyright: ignore[reportUnusedFunction]
        """Start a captioning run.

        Optional JSON body fields (all override config / env defaults):
            api_url, api_token, api_model_name, env,
            prompt_template, prompt_name, max_tokens, image_quality,
            overwrite, draft, reasoning, reasoning_effort, reasoning_exclude_output
        """
        raw: dict[str, Any] = await request.get_json(silent=True) or {}

        try:
            options = CaptionJobOptions.model_validate(raw)
        except pydantic.ValidationError as e:
            details = [APIErrorDetail.from_pydantic_error(err) for err in e.errors()]
            return jsonify_error("Validation failed", details=details, status=400, code=ErrorCode.VALIDATION_ERROR)

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
            info: JobInfo = captioning.start_job(name, options)
        except ValueError as e:
            return jsonify_error(str(e), status=409, code=ErrorCode.CONFLICT)

        return jsonify_dataclass(info), 202

    @app.delete("/datasets/<name>/caption")
    def stop_captioning(name: str):  # pyright: ignore[reportUnusedFunction]
        """Stop a running captioning run."""
        stopped = captioning.stop_job(name)
        if not stopped:
            return jsonify_error("No running captioning job for this dataset", status=404, code=ErrorCode.NOT_FOUND)
        return jsonify({"status": "stopping"})

    @app.get("/datasets/<name>/caption")
    def get_captioning_status(name: str):  # pyright: ignore[reportUnusedFunction]
        """Get the current captioning status for a dataset."""
        info = captioning.get_status(name)
        return jsonify_dataclass(info), 200

    @app.post("/datasets/<name>/images/<int:image_id>/caption")
    async def caption_single_image(name: str, image_id: int):  # pyright: ignore[reportUnusedFunction]
        """Start a single-image captioning job.

        Accepts the same JSON body fields as the batch endpoint.
        Returns job status immediately (202 Accepted); listen to SSE events
        or poll GET /datasets/<name>/caption for completion.
        """
        raw: dict[str, Any] = await request.get_json(silent=True) or {}

        try:
            options = CaptionJobOptions.model_validate(raw)
        except pydantic.ValidationError as e:
            details = [APIErrorDetail.from_pydantic_error(err) for err in e.errors()]
            return jsonify_error("Validation failed", details=details, status=400, code=ErrorCode.VALIDATION_ERROR)

        try:
            cmd_envs.load_env(options.env, password=options.password)
        except cmd_envs.PasswordRequiredError:
            return jsonify_error(
                "Password required to decrypt environment settings",
                status=403,
                code=ErrorCode.PASSWORD_REQUIRED,
            )

        try:
            info: JobInfo = captioning.caption_single(name, image_id, options)
        except ValueError as e:
            return jsonify_error(str(e), status=409, code=ErrorCode.CONFLICT)

        return jsonify_dataclass(info), 202
