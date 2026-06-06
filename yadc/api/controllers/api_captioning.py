from quart import jsonify, request

from yadc.cmd import envs as cmd_envs
from yadc.core.captioning import CaptionJobOptions

from ..modules.logging_factory import LoggingFactory
from ..services.captioning import CaptioningService, JobInfo
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
