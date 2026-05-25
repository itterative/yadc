from typing import Any

import pydantic
from flask import jsonify, request

from ..json_utils import jsonify_dataclass
from ..modules.logging_factory import LoggingFactory
from ..services.captioning import CaptioningService, CaptionJobOptions, JobInfo
from . import controller
from .blueprints import ApiBlueprint


@controller
def api_captioning(app: ApiBlueprint, logging: LoggingFactory, captioning: CaptioningService):
    _logger = logging.get_logger(__name__)

    @app.post("/datasets/<name>/caption")
    def start_captioning(name: str):  # pyright: ignore[reportUnusedFunction]
        """Start a captioning run.

        Optional JSON body fields (all override config / env defaults):
            api_url, api_token, api_model_name, env,
            prompt_template, prompt_name, max_tokens, image_quality,
            overwrite, draft, reasoning, reasoning_effort, reasoning_exclude_output
        """
        raw: dict[str, Any] = request.get_json(silent=True) or {}

        try:
            options = CaptionJobOptions.model_validate(raw)
        except pydantic.ValidationError as e:
            return jsonify({"error": e.errors()}), 400

        try:
            info: JobInfo = captioning.start_job(name, options)
        except ValueError as e:
            return jsonify({"error": str(e)}), 409

        return jsonify_dataclass(info), 202

    @app.delete("/datasets/<name>/caption")
    def stop_captioning(name: str):  # pyright: ignore[reportUnusedFunction]
        """Stop a running captioning run."""
        stopped = captioning.stop_job(name)
        if not stopped:
            return jsonify({"error": "No running captioning job for this dataset"}), 404
        return jsonify({"status": "stopping"})

    @app.get("/datasets/<name>/caption")
    def get_captioning_status(name: str):  # pyright: ignore[reportUnusedFunction]
        """Get the current captioning status for a dataset."""
        info = captioning.get_status(name)
        return jsonify_dataclass(info), 200

    @app.post("/datasets/<name>/images/<int:image_id>/caption")
    def caption_single_image(name: str, image_id: int):  # pyright: ignore[reportUnusedFunction]
        """Caption a single image synchronously.

        Accepts the same JSON body fields as the batch endpoint.
        Returns the generated caption.
        """
        raw: dict[str, Any] = request.get_json(silent=True) or {}

        try:
            options = CaptionJobOptions.model_validate(raw)
        except pydantic.ValidationError as e:
            return jsonify({"error": e.errors()}), 400

        try:
            result = captioning.caption_single(name, image_id, options)
        except ValueError as e:
            return jsonify({"error": str(e)}), 409

        return jsonify(result)
