import json
from typing import Any

import pydantic
from flask import Response, jsonify, request, stream_with_context

from ..events import CaptioningStatusEvent
from ..json_utils import DataclassJSONEncoder, jsonify_dataclass
from ..modules.logging_factory import LoggingFactory
from ..modules.sse_events import SSEEvents
from ..services.captioning import CaptioningService, CaptionJobOptions, JobInfo
from . import controller
from .blueprints import ApiBlueprint


@controller
def api_captioning(app: ApiBlueprint, logging: LoggingFactory, captioning: CaptioningService, sse_events: SSEEvents):
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

    @app.get("/datasets/<name>/caption/status")
    def captioning_status(name: str):  # pyright: ignore[reportUnusedFunction]
        """SSE stream for captioning progress for a specific dataset.

        Sends an immediate snapshot of the current status, then streams
        ``CaptioningStatusEvent`` events from the global SSE bus (filtered
        to this dataset) until the job finishes.
        """

        def _stream():
            # 1. Immediate snapshot so the caller doesn't have to wait
            info = captioning.get_status(name)
            initial = CaptioningStatusEvent(
                status=info.status,
                dataset_name=info.dataset_name,
                processed=info.processed,
                total=info.total,
                errors=info.errors,
                error=info.error,
            )
            yield f"event: {initial.TYPE}\ndata: {json.dumps(initial, cls=DataclassJSONEncoder)}\n\n"

            # If already idle/done/error, don't open a long-lived stream
            if info.status in ("idle", "done", "error"):
                return

            # 2. Subscribe to the global SSE bus, filtered to our event type
            for event in sse_events.receive(CaptioningStatusEvent):
                # receive() yields events of the requested type, but the return
                # annotation is on the base Event class.
                assert isinstance(event, CaptioningStatusEvent)

                # Only forward events for this dataset
                if event.dataset_name != name:
                    continue

                yield f"event: {event.TYPE}\ndata: {json.dumps(event, cls=DataclassJSONEncoder)}\n\n"

                if event.status in ("done", "error"):
                    break

        return Response(stream_with_context(_stream()), mimetype="text/event-stream")

    @app.delete("/datasets/<name>/caption")
    def stop_captioning(name: str):  # pyright: ignore[reportUnusedFunction]
        """Stop a running captioning run."""
        stopped = captioning.stop_job(name)
        if not stopped:
            return jsonify({"error": "No running captioning job for this dataset"}), 404
        return jsonify({"status": "stopping"})

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
