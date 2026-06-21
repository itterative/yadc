"""``/api/prompts/history/...`` endpoints — save / list / get / delete prompt-history entries.

Mirror of the prompt-generation endpoint contract: small Pydantic
body for the save, JSON-friendly dicts for the list and get
responses, opaque pagination cursor for the list. Separate file
from ``api_prompts.py`` to keep the streaming controller readable
— the history endpoints are all request/response, no streaming.

Each save goes through :meth:`PromptHistoryService.save_entry`,
which inserts the row and prunes to ``PROMPT_HISTORY_MAX_ENTRIES``
in one transaction. The list view is computed on the server so
the frontend doesn't have to derive ``intent_preview`` /
``example_count`` / ``had_template`` from raw rows.
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal

import pydantic
from quart import jsonify, request

from ..services.prompt_generation import ExamplePair, PromptGenerationFocus
from ..services.prompt_history import PromptHistorySaveRequest, PromptHistoryService
from . import controller
from .blueprints import ApiBlueprint
from .utils_json import ErrorCode, jsonify_dataclass, jsonify_error, validate_body


class SaveHistoryBody(pydantic.BaseModel):
    """Wire-format body for ``POST /api/prompts/history``.

    Mirrors the service-layer :class:`PromptHistorySaveRequest` —
    the controller validates the raw dict, then forwards to
    ``service.save_entry`` which re-validates with the service
    model. The double validation is cheap and keeps the
    service contract self-contained for tests.
    """

    mode: Literal["generate", "refine"]
    intent: str
    focus: PromptGenerationFocus
    examples: list[ExamplePair] = pydantic.Field(default_factory=list)
    template_content: str | None = None

    model_config: ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(extra="forbid")


def _to_save_request(body: SaveHistoryBody) -> PromptHistorySaveRequest:
    """Re-validate the body through the service-layer Pydantic model.

    Pydantic's ``model_validate`` runs the full validator chain, so
    any field constraint (e.g. ``focus`` being a Literal) gets
    enforced here. The service-layer model is the source of truth
    for the save contract.
    """
    return PromptHistorySaveRequest(
        mode=body.mode,
        intent=body.intent,
        focus=body.focus,
        examples=body.examples,
        template_content=body.template_content,
    )


@controller
def api_prompts_history(app: ApiBlueprint, prompt_history: PromptHistoryService):
    @app.post("/prompts/history")
    async def save_history():  # pyright: ignore[reportUnusedFunction]
        """Save a prompt-history entry.

        Body: ``{mode, intent, focus, examples, template_content?}``.
        The full entry is returned (with the new ``id``) so the
        frontend can update its in-memory cache without a re-fetch.
        """
        body = validate_body(SaveHistoryBody, await request.get_json(silent=True))
        save_request = _to_save_request(body)
        row_id = prompt_history.save_entry(save_request)
        entry = prompt_history.get_entry(row_id)
        if entry is None:
            # Shouldn't happen — we just inserted it. Surface as 500
            # rather than swallow.
            return jsonify_error("Failed to read back saved entry", status=500, code=ErrorCode.INTERNAL_ERROR)
        return jsonify(entry)

    @app.get("/prompts/history")
    def list_history():  # pyright: ignore[reportUnusedFunction]
        """List history entries (newest first, paginated).

        Query params:
            limit: Max entries to return (default 50).
            next:  Opaque pagination cursor from the previous
                   page's ``next_token``.

        The list view omits the full ``examples`` payload (avoids
        shipping base64 image blobs for the whole history on every
        list refresh) and the full ``intent`` (the preview is
        enough for the row card; the full text comes back on the
        ``GET /history/<id>`` fetch that restore triggers).
        """
        limit = request.args.get("limit", 50, type=int)
        next_token = request.args.get("next")
        page = prompt_history.list_history(limit=limit, next_token=next_token)
        return jsonify_dataclass(page)

    @app.get("/prompts/history/<int:entry_id>")
    def get_history(entry_id: int):  # pyright: ignore[reportUnusedFunction]
        """Return a single history entry with the full ``examples`` list (for restore)."""
        entry: dict[str, Any] | None = prompt_history.get_entry(entry_id)
        if entry is None:
            return jsonify_error(f"History entry {entry_id} not found", status=404, code=ErrorCode.NOT_FOUND)
        return jsonify(entry)

    @app.delete("/prompts/history/<int:entry_id>")
    def delete_history(entry_id: int):  # pyright: ignore[reportUnusedFunction]
        """Delete a single history entry. 404 if not found."""
        if not prompt_history.delete_entry(entry_id):
            return jsonify_error(f"History entry {entry_id} not found", status=404, code=ErrorCode.NOT_FOUND)
        return jsonify({"status": "ok"})
