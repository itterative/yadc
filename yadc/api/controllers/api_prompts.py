"""``/api/prompts/...`` endpoints — generate Jinja2 prompt templates via an LLM.

Single endpoint: ``POST /api/prompts/generate`` returns a streaming
NDJSON response that the frontend consumes with ``fetch`` +
``ReadableStream`` + ``AbortController`` (no SSE event bus — the
generator is one-shot and doesn't need history replay).

Each line is a JSON object:

- ``{"type": "token", "text": "...", "reasoning": "..."}`` — one per
  ``StreamChunk`` yielded by the LLM
- ``{"type": "done"}`` — sent after the stream completes naturally
- ``{"type": "error", "message": "..."}`` — sent if the service raises
  mid-stream

Preflight errors (missing env URL, password required) return the
standard JSON error responses (400 / 403) *before* the streaming
response starts, so the frontend doesn't have to parse an error out of
the stream. Cancellation propagates the normal Quart way: a client
disconnect cancels the response coroutine, which propagates
``CancelledError`` into the inner generator's ``async for``, which
closes the underlying ``MessageStream`` and the upstream HTTP response.
"""

import json
from collections.abc import AsyncIterator
from typing import Any, ClassVar

import pydantic
from quart import Response, request

from yadc.cmd import envs as cmd_envs
from yadc.cmd.envs.keystorage_password import PasswordRequiredError

from ..modules.logging_factory import LoggingFactory
from ..services.prompt_generation import ExamplePair, PromptGenerationFocus, PromptGenerationRequest, PromptGenerationService
from . import controller
from ._password import resolve_request_password
from .blueprints import ApiBlueprint
from .utils_json import ErrorCode, jsonify_error, validate_body


class GeneratePromptBody(pydantic.BaseModel):
    """Body for ``POST /api/prompts/generate``."""

    env: str
    intent: str
    examples: list[ExamplePair] = pydantic.Field(default_factory=list)
    focus: PromptGenerationFocus = "both"
    api_model_name: str | None = None

    model_config: ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(extra="forbid")


def _chunk_to_event(chunk: Any) -> dict[str, Any]:
    """Translate a :class:`StreamChunk` to a serialisable NDJSON line."""
    event: dict[str, Any] = {"type": "token"}
    if chunk.text is not None:
        event["text"] = chunk.text
    if chunk.reasoning is not None:
        event["reasoning"] = chunk.reasoning
    if chunk.reasoning_summary is not None:
        event["reasoning_summary"] = chunk.reasoning_summary
    return event


@controller
def api_prompts(app: ApiBlueprint, logging: LoggingFactory, prompt_generation: PromptGenerationService):
    _logger = logging.get_logger(__name__)

    @app.post("/prompts/generate")
    async def generate_prompt():  # pyright: ignore[reportUnusedFunction]
        """Generate a Jinja2 prompt template via the named env's model.

        Body:
            env (str):                 Environment name (decrypts the API token).
            intent (str):              What the prompt should do.
            examples (ExamplePair[]):  Optional few-shot examples (subject → caption).
            focus ("system" | "user" | "both"): Which template blocks to populate.
            api_model_name (str|null): Override the env's default model.

        Returns a streaming NDJSON response — see the module docstring for
        the wire format. The decryption password is read from the
        ``yadc_password`` session cookie (with the ``YADC_PASSWORD``
        env-var fallback).
        """
        body = validate_body(GeneratePromptBody, await request.get_json(silent=True))
        password = resolve_request_password(request)

        # Synchronous preflight: catch password-required + misconfig
        # BEFORE the streaming response starts so the HTTP status code
        # can be 403/400 instead of a body-level error line. Mirrors
        # ``api_captioning.start_captioning``.
        try:
            preflight_config = cmd_envs.load_env(body.env, password=password)
        except PasswordRequiredError:
            return jsonify_error(
                "Password required to decrypt environment settings",
                status=403,
                code=ErrorCode.PASSWORD_REQUIRED,
            )

        if not preflight_config.api.url:
            return jsonify_error(
                f"Environment '{body.env}' has no api_url configured",
                status=400,
                code=ErrorCode.BAD_REQUEST,
            )

        if not (body.api_model_name or preflight_config.api.model_name):
            return jsonify_error(
                f"Environment '{body.env}' has no api_model_name configured; pass api_model_name explicitly",
                status=400,
                code=ErrorCode.BAD_REQUEST,
            )

        request_model = PromptGenerationRequest(
            env=body.env,
            intent=body.intent,
            examples=body.examples,
            focus=body.focus,
            api_model_name=body.api_model_name,
        )

        async def _stream() -> AsyncIterator[str]:
            # Track whether the model produced any text content. A
            # 200-OK stream that completes without text (safety
            # refusal, reasoning-only response, network trim) isn't a
            # usable artifact — surface it as an error so the user can
            # retry with a different model or rephrased intent.
            produced_text = False
            try:
                async for chunk in prompt_generation.generate(request=request_model, password=password):
                    if chunk.text:
                        produced_text = True
                    yield json.dumps(_chunk_to_event(chunk)) + "\n"
            except PasswordRequiredError:
                # Defensive: the controller's preflight caught this
                # already, but if a concurrent key-mode change re-encrypts
                # the token between preflight and the first chunk, surface
                # it cleanly rather than crashing the stream.
                yield json.dumps({"type": "error", "message": "Password required to decrypt environment settings"}) + "\n"
            except ValueError as e:
                _logger.warning("Prompt generation failed: %s", e)
                yield json.dumps({"type": "error", "message": str(e)}) + "\n"
            except Exception as e:
                _logger.exception("Prompt generation failed unexpectedly: %s", e)
                yield json.dumps({"type": "error", "message": f"Prompt generation failed: {e}"}) + "\n"
            else:
                if not produced_text:
                    _logger.warning("Prompt generation returned an empty response (no text chunks).")
                    yield json.dumps({"type": "error", "message": "Model returned an empty response"}) + "\n"
                else:
                    yield json.dumps({"type": "done"}) + "\n"

        response = Response(_stream(), mimetype="application/x-ndjson")
        response.headers["Cache-Control"] = "no-cache"
        response.headers["X-Accel-Buffering"] = "no"
        response.timeout = None
        return response
