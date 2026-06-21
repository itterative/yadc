"""``APICaptioner`` — captioner glue over a composed :class:`BaseLLMClient`.

Handles everything *captioning*-specific and delegates the HTTP /
streaming / SSE-decoding / native-reasoning separation to a
:class:`yadc.llm.BaseLLMClient`:

- Jinja prompt rendering (``prompts_from_image``) and image resize +
  base64 encoding (``_encode_image``) — inherited from :class:`Captioner`.
- The advanced-settings ``conversation_overrides``: role overrides,
  ``assistant_prefill``, reply history (``extra_messages``), and
  arbitrary body keys for power users.
- :class:`ThinkingMixin`, which strips inline ``<think>`` blocks
  (llama.cpp) from the visible caption.
- The :class:`PredictionContext` reset + reasoning population.

Construction::

    client = await create_client(api_url=..., api_token=..., cache=..., response_logger=..., async_session=...)
    captioner = APICaptioner(
        client=client,
        prompt_template=..., image_quality=..., store_conversation=...,
        reasoning=..., reasoning_effort=..., reasoning_exclude_output=...,
    )

Reasoning: the client separates *native* reasoning (OpenAI
``reasoning_details``/``reasoning_content``, Gemini ``thought:true``)
into ``chunk.reasoning``, which this captioner folds straight into the
``PredictionContext`` — it never enters the visible caption.
*Inline* reasoning (llama.cpp ``<think>`` embedded in content) stays in
``chunk.text`` and is stripped here by :class:`ThinkingMixin`.
"""

import copy
from collections.abc import AsyncGenerator
from enum import Enum
from pathlib import Path
from typing import Any, Literal, cast

from typing_extensions import override

from yadc.core import DatasetImage, logging
from yadc.core.captioner import ROLE_ASSISTANT, ROLE_USER, Captioner, ReplyRound
from yadc.core.prediction import PredictionContext
from yadc.llm import BaseLLMClient, ImageUrl, ImageUrlPart, Message, TextPart
from yadc.llm.constants import DEFAULT_MODELS_CACHE_TTL_SECONDS
from yadc.llm.gemini import GeminiLLMClient
from yadc.llm.openai_compatible import OpenAICompatibleLLMClient

from .utils.thinking import ThinkingMixin

_logger = logging.get_logger(__name__)


class APITypes(str, Enum):
    GEMINI = "gemini"
    OPENAI = "openai"
    OPENROUTER = "openrouter"
    LLAMACPP = "llamacpp"
    KOBOLDCPP = "koboldcpp"
    VLLM = "vllm"
    OLLAMA = "ollama"

    @override
    def __str__(self) -> str:
        return self.value


# Maps the composed client's class to its APITypes label (the ``api_type``
# property) and to per-backend image encode limits. Backend → label/limits
# mapping kept in one place; resolved via the client's MRO so a (hypothetical)
# subclass still picks up its backend's entry.
_CLIENT_API_TYPE: dict[type[BaseLLMClient], APITypes] = {}

# Per-backend image encode limits. Cloud backends are stricter (10 MiB /
# 1024²) than local OpenAI-compatible servers (25 MiB / 1536²); Gemini is
# its own thing (5 MiB / 2048²). Resolved via the client's MRO so a
# (hypothetical) subclass still picks up its backend's entry.
_IMAGE_LIMITS: dict[type[BaseLLMClient], tuple[tuple[int, int], int]] = {}

_DEFAULT_IMAGE_LIMITS: tuple[tuple[int, int], int] = ((1536, 1536), 25 * 1024 * 1024)


def _image_limits_for(client: BaseLLMClient) -> tuple[tuple[int, int], int]:
    for cls in type(client).__mro__:
        if cls in _IMAGE_LIMITS:
            return _IMAGE_LIMITS[cls]
    return _DEFAULT_IMAGE_LIMITS


def _api_type_for(client: BaseLLMClient) -> APITypes:
    for cls in type(client).__mro__:
        if cls in _CLIENT_API_TYPE:
            return _CLIENT_API_TYPE[cls]
    return APITypes.OPENAI


# Register each client class with its label + image limits. Imported here (after
# the classes exist) so the captioners.api package stays the only composition root.
from yadc.llm.koboldcpp import KoboldcppLLMClient  # noqa: E402
from yadc.llm.llamacpp import LlamacppLLMClient  # noqa: E402
from yadc.llm.ollama import OllamaLLMClient  # noqa: E402
from yadc.llm.openai import OpenAILLMClient  # noqa: E402
from yadc.llm.openrouter import OpenRouterLLMClient  # noqa: E402
from yadc.llm.vllm import VllmLLMClient  # noqa: E402

_CLOUD_LIMITS: tuple[tuple[int, int], int] = ((1024, 1024), 10 * 1024 * 1024)
_LOCAL_LIMITS: tuple[tuple[int, int], int] = ((1536, 1536), 25 * 1024 * 1024)

_CLIENT_API_TYPE.update(
    {
        OpenAILLMClient: APITypes.OPENAI,
        OpenRouterLLMClient: APITypes.OPENROUTER,
        GeminiLLMClient: APITypes.GEMINI,
        LlamacppLLMClient: APITypes.LLAMACPP,
        KoboldcppLLMClient: APITypes.KOBOLDCPP,
        VllmLLMClient: APITypes.VLLM,
        OllamaLLMClient: APITypes.OLLAMA,
    }
)
_IMAGE_LIMITS.update(
    {
        OpenAILLMClient: _CLOUD_LIMITS,
        OpenRouterLLMClient: _CLOUD_LIMITS,
        GeminiLLMClient: ((2048, 2048), 5 * 1024 * 1024),
        LlamacppLLMClient: _LOCAL_LIMITS,
        KoboldcppLLMClient: _LOCAL_LIMITS,
        VllmLLMClient: _LOCAL_LIMITS,
        OllamaLLMClient: _LOCAL_LIMITS,
    }
)


class APICaptioner(Captioner, ThinkingMixin):
    """Captioner that delegates prediction to a composed :class:`BaseLLMClient`.

    Takes a ``client`` (:class:`BaseLLMClient`) plus the standard
    :class:`Captioner` kwargs (``prompt_template``, ``jinja_args``, …) and
    inherits everything else (Jinja rendering, image encoding, thinking
    strip, ``PredictionContext`` wiring) from the parent classes.
    """

    def __init__(self, *, client: BaseLLMClient, **kwargs: Any):
        Captioner.__init__(self, **kwargs)
        ThinkingMixin.__init__(self, **kwargs)

        self._client: BaseLLMClient = client

        self._image_quality: str = kwargs.pop("image_quality", "auto")
        self._store_conversation: bool = kwargs.pop("store_conversation", False)

        self._reasoning: bool = kwargs.pop("reasoning", False)
        self._reasoning_effort: str = kwargs.pop("reasoning_effort", "low")
        self._reasoning_exclude_output: bool = kwargs.pop("reasoning_exclude_output", True)

        _logger.info("API set to %s.", self.api_type)

    @property
    def client(self) -> BaseLLMClient:
        """The composed LLM client (load/list/predict happen through here)."""
        return self._client

    @property
    def api_type(self) -> APITypes:
        """Backend label derived from the client class (e.g. for prefill gating)."""
        return _api_type_for(self._client)

    # ---- proxies to the client (keep the runner / CLI untouched) ----

    @override
    async def load_model(self, model_repo: str, **kwargs: Any) -> None:
        await self._client.load_model(model_repo, **kwargs)

    @override
    def unload_model(self) -> None:
        self._client.unload_model()

    @override
    def offload_model(self) -> None:
        self._client.offload_model()

    def log_usage(self) -> None:
        self._client.log_usage()

    async def list_models(self, cache_ttl: float | None = DEFAULT_MODELS_CACHE_TTL_SECONDS) -> list[str]:
        return await self._client.list_models(cache_ttl=cache_ttl)

    # ---- prediction ----

    @staticmethod
    def _before_predict(kwargs: dict[str, Any]) -> None:
        ctx = kwargs.get("prediction_context", None)
        if ctx is not None:
            assert isinstance(ctx, PredictionContext), f"prediction_context must be a PredictionContext, got {type(ctx)}"
            ctx.reasoning = None
            ctx.reasoning_summary = None
            ctx.reasoning_encrypted = None

    def _reasoning_effort_for_call(self) -> Literal["low", "medium", "high"] | None:
        """The ``reasoning`` kwarg to pass the client: effort level, or ``None`` when reasoning is off."""
        if not self._reasoning:
            return None
        # Validated upstream by the config (``thinking_effort`` ∈ low/medium/high);
        # the client takes a ``Literal``, so narrow it.
        return cast(Literal["low", "medium", "high"], self._reasoning_effort)

    def _build_messages(self, image: DatasetImage, **kwargs: Any) -> tuple[list[Message], dict[str, Any], str, int]:
        """Render prompts + encode the image into chat messages.

        Returns ``(messages, body_opts, assistant_prefill, max_tokens)``:

        - *messages*: system + user (image) + reply history + an optional
          trailing assistant prefill message.
        - *body_opts*: advanced-settings keys to merge into the request body
          (the client applies them), plus backend-specific captioner config
          (``store`` for OpenAI, ``image_quality`` for Gemini).
        - *assistant_prefill*: the prefill text, returned separately so the
          caller can prepend it to the model's continuation.
        """
        # Pop captioner-internal kwargs before rendering so they don't leak
        # into the Jinja context or image encoder as unknown vars.
        conversation_overrides: dict[str, Any] = kwargs.pop("conversation_overrides", {}) or {}
        extra_messages: list[ReplyRound] | None = kwargs.pop("extra_messages", None)
        max_tokens: int = kwargs.pop("max_new_tokens", 512)
        # The runner also passes `prefill` directly; the authoritative source is
        # conversation_overrides["assistant_prefill"], so ignore the bare kwarg.
        kwargs.pop("prefill", None)

        system_prompt, user_prompt = self.prompts_from_image(image, **kwargs)

        max_image_size, max_image_encoded_size = _image_limits_for(self._client)
        mime_type, encoded_image = self._encode_image(image, max_image_size=max_image_size, max_image_encoded_size=max_image_encoded_size, **kwargs)

        try:
            assert isinstance(conversation_overrides, dict), f"bad value for conversation_overrides; expected dict, got {type(conversation_overrides)}"
            conversation_overrides = copy.deepcopy(conversation_overrides)

            # These are built by the client from the messages/body — never let
            # advanced settings clobber them.
            for _protected in ("stream", "store", "messages", "contents", "model", "system_instruction"):
                conversation_overrides.pop(_protected, None)

            system_role: str = conversation_overrides.pop("system_role", None) or "system"
            user_role: str = conversation_overrides.pop("user_role", None) or "user"
            assistant_role: str = conversation_overrides.pop("assistant_role", None) or "assistant"
            assert isinstance(system_role, str), f"bad value for advanced settings system_role; expected str, got {type(system_role)}"
            assert isinstance(user_role, str), f"bad value for advanced settings user_role; expected str, got {type(user_role)}"
            assert isinstance(assistant_role, str), f"bad value for advanced settings assistant_role; expected str, got {type(assistant_role)}"

            assistant_prefill: str = conversation_overrides.pop("assistant_prefill", "")
            assert isinstance(assistant_prefill, str), f"bad value for advanced settings assistant_prefill; expected str, got {type(assistant_prefill)}"

            assert extra_messages is None or isinstance(extra_messages, list), f"bad value for extra_messages; expected list, got {type(extra_messages)}"
        except AssertionError as e:
            raise ValueError(e) from e

        messages: list[Message] = [
            Message(role=system_role, content=[TextPart(text=system_prompt)]),
            Message(
                role=user_role,
                content=[
                    ImageUrlPart(
                        image_url=ImageUrl(url=f"data:{mime_type};base64,{encoded_image}", detail=cast(Literal["auto", "low", "high"], self._image_quality))
                    ),
                    TextPart(text=user_prompt),
                ],
            ),
        ]

        if extra_messages:
            for msg in extra_messages:
                assert isinstance(msg, ReplyRound), f"extra_messages must be ReplyRound instances, got {type(msg)}"
                assert msg.role in (ROLE_USER, ROLE_ASSISTANT), f"extra_messages role must be ROLE_USER or ROLE_ASSISTANT, got {msg.role!r}"

                role = assistant_role if msg.role == ROLE_ASSISTANT else user_role
                messages.append(Message(role=role, content=msg.content, reasoning=msg.reasoning or None, reasoning_encrypted=msg.reasoning_encrypted))

        if assistant_prefill:
            # Trailing assistant message — the API continues from here. The
            # caller prepends `assistant_prefill` to the model's output below.
            messages.append(Message(role=assistant_role, content=assistant_prefill))

        # Body opts: remaining advanced-settings keys (power-user body tweaks)
        # plus backend-specific captioner config.
        opts: dict[str, Any] = dict(conversation_overrides)
        if isinstance(self._client, OpenAICompatibleLLMClient):
            opts["store"] = self._store_conversation
        elif isinstance(self._client, GeminiLLMClient):
            # Gemini maps image_quality to mediaResolution; OpenAI carries it
            # via the image_url `detail` field instead.
            opts["image_quality"] = self._image_quality

        return messages, opts, assistant_prefill, max_tokens

    @override
    async def predict(self, image: DatasetImage, **kwargs: Any) -> str:
        self._before_predict(kwargs)
        ctx: PredictionContext | None = kwargs.get("prediction_context", None)

        messages, opts, assistant_prefill, max_tokens = self._build_messages(image, **kwargs)

        stream = await self._client.predict_next_message_stream(
            messages,
            max_tokens=max_tokens,
            reasoning=self._reasoning_effort_for_call(),
            reasoning_exclude=self._reasoning_exclude_output,
            capture_name=Path(image.path).stem,
            **opts,
        )
        message = await stream.collect()

        if ctx is not None:
            ctx.reasoning = message.reasoning
            ctx.reasoning_summary = message.reasoning_summary
            ctx.reasoning_encrypted = message.reasoning_encrypted

        content = (assistant_prefill or "") + _message_text(message)
        return self._handle_thinking(content)

    @override
    async def predict_stream(self, image: DatasetImage, **kwargs: Any) -> AsyncGenerator[str, None]:
        self._before_predict(kwargs)
        ctx: PredictionContext | None = kwargs.get("prediction_context", None)

        messages, opts, assistant_prefill, max_tokens = self._build_messages(image, **kwargs)

        stream = await self._client.predict_next_message_stream(
            messages,
            max_tokens=max_tokens,
            reasoning=self._reasoning_effort_for_call(),
            reasoning_exclude=self._reasoning_exclude_output,
            capture_name=Path(image.path).stem,
            **opts,
        )

        async def _raw() -> AsyncGenerator[str, None]:
            if assistant_prefill:
                yield assistant_prefill

            native_thinking_started = False
            native_thinking_buffer = ""

            async for chunk in stream:
                if chunk.reasoning:
                    if not native_thinking_started:
                        _logger.debug("")
                        _logger.info("Thinking...")
                        native_thinking_started = True
                    native_thinking_buffer += chunk.reasoning
                    if ctx is not None:
                        ctx.reasoning = (ctx.reasoning or "") + chunk.reasoning
                if chunk.reasoning_summary and ctx is not None:
                    ctx.reasoning_summary = (ctx.reasoning_summary or "") + chunk.reasoning_summary
                if chunk.reasoning_encrypted and ctx is not None:
                    ctx.reasoning_encrypted = (ctx.reasoning_encrypted or []) + chunk.reasoning_encrypted

                if chunk.text:
                    if native_thinking_started:
                        _log_thinking_block(native_thinking_buffer)
                        _logger.info("Thinking done.\n")
                        native_thinking_started = False
                        native_thinking_buffer = ""
                    yield chunk.text

            if native_thinking_started and native_thinking_buffer:
                # Stream ended inside the reasoning block (no content followed).
                _log_thinking_block(native_thinking_buffer)
                _logger.warning("Warning: Thinking was not finished.")

        async for token in self._handle_thinking_streaming_async(_raw()):
            yield token


def _log_thinking_block(buffer: str) -> None:
    for line in buffer.strip().splitlines():
        _logger.info("> %s", line)


def _message_text(message: Message) -> str:
    """Flatten a message's content to a string (the client always accumulates str)."""
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(p.text for p in content if isinstance(p, TextPart))
