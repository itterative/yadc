"""Tests for :class:`PromptGenerationService`.

Service-level only — the LLM client is mocked so we don't need real HTTP
or running backends. The streaming chunk mechanics are exercised by
``tests/llm/``; here we verify the service wires the pieces correctly
(load env → resolve model → build the multi-turn meta-conversation →
call client → yield chunks → close client).

Test scope discipline: this file tests the **service's behaviour and
contracts**, not the prompt wording. The meta-conversation structure
(system message + user/assistant turns) is verified end-to-end by
manual runs of the prompt-generator UI; pinning the exact text of every
priming ack or final "go" message in unit tests just means we have to
update the tests every time we tune a prompt. The two system-prompt
tests are the exception — they catch a real bug (the dead-code
``_SYSTEM_PROMPT`` that was never sent to the LLM before Phase 5b
fixed the wire-up) and pin the constant-to-constant equality so a
future refactor can't silently drop the loading code.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from yadc.api.services.prompt_generation import (
    ExamplePair,
    PromptGenerationFocus,
    PromptGenerationRequest,
    PromptGenerationService,
)
from yadc.cmd.envs.keystorage_password import PasswordRequiredError
from yadc.core.user_config import UserConfig, UserConfigApi
from yadc.llm import ImageUrlPart, Message, MessageStream, StreamChunk, TextPart

_PATCH_CMD_ENVS = "yadc.api.services.prompt_generation.service.cmd_envs"
_PATCH_CREATE_CLIENT = "yadc.api.services.prompt_generation.service.create_client"


def _user_config(url: str = "https://api.example.com/v1", token: str = "tk", model: str = "gpt-x") -> UserConfig:
    return UserConfig(api=UserConfigApi(url=url, token=token, model_name=model))


# 1x1 transparent PNG, base64-encoded.
_TINY_PNG = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="


def _example(subject: str = "a cat", caption: str = "a small tabby cat", image: str = _TINY_PNG) -> ExamplePair:
    return ExamplePair(subject=subject, caption=caption, image_data_url=image)


def _request(
    env: str = "default",
    intent: str = "caption cats concisely",
    examples: list[ExamplePair] | None = None,
    focus: PromptGenerationFocus = "both",
    api_model_name: str | None = None,
    template_content: str | None = None,
) -> PromptGenerationRequest:
    return PromptGenerationRequest(
        env=env,
        intent=intent,
        examples=examples or [],
        focus=focus,
        api_model_name=api_model_name,
        template_content=template_content,
    )


def _make_client(
    chunks: list[StreamChunk] | None = None,
    *,
    raise_on_stream: BaseException | None = None,
) -> MagicMock:
    """Build an ``AsyncMock``-backed ``BaseLLMClient`` that yields ``chunks``.

    ``predict_next_message_stream`` is an :class:`AsyncMock` whose return
    value is a real :class:`MessageStream` (so the service can ``async
    for`` over it and tests can introspect the call args). Pass
    ``raise_on_stream`` to make the underlying chunk iterator raise
    mid-stream.
    """
    client = MagicMock()
    client.load_model = AsyncMock()
    client.aclose = AsyncMock()

    if raise_on_stream is not None:

        async def _raising_chunks() -> Any:
            raise raise_on_stream
            yield  # pragma: no cover

        stream = MessageStream(message=Message(role="assistant", content=""), chunks=_raising_chunks())
    else:
        chunks = chunks or []

        async def _iter_chunks() -> Any:
            for c in chunks:
                yield c

        stream = MessageStream(message=Message(role="assistant", content=""), chunks=_iter_chunks())

    client.predict_next_message_stream = AsyncMock(return_value=stream)

    return client


@pytest.fixture
def service(logging_factory) -> PromptGenerationService:
    return PromptGenerationService(logging_factory)


class TestExamplePair:
    """``ExamplePair`` Pydantic model — ``image_data_url`` is required."""

    def test_image_data_url_required(self):
        with pytest.raises(ValueError):
            ExamplePair(subject="a cat", caption="a tabby")  # type: ignore[call-arg]

    def test_image_data_url_accepted(self):
        ex = ExamplePair(subject="a cat", caption="a tabby", image_data_url=_TINY_PNG)
        assert ex.image_data_url == _TINY_PNG
        assert ex.subject == "a cat"
        assert ex.caption == "a tabby"


class TestPromptGenerationRequest:
    """``PromptGenerationRequest`` body shape + validation."""

    def test_defaults(self):
        req = PromptGenerationRequest(env="default", intent="caption cats")
        assert req.examples == []
        assert req.focus == "both"
        assert req.api_model_name is None
        assert req.template_content is None

    def test_extras_rejected(self):
        with pytest.raises(ValueError):
            PromptGenerationRequest.model_validate({"env": "default", "intent": "x", "unknown_field": "y"})

    def test_template_content_must_be_non_empty(self):
        with pytest.raises(ValueError, match="template_content"):
            PromptGenerationRequest(env="default", intent="x", template_content="")


class TestBuildMessages:
    """Structural contracts of the message list — what the LLM client sees.

    These are intentionally minimal: one wire-up test for the system
    message and one structural test for the multimodal example format.
    Everything else (priming acks, final "go" message, 0 vs N example
    branching) is verified end-to-end in the prompt-generator UI, not
    here, because pinning the exact wording in unit tests just means
    we have to update the tests every time we tune a prompt.
    """

    @pytest.mark.asyncio
    async def test_system_message_is_first(self, service):
        """The system prompt is prepended to the message list at index 0.

        Pinned to the constant loaded by ``service.py`` from
        ``prompts/generate.txt`` so a future refactor can't silently
        drop the loading code (this was a real bug before Phase 5b —
        the old ``_SYSTEM_PROMPT`` was dead code).
        """
        from yadc.api.services.prompt_generation.service import _GENERATE_SYSTEM_PROMPT

        client = _make_client([StreamChunk(text="x")])

        with (
            patch(_PATCH_CMD_ENVS) as patched_cmd_envs,
            patch(_PATCH_CREATE_CLIENT, return_value=client),
        ):
            patched_cmd_envs.load_env.return_value = _user_config()

            async for _ in service.generate(request=_request()):
                pass

        messages: list[Message] = client.predict_next_message_stream.call_args.args[0]
        assert messages[0].role == "system"
        assert messages[0].content == _GENERATE_SYSTEM_PROMPT

    @pytest.mark.asyncio
    async def test_example_user_turn_is_multimodal(self, service):
        """Each example is sent as a user turn with a text part (subject + caption) and an image part.

        If this ever breaks, the LLM stops seeing the example images.
        The exact label prefix ("Example 1/N") is not asserted — that's
        a wording concern, not a structural one.
        """
        client = _make_client([StreamChunk(text="x")])
        ex = _example(subject="a cat", caption="a small tabby")

        with (
            patch(_PATCH_CMD_ENVS) as patched_cmd_envs,
            patch(_PATCH_CREATE_CLIENT, return_value=client),
        ):
            patched_cmd_envs.load_env.return_value = _user_config()

            async for _ in service.generate(request=_request(examples=[ex])):
                pass

        messages: list[Message] = client.predict_next_message_stream.call_args.args[0]
        # The example turn is the only ``user`` turn with a list content
        # (the system message is a string, the intro/final-go are strings).
        example_turns = [m for m in messages if m.role == "user" and isinstance(m.content, list)]
        assert len(example_turns) == 1
        parts = example_turns[0].content
        assert len(parts) == 2
        assert isinstance(parts[0], TextPart)
        assert "a cat" in parts[0].text
        assert "a small tabby" in parts[0].text
        assert isinstance(parts[1], ImageUrlPart)
        assert parts[1].image_url.url == _TINY_PNG


class TestRefineMode:
    """Mode branching: refine mode (when ``template_content`` is set) uses a different system prompt.

    The structure of the meta-conversation in refine mode is verified
    end-to-end in the prompt-generator UI, not here, for the same
    reason as :class:`TestBuildMessages` — pinning the exact wording
    of the refine priming acks or the existing-template user turn
    prefix would mean updating the tests every time we tune
    ``refine.txt``.
    """

    @pytest.mark.asyncio
    async def test_uses_refine_system_prompt(self, service):
        """When ``template_content`` is set, the system message is the refine prompt (not the generate one)."""
        from yadc.api.services.prompt_generation.service import (
            _GENERATE_SYSTEM_PROMPT,
            _REFINE_SYSTEM_PROMPT,
        )

        client = _make_client([StreamChunk(text="x")])

        with (
            patch(_PATCH_CMD_ENVS) as patched_cmd_envs,
            patch(_PATCH_CREATE_CLIENT, return_value=client),
        ):
            patched_cmd_envs.load_env.return_value = _user_config()

            async for _ in service.generate(request=_request(template_content="existing template")):
                pass

        messages: list[Message] = client.predict_next_message_stream.call_args.args[0]
        assert messages[0].role == "system"
        assert messages[0].content == _REFINE_SYSTEM_PROMPT
        # Sanity check: the two prompts are actually different. Catches
        # a copy-paste mistake where someone makes ``refine.txt``
        # identical to ``generate.txt``.
        assert messages[0].content != _GENERATE_SYSTEM_PROMPT


class TestGenerate:
    """``PromptGenerationService.generate`` — end-to-end wiring with a mocked client."""

    @pytest.mark.asyncio
    async def test_loads_env_resolves_model_calls_client_yields_chunks(self, service):
        """Happy path: load env → resolve model → call client → yield every chunk → close client."""
        chunks = [
            StreamChunk(text="{% set "),
            StreamChunk(text="system_prompt %}hi{% endset %}"),
            StreamChunk(reasoning="thinking..."),
            StreamChunk(text="{% set user_prompt %}body{% endset %}"),
        ]
        client = _make_client(chunks)
        request = _request()

        with (
            patch(_PATCH_CMD_ENVS) as patched_cmd_envs,
            patch(_PATCH_CREATE_CLIENT, return_value=client),
        ):
            patched_cmd_envs.load_env.return_value = _user_config()

            yielded: list[StreamChunk] = []
            async for chunk in service.generate(request=request, password="pw"):
                yielded.append(chunk)

        # Env loaded with the password.
        patched_cmd_envs.load_env.assert_called_once_with("default", password="pw")
        # Model loaded with the resolved name (no override → use env default).
        client.load_model.assert_awaited_once_with("gpt-x")
        # Chunks yielded in order.
        assert yielded == chunks
        # Client closed even on natural exhaustion.
        client.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_request_api_model_name_overrides_env_default(self, service):
        """``request.api_model_name`` takes precedence over the env's configured model."""
        client = _make_client([StreamChunk(text="x")])
        request = _request(api_model_name="override-model")

        with (
            patch(_PATCH_CMD_ENVS) as patched_cmd_envs,
            patch(_PATCH_CREATE_CLIENT, return_value=client),
        ):
            patched_cmd_envs.load_env.return_value = _user_config(model="env-default-model")

            async for _ in service.generate(request=request):
                pass

        client.load_model.assert_awaited_once_with("override-model")

    @pytest.mark.asyncio
    async def test_reasoning_disabled(self, service):
        """Reasoning is explicitly ``None`` — the artifact is the template body, thinking wastes tokens."""
        client = _make_client([StreamChunk(text="x")])

        with (
            patch(_PATCH_CMD_ENVS) as patched_cmd_envs,
            patch(_PATCH_CREATE_CLIENT, return_value=client),
        ):
            patched_cmd_envs.load_env.return_value = _user_config()

            async for _ in service.generate(request=_request()):
                pass

        kwargs = client.predict_next_message_stream.call_args.kwargs
        assert kwargs.get("reasoning") is None

    @pytest.mark.asyncio
    async def test_client_aclosed_even_when_stream_raises(self, service):
        """``client.aclose()`` is called from the finally block on stream errors."""
        client = _make_client(raise_on_stream=ValueError("upstream blew up"))

        with (
            patch(_PATCH_CMD_ENVS) as patched_cmd_envs,
            patch(_PATCH_CREATE_CLIENT, return_value=client),
        ):
            patched_cmd_envs.load_env.return_value = _user_config()

            with pytest.raises(ValueError, match="upstream blew up"):
                async for _ in service.generate(request=_request()):
                    pass

        client.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_client_aclosed_even_when_load_model_fails(self, service):
        """``client.aclose()`` runs even when ``load_model`` raises (the client was created)."""
        client = _make_client([StreamChunk(text="x")])
        client.load_model.side_effect = ValueError("model not found")

        with (
            patch(_PATCH_CMD_ENVS) as patched_cmd_envs,
            patch(_PATCH_CREATE_CLIENT, return_value=client),
        ):
            patched_cmd_envs.load_env.return_value = _user_config()

            with pytest.raises(ValueError, match="model not found"):
                async for _ in service.generate(request=_request()):
                    pass

        client.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_password_required_error_propagates(self, service):
        """``PasswordRequiredError`` from ``cmd_envs.load_env`` propagates untouched
        (the controller catches it for 403 mapping)."""
        with patch(_PATCH_CMD_ENVS) as patched_cmd_envs:
            patched_cmd_envs.load_env.side_effect = PasswordRequiredError("locked")

            with pytest.raises(PasswordRequiredError):
                async for _ in service.generate(request=_request(), password="wrong"):
                    pass

    @pytest.mark.asyncio
    async def test_missing_api_url_raises_value_error(self, service):
        """No api_url on the resolved env → ``ValueError``."""
        client = _make_client([StreamChunk(text="x")])

        with (
            patch(_PATCH_CMD_ENVS) as patched_cmd_envs,
            patch(_PATCH_CREATE_CLIENT, return_value=client),
        ):
            patched_cmd_envs.load_env.return_value = _user_config(url="")

            with pytest.raises(ValueError, match="has no api_url"):
                async for _ in service.generate(request=_request()):
                    pass

    @pytest.mark.asyncio
    async def test_missing_model_name_raises_value_error(self, service):
        """No api_model_name on env AND no override → ``ValueError``."""
        client = _make_client([StreamChunk(text="x")])

        with (
            patch(_PATCH_CMD_ENVS) as patched_cmd_envs,
            patch(_PATCH_CREATE_CLIENT, return_value=client),
        ):
            patched_cmd_envs.load_env.return_value = _user_config(model="")

            with pytest.raises(ValueError, match="has no api_model_name"):
                async for _ in service.generate(request=_request()):
                    pass

    @pytest.mark.asyncio
    async def test_yields_chunks_unchanged(self, service):
        """The service yields each ``StreamChunk`` from the client untouched (no transformation)."""
        chunks = [
            StreamChunk(text="a"),
            StreamChunk(reasoning="think"),
            StreamChunk(text="b", reasoning="more"),
            StreamChunk(text="c", reasoning_summary="sum"),
            StreamChunk(text="d", reasoning_encrypted=[{"type": "reasoning.encrypted", "data": "x"}]),
        ]
        client = _make_client(chunks)

        with (
            patch(_PATCH_CMD_ENVS) as patched_cmd_envs,
            patch(_PATCH_CREATE_CLIENT, return_value=client),
        ):
            patched_cmd_envs.load_env.return_value = _user_config()

            yielded = [c async for c in service.generate(request=_request())]

        assert yielded == chunks
