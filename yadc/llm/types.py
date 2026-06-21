"""Core message and stream types for the ``yadc.llm`` client layer.

These are transport-agnostic: :class:`Message` follows the OpenAI
chat-completions shape (``role`` + ``content``, where ``content`` is a
plain string for text or a list of typed parts for multimodal). Each
backend client (``OpenAICompatibleLLMClient``, ``GeminiLLMClient``)
translates to and from its own native wire format.

:class:`MessageStream` wraps an already-open streaming response returned
by :meth:`BaseLLMClient.predict_next_message_stream`. Because the
response is open at return time, cancellation is a concrete operation
(``await stream.cancel()`` closes it) — no background task or queue is
needed. See its docstring for usage.
"""

from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, ClassVar, Literal

import pydantic

from yadc.core import logging

_logger = logging.get_logger(__name__)


class TextPart(pydantic.BaseModel):
    type: Literal["text"] = "text"
    text: str


class ImageUrl(pydantic.BaseModel):
    url: str
    detail: Literal["auto", "low", "high"] | None = None


class ImageUrlPart(pydantic.BaseModel):
    type: Literal["image_url"] = "image_url"
    image_url: ImageUrl


ContentPart = TextPart | ImageUrlPart


class Message(pydantic.BaseModel):
    """An OpenAI-style chat message.

    ``content`` is a plain ``str`` for text-only messages or a list of
    typed parts (text / image_url) for multimodal ones. Assistant
    messages produced by a model may additionally carry reasoning fields
    populated from the backend's thinking / chain-of-thought output.
    """

    # ``str`` (not a ``Literal``) so captioners can honour the advanced
    # ``system_role`` override (e.g. ``"developer"`` for newer OpenAI models).
    # The client's wire translation treats ``"system"`` / ``"developer"`` as
    # the system message and maps ``"assistant"`` to the backend's model role.
    role: str
    content: str | list[ContentPart]
    reasoning: str | None = None
    reasoning_summary: str | None = None
    reasoning_encrypted: list[dict[str, Any]] | None = None
    model_config: ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(extra="allow")


class StreamChunk(pydantic.BaseModel):
    """A single chunk emitted by :class:`MessageStream`.

    At least one of ``text`` / ``reasoning`` / ``reasoning_summary`` /
    ``reasoning_encrypted`` is set on any chunk that carries data.
    ``reasoning_encrypted`` is typically set once, at end-of-stream.
    """

    text: str | None = None
    reasoning: str | None = None
    reasoning_summary: str | None = None
    reasoning_encrypted: list[dict[str, Any]] | None = None


def apply_chunk(chunk: StreamChunk, message: Message) -> None:
    """Fold a :class:`StreamChunk` into the accumulated ``message`` in place.

    Used by :class:`MessageStream` after each ``__anext__`` so that
    ``stream.message`` always reflects every chunk yielded so far.
    """
    if chunk.text:
        # The accumulated message is always created with a string content by
        # the client, so this stays a str. The isinstance guard is defensive.
        existing = message.content if isinstance(message.content, str) else ""
        message.content = existing + chunk.text
    if chunk.reasoning:
        message.reasoning = (message.reasoning or "") + chunk.reasoning
    if chunk.reasoning_summary:
        message.reasoning_summary = (message.reasoning_summary or "") + chunk.reasoning_summary
    if chunk.reasoning_encrypted:
        message.reasoning_encrypted = (message.reasoning_encrypted or []) + chunk.reasoning_encrypted


class MessageStream:
    """Async-iterable stream of :class:`StreamChunk` that self-accumulates.

    Wraps an already-open streaming response obtained by awaiting
    :meth:`BaseLLMClient.predict_next_message_stream`. Iteration decodes
    backend chunks via the ``chunks`` async iterator and folds each into
    ``.message``. Because the underlying response is open for the lifetime
    of the stream, ``cancel()`` is a concrete close operation.

    Cancellation also propagates the normal asyncio way: cancelling the
    task that is ``async for``-ing the stream raises ``CancelledError``
    inside ``__anext__``. ``cancel()`` is the explicit, non-throwing
    variant for the "client clicked Stop" path.

    Example::

        stream = await client.predict_next_message_stream(messages)
        async for chunk in stream:
            render(chunk)
        message = await stream.collect()  # already drained — returns fast

        # cancel mid-stream
        stream = await client.predict_next_message_stream(messages)
        async for chunk in stream:
            if cancelled:
                await stream.cancel()
                break
    """

    _message: Message
    _chunks: AsyncIterator[StreamChunk]
    _cancel_fn: Callable[[], Awaitable[None]] | None
    _done: bool
    _cancelled: bool

    def __init__(
        self,
        *,
        message: Message,
        chunks: AsyncIterator[StreamChunk],
        cancel_fn: Callable[[], Awaitable[None]] | None = None,
    ):
        self._message = message
        self._chunks = chunks
        self._cancel_fn = cancel_fn
        self._done = False
        self._cancelled = False

    @property
    def message(self) -> Message:
        """The accumulated message so far (partial while iterating)."""
        return self._message

    def __aiter__(self) -> "MessageStream":
        return self

    async def __anext__(self) -> StreamChunk:
        try:
            chunk = await self._chunks.__anext__()
        except StopAsyncIteration:
            self._done = True
            raise
        apply_chunk(chunk, self._message)
        return chunk

    async def collect(self) -> Message:
        """Drain remaining chunks (if any) and return the accumulated message.

        No-op fast path if the stream is already exhausted.
        """
        if not self._done:
            async for _ in self:
                pass
        return self._message

    async def cancel(self) -> None:
        """Close the underlying streaming response. Idempotent.

        Safe to call from the consumer's iteration task (e.g. a Cancel
        button), and safe to call again after the stream has drained
        naturally. The decoder's own cleanup also closes the response, so
        this never double-closes.
        """
        if self._cancelled or self._cancel_fn is None:
            return
        self._cancelled = True
        await self._cancel_fn()
