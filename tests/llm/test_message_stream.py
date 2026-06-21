"""``MessageStream`` — accumulation, collect, cancel, and cancellation propagation."""

import asyncio

import pytest

from yadc.llm.types import Message, MessageStream, StreamChunk, apply_chunk


def _message() -> Message:
    return Message(role="assistant", content="")


async def _gen(*chunks: StreamChunk):
    for c in chunks:
        yield c


class TestApplyChunk:
    def test_text_appends_to_content(self):
        m = _message()
        apply_chunk(StreamChunk(text="foo"), m)
        apply_chunk(StreamChunk(text="bar"), m)
        assert m.content == "foobar"

    def test_reasoning_appends(self):
        m = _message()
        apply_chunk(StreamChunk(reasoning="a"), m)
        apply_chunk(StreamChunk(reasoning="b"), m)
        assert m.reasoning == "ab"

    def test_encrypted_extends_list(self):
        m = _message()
        apply_chunk(StreamChunk(reasoning_encrypted=[{"a": 1}]), m)
        apply_chunk(StreamChunk(reasoning_encrypted=[{"b": 2}]), m)
        assert m.reasoning_encrypted == [{"a": 1}, {"b": 2}]


class TestIteration:
    @pytest.mark.asyncio
    async def test_iter_yields_chunks_and_accumulates(self):
        m = _message()
        stream = MessageStream(message=m, chunks=_gen(StreamChunk(text="a"), StreamChunk(text="b")))

        out = [c async for c in stream]
        assert [c.text for c in out] == ["a", "b"]
        assert stream.message.content == "ab"

    @pytest.mark.asyncio
    async def test_collect_drains_and_returns_message(self):
        m = _message()
        stream = MessageStream(message=m, chunks=_gen(StreamChunk(text="a"), StreamChunk(text="b")))

        result = await stream.collect()
        assert result.content == "ab"
        assert result is stream.message

    @pytest.mark.asyncio
    async def test_collect_after_iteration_is_fast_path(self):
        m = _message()
        stream = MessageStream(message=m, chunks=_gen(StreamChunk(text="a")))

        async for _ in stream:
            pass
        # Second collect must not re-consume (already drained).
        result = await stream.collect()
        assert result.content == "a"


class TestCancel:
    @pytest.mark.asyncio
    async def test_cancel_is_idempotent(self):
        calls = 0

        async def _cancel():
            nonlocal calls
            calls += 1

        stream = MessageStream(message=_message(), chunks=_gen(), cancel_fn=_cancel)

        await stream.cancel()
        await stream.cancel()  # second call must not re-invoke cancel_fn
        await stream.cancel()

        assert calls == 1

    @pytest.mark.asyncio
    async def test_cancel_without_fn_is_noop(self):
        stream = MessageStream(message=_message(), chunks=_gen())
        await stream.cancel()  # must not raise


class TestCancellationPropagation:
    """Cancelling the iterating task surfaces CancelledError inside __anext__."""

    @pytest.mark.asyncio
    async def test_task_cancel_raises_in_iteration(self):
        started = asyncio.Event()

        async def slow_chunks():
            yield StreamChunk(text="first")
            started.set()
            await asyncio.sleep(10)  # block until cancelled
            yield StreamChunk(text="never")

        stream = MessageStream(message=_message(), chunks=slow_chunks())

        async def consume():
            async for _ in stream:
                pass

        task = asyncio.create_task(consume())
        await started.wait()
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task
