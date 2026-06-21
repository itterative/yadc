"""Tests for the per-instance ``_api_usage_lock`` on the LLM client.

The lock guards the client's ``_api_usage`` dict so concurrent
``predict_next_message_stream`` calls (e.g. ``CaptioningRunner`` with
``max_concurrent > 1``) don't race on the end-of-response dict write. It
moved from the old ``BaseAPICaptioner`` to :class:`BaseLLMClient` in the
Phase 1b collapse; this file tests it on the client directly.
"""

import asyncio
from unittest.mock import MagicMock

import pytest

from yadc.llm import OpenAILLMClient
from yadc.llm.async_session import AsyncSession
from yadc.llm.base import APIUsage


def _build_client() -> OpenAILLMClient:
    """An ``OpenAILLMClient`` with a mocked session — no real HTTP needed."""
    session = MagicMock(spec=AsyncSession)
    return OpenAILLMClient(api_url="https://api.openai.com/v1", api_token="test-token", async_session=session, warnings=False)


class TestApiUsageLockExists:
    """The lock is created on every client instance."""

    def test_client_has_lock(self):
        client = _build_client()
        assert isinstance(client._api_usage_lock, asyncio.Lock)


class TestApiUsageLockSerialises:
    """Concurrent writers to ``_api_usage`` don't lose entries."""

    @pytest.mark.asyncio
    async def test_concurrent_writes_all_survive(self):
        client = _build_client()

        async def writer(key: str) -> None:
            async with client._api_usage_lock:
                await asyncio.sleep(0)
                client._api_usage[key] = APIUsage(prompt_tokens=2, response_tokens=1, total_tokens=3, thoughts_tokens=0)

        keys = [f"key-{i}" for i in range(50)]
        await asyncio.gather(*(writer(k) for k in keys))

        assert set(client._api_usage.keys()) == set(keys)
        for k in keys:
            assert client._api_usage[k].total_tokens == 3

    @pytest.mark.asyncio
    async def test_lock_is_released_after_use(self):
        """A failed writer doesn't deadlock the lock."""
        client = _build_client()

        async def failing_writer() -> None:
            async with client._api_usage_lock:
                raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            await failing_writer()

        # Lock is still acquirable.
        async with client._api_usage_lock:
            pass
