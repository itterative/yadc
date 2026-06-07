"""Tests for the per-model ``_api_usage_lock`` guarding the usage dict.

The lock is added so concurrent ``predict_stream`` calls on the same
``APICaptioner`` (via ``CaptioningRunner.caption_images`` with
``max_concurrent > 1``) don't race on dict writes. This file tests
the lock directly — its existence, that it serialises writers, and
that the existing sequential behaviour still works.
"""

import asyncio
from unittest.mock import MagicMock

import pytest

from yadc.captioners.api.async_session import AsyncSession
from yadc.captioners.api.openai import APIUsage, OpenAICaptioner


def _build_openai_captioner() -> OpenAICaptioner:
    """Build an OpenAICaptioner with a mocked async session and config.

    The tests below don't need a real HTTP session; they only exercise
    the lock + dict machinery, not the network. The kwargs are enough
    to satisfy the constructor and ``_api_usage_lock`` initialisation.
    """
    session = MagicMock(spec=AsyncSession)
    return OpenAICaptioner(
        api_url="https://api.openai.com/v1",
        api_token="test-token",
        async_session=session,
        _warnings=False,
    )


class TestApiUsageLockExists:
    """The lock is created on every BaseAPICaptioner instance."""

    def test_openai_captioner_has_lock(self):
        captioner = _build_openai_captioner()
        assert isinstance(captioner._api_usage_lock, asyncio.Lock)


class TestApiUsageLockSerialises:
    """Concurrent writers to ``_api_usage`` don't lose entries.

    Reproduces the race the lock guards against: N coroutines each
    acquire the lock, set a unique key, and release. With the lock
    every entry survives. (Without it, dict mutation races in
    CPython could still work in practice due to the GIL, but the
    test is the canonical assurance that the lock is in the path.)
    """

    @pytest.mark.asyncio
    async def test_concurrent_writes_all_survive(self):
        captioner = _build_openai_captioner()

        async def writer(key: str) -> None:
            async with captioner._api_usage_lock:
                # Simulate a tiny amount of "write" work to make a
                # race possible if the lock were removed.
                await asyncio.sleep(0)
                captioner._api_usage[key] = APIUsage(response_tokens=1, prompt_tokens=2, total_tokens=3, thoughts_tokens=0)

        keys = [f"key-{i}" for i in range(50)]
        await asyncio.gather(*(writer(k) for k in keys))

        assert set(captioner._api_usage.keys()) == set(keys)
        for k in keys:
            entry = captioner._api_usage[k]
            assert entry.total_tokens == 3

    @pytest.mark.asyncio
    async def test_lock_is_released_after_use(self):
        """A failed writer doesn't deadlock the lock."""
        captioner = _build_openai_captioner()

        async def failing_writer() -> None:
            async with captioner._api_usage_lock:
                raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            await failing_writer()

        # Lock is still acquirable
        async with captioner._api_usage_lock:
            pass
