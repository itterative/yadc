"""Shared fixtures for the ``yadc.llm`` client tests.

Reuses :class:`tests.captioners.api.conftest.MockAsyncSession` (extended to
support :meth:`AsyncSession.open_stream`) and the real API response
fixtures under ``tests/captioners/api/test_data/`` so the client is
exercised against the same captured payloads as the captioners.
"""

import pathlib

import pytest

from tests.captioners.api.conftest import MockAsyncSession

_TEST_DATA = pathlib.Path(__file__).parent.parent / "captioners" / "api" / "test_data"


@pytest.fixture
def load_test_data():
    def _load_test_data(case: str) -> str:
        return (_TEST_DATA / case).read_text()

    return _load_test_data


@pytest.fixture
def mock_async_session():
    return MockAsyncSession()


@pytest.fixture
def make_session():
    """Factory for a fresh :class:`MockAsyncSession` (one route set per test)."""

    def _make() -> MockAsyncSession:
        return MockAsyncSession()

    return _make
