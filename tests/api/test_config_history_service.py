"""Tests for ConfigHistoryService.list_history — opaque ``next`` cursor decoding."""

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from yadc.api.services.config_history import ConfigHistoryPage, ConfigHistoryService


@dataclass
class FakeEntry:
    """Minimal stand-in for ConfigHistoryEntry used in these tests."""

    id: int
    dataset_name: str = "alpha"
    content: str = ""
    created_t: float = 0.0


@pytest.fixture
def service(logging_factory):
    """ConfigHistoryService with a mocked repo.

    Uses the real constructor so the service is fully initialized, but
    substitutes a ``MagicMock`` repo so we can drive return values and
    assert on calls without touching the real DB.
    """
    mock_repo = MagicMock()
    mock_repo.list_history.return_value = []
    return ConfigHistoryService(
        db=MagicMock(),
        logging=logging_factory,
        repo=mock_repo,
        datasets=MagicMock(),
    )


class TestListHistory:
    """``list_history`` — opaque ``next`` cursor + ``ConfigHistoryPage`` wrapper."""

    def test_first_page_passes_unbounded_cursor(self, service):
        """No ``next`` token means "first page" — repo gets an
        effectively-unbounded ``before_id`` (max int) so all rows
        pass the ``id < ?`` filter."""
        page = service.list_history("alpha", limit=10)

        service._repo.list_history.assert_called_once()
        _, kwargs = service._repo.list_history.call_args
        # limit+1 for the "is there a next page" probe
        assert kwargs["limit"] == 11
        assert kwargs["before_id"] == 2**63 - 1
        assert page.entries == []
        assert page.next_token is None

    def test_next_token_decoded_to_before_id(self, service):
        """An opaque ``next`` token is decoded into the internal
        ``before_id`` cursor; the client never sees the encoding."""
        service.list_history("alpha", limit=10, next="42")

        _, kwargs = service._repo.list_history.call_args
        assert kwargs["before_id"] == 42

    def test_next_token_emitted_when_more_pages(self, service):
        """When the repo returns ``limit+1`` rows, the service
        trims to ``limit`` and emits a ``next_token`` for the
        next page (authoritative: server knows when there are
        more pages, not the heuristic ``len === limit``)."""
        # Simulate limit+1 rows: 5 asked, 6 returned → there's a
        # next page.
        rows = [FakeEntry(id=10 - i) for i in range(6)]
        service._repo.list_history.return_value = rows

        page = service.list_history("alpha", limit=5)

        # Trimmed to the first 5
        assert len(page.entries) == 5
        assert [e.id for e in page.entries] == [10, 9, 8, 7, 6]
        # Token = smallest id on the returned page = last id in DESC
        assert page.next_token == "6"

    def test_no_next_token_on_last_page(self, service):
        """When the repo returns ``<= limit`` rows, ``next_token``
        is ``None`` (this is the last page)."""
        # Asked for 5, got 3 (less than limit) → no more pages
        rows = [FakeEntry(id=10 - i) for i in range(3)]
        service._repo.list_history.return_value = rows

        page = service.list_history("alpha", limit=5)

        assert len(page.entries) == 3
        assert page.next_token is None

    def test_empty_token_means_first_page(self, service):
        """An empty ``next`` string is treated as "first page"."""
        service.list_history("alpha", limit=10, next="")

        _, kwargs = service._repo.list_history.call_args
        assert kwargs["before_id"] == 2**63 - 1

    def test_returns_config_history_page(self, service):
        """The return type is a :class:`ConfigHistoryPage` (not a
        bare list) so the controller can serialize it as the
        response wrapper without re-wrapping."""
        service._repo.list_history.return_value = [FakeEntry(id=1)]

        page = service.list_history("alpha", limit=10)
        assert isinstance(page, ConfigHistoryPage)
        assert hasattr(page, "entries")
        assert hasattr(page, "next_token")
