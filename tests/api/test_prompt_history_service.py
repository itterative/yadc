"""Tests for :class:`PromptHistoryService`.

Mirrors :class:`ConfigHistoryService`'s test pattern: mocked repo
for orchestration (prune policy, pagination), real repo for the
base64 ↔ BLOB translation round-trip (the value-add, so the
fixture-driven approach is more meaningful than a mock).
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from yadc.api.modules.db_connection_factory import DBConnectionFactory
from yadc.api.modules.logging_factory import LoggingFactory
from yadc.api.services.prompt_generation import ExamplePair
from yadc.api.services.prompt_history import (
    INTENT_PREVIEW_MAX_CHARS,
    PROMPT_HISTORY_MAX_ENTRIES,
    PromptHistoryListItem,
    PromptHistoryPage,
    PromptHistorySaveRequest,
    PromptHistoryService,
)
from yadc.api.services.prompt_history_repository import PromptExample, PromptHistoryRepository

# --- Test doubles ---


@dataclass
class FakeEntry:
    """Minimal stand-in for :class:`PromptHistoryEntry` used in mocked-repo tests."""

    id: int
    mode: str = "generate"
    intent: str = "Caption this image."
    focus: str = "both"
    template_content: str | None = None
    created_t: float = 0.0


@pytest.fixture
def service_with_mock(logging_factory: LoggingFactory):
    """PromptHistoryService with a mocked repo — for orchestration tests."""
    mock_repo = MagicMock()
    return PromptHistoryService(
        db=MagicMock(),
        logging=logging_factory,
        repo=mock_repo,
    )


@pytest.fixture
def service_with_real(
    db_connection_factory: DBConnectionFactory,
    logging_factory: LoggingFactory,
    repo: PromptHistoryRepository,
) -> PromptHistoryService:
    """PromptHistoryService with the real repo + real DB — for decode round-trip tests."""
    return PromptHistoryService(
        db=db_connection_factory,
        logging=logging_factory,
        repo=repo,
    )


@pytest.fixture
def repo(
    db_connection_factory: DBConnectionFactory,
    logging_factory: LoggingFactory,
) -> PromptHistoryRepository:
    return PromptHistoryRepository(db=db_connection_factory, logging=logging_factory)


# --- save_entry ---


class TestSaveEntry:
    def test_inserts_examples_and_prunes_in_one_transaction(self, service_with_mock):
        """Save calls ``insert_entry``, ``insert_examples``, and ``prune_old`` on the same transaction."""
        row_id = service_with_mock.save_entry(PromptHistorySaveRequest(mode="generate", intent="x", focus="both"))

        service_with_mock._repo.insert_entry.assert_called_once()
        service_with_mock._repo.insert_examples.assert_called_once()
        service_with_mock._repo.prune_old.assert_called_once_with(PROMPT_HISTORY_MAX_ENTRIES)
        assert row_id is not None

    def test_passes_template_content_through(self, service_with_mock):
        """Refine-mode ``template_content`` is forwarded to the repo unchanged."""
        service_with_mock.save_entry(
            PromptHistorySaveRequest(
                mode="refine",
                intent="refine this",
                focus="system",
                template_content="{% set system_prompt %}\nfoo\n{% endset %}",
            )
        )

        _, kwargs = service_with_mock._repo.insert_entry.call_args
        assert kwargs["mode"] == "refine"
        assert kwargs["template_content"] == "{% set system_prompt %}\nfoo\n{% endset %}"

    def test_passes_null_template_content_for_generate(self, service_with_mock):
        """Generate-mode ``template_content`` is ``None``."""
        service_with_mock.save_entry(PromptHistorySaveRequest(mode="generate", intent="x", focus="both"))

        _, kwargs = service_with_mock._repo.insert_entry.call_args
        assert kwargs["template_content"] is None

    def test_decodes_examples_to_bLOBs(self, service_with_mock):
        """Each ``image_data_url`` is decoded into ``(mime, bytes)`` before storage."""
        import base64 as _b64

        service_with_mock._repo.insert_entry.return_value = 42

        examples = [
            ExamplePair(
                subject="a cat",
                caption="a tabby",
                image_data_url=f"data:image/png;base64,{_b64.b64encode(bytes([0, 0, 0])).decode('ascii')}",
            ),
            ExamplePair(
                subject="a dog",
                caption="a lab",
                image_data_url=f"data:image/jpeg;base64,{_b64.b64encode(bytes([0xFF, 0xD8, 0xFF])).decode('ascii')}",
            ),
        ]
        service_with_mock.save_entry(PromptHistorySaveRequest(mode="generate", intent="x", focus="both", examples=examples))

        # First call to insert_examples (positional) gets the example list.
        args, _ = service_with_mock._repo.insert_examples.call_args
        inserted: list[PromptExample] = args[0]
        assert len(inserted) == 2
        assert inserted[0].mime == "image/png"
        assert inserted[0].data == bytes([0, 0, 0])
        assert inserted[0].subject == "a cat"
        assert inserted[0].caption == "a tabby"
        assert inserted[0].ordinal == 0
        assert inserted[1].mime == "image/jpeg"
        assert inserted[1].data == bytes([0xFF, 0xD8, 0xFF])
        assert inserted[1].ordinal == 1

    def test_silently_skips_malformed_examples(self, service_with_mock):
        """A malformed ``image_data_url`` is dropped (one bad example doesn't block the save)."""
        service_with_mock._repo.insert_entry.return_value = 1
        examples = [
            ExamplePair(subject="good", caption="ok", image_data_url="data:image/png;base64,AAAA"),
            ExamplePair(subject="bad", caption="missing image", image_data_url="not a data url"),
        ]
        service_with_mock.save_entry(PromptHistorySaveRequest(mode="generate", intent="x", focus="both", examples=examples))

        args, _ = service_with_mock._repo.insert_examples.call_args
        inserted: list[PromptExample] = args[0]
        assert len(inserted) == 1
        assert inserted[0].subject == "good"


# --- list_history ---


class TestListHistory:
    def test_first_page_passes_unbounded_cursor(self, service_with_mock):
        service_with_mock._repo.list_entries_with_counts.return_value = []
        service_with_mock.list_history(limit=10)

        _, kwargs = service_with_mock._repo.list_entries_with_counts.call_args
        # limit+1 for the "is there a next page" probe
        assert kwargs["limit"] == 11
        assert kwargs["before_id"] == 2**63 - 1

    def test_next_token_decoded_to_before_id(self, service_with_mock):
        service_with_mock._repo.list_entries_with_counts.return_value = []
        service_with_mock.list_history(limit=10, next_token="42")

        _, kwargs = service_with_mock._repo.list_entries_with_counts.call_args
        assert kwargs["before_id"] == 42

    def test_empty_token_means_first_page(self, service_with_mock):
        service_with_mock._repo.list_entries_with_counts.return_value = []
        service_with_mock.list_history(limit=10, next_token="")

        _, kwargs = service_with_mock._repo.list_entries_with_counts.call_args
        assert kwargs["before_id"] == 2**63 - 1

    def test_returns_page_with_list_items(self, service_with_mock):
        """The response is a :class:`PromptHistoryPage` of :class:`PromptHistoryListItem`\\s."""
        rows = [
            (FakeEntry(id=10, mode="refine", intent="x", focus="system", template_content="t"), 2),
            (FakeEntry(id=9, mode="generate", intent="y", focus="both"), 0),
        ]
        service_with_mock._repo.list_entries_with_counts.return_value = rows

        page = service_with_mock.list_history(limit=10)
        assert isinstance(page, PromptHistoryPage)
        assert len(page.entries) == 2
        assert all(isinstance(e, PromptHistoryListItem) for e in page.entries)
        assert page.entries[0].had_template is True  # template_content is "t"
        assert page.entries[1].had_template is False
        assert page.entries[0].example_count == 2
        assert page.entries[1].example_count == 0

    def test_truncates_long_intent_preview(self, service_with_mock):
        """Intents longer than the preview cap get truncated with an ellipsis."""
        long_intent = "x" * (INTENT_PREVIEW_MAX_CHARS + 50)
        rows = [(FakeEntry(id=1, intent=long_intent), 0)]
        service_with_mock._repo.list_entries_with_counts.return_value = rows

        page = service_with_mock.list_history(limit=10)
        assert len(page.entries[0].intent_preview) <= INTENT_PREVIEW_MAX_CHARS + 1  # +1 for the ellipsis
        assert page.entries[0].intent_preview.endswith("…")

    def test_keeps_short_intent_intact(self, service_with_mock):
        """Short intents pass through unmodified."""
        rows = [(FakeEntry(id=1, intent="short intent"), 0)]
        service_with_mock._repo.list_entries_with_counts.return_value = rows

        page = service_with_mock.list_history(limit=10)
        assert page.entries[0].intent_preview == "short intent"

    def test_next_token_emitted_when_more_pages(self, service_with_mock):
        """Repo returns ``limit+1`` → page trims to ``limit`` and emits ``next_token``."""
        rows = [(FakeEntry(id=10 - i), 0) for i in range(6)]
        service_with_mock._repo.list_entries_with_counts.return_value = rows

        page = service_with_mock.list_history(limit=5)
        assert len(page.entries) == 5
        assert page.next_token == "6"

    def test_no_next_token_on_last_page(self, service_with_mock):
        """Repo returns ``<= limit`` → ``next_token`` is ``None``."""
        rows = [(FakeEntry(id=10 - i), 0) for i in range(3)]
        service_with_mock._repo.list_entries_with_counts.return_value = rows

        page = service_with_mock.list_history(limit=5)
        assert len(page.entries) == 3
        assert page.next_token is None


# --- get_entry (real-repo round-trip) ---


class TestGetEntry:
    def test_returns_none_for_missing(self, service_with_real):
        assert service_with_real.get_entry(9999) is None

    def test_round_trips_examples_base64(self, service_with_real):
        """Insert + ``get_entry`` round-trips ``image_data_url`` byte-for-byte."""
        raw_png = b"\x89PNG\r\n\x1a\n" + b"\x01\x02\x03"
        examples = [
            ExamplePair(subject="a", caption="b", image_data_url=f"data:image/png;base64,{_b64(raw_png)}"),
            ExamplePair(subject="c", caption="d", image_data_url=f"data:image/jpeg;base64,{_b64(b'\\xff\\xd8\\xff')}".replace(" ", "")),
        ]
        service_with_real.save_entry(
            PromptHistorySaveRequest(
                mode="refine",
                intent="refine this",
                focus="system",
                examples=examples,
                template_content="{% set system_prompt %}\nfoo\n{% endset %}",
            )
        )

        # Find the saved entry.
        page = service_with_real.list_history(limit=1)
        entry_id = page.entries[0].id

        entry = service_with_real.get_entry(entry_id)
        assert entry is not None
        assert entry["mode"] == "refine"
        assert entry["intent"] == "refine this"
        assert entry["focus"] == "system"
        assert entry["template_content"] == "{% set system_prompt %}\nfoo\n{% endset %}"
        assert entry["examples"] == [
            {"subject": "a", "caption": "b", "image_data_url": examples[0].image_data_url},
            {"subject": "c", "caption": "d", "image_data_url": examples[1].image_data_url},
        ]

    def test_round_trips_empty_examples(self, service_with_real):
        """An entry with no examples round-trips with ``examples == []``."""
        service_with_real.save_entry(PromptHistorySaveRequest(mode="generate", intent="x", focus="both"))
        page = service_with_real.list_history(limit=1)
        entry = service_with_real.get_entry(page.entries[0].id)

        assert entry is not None
        assert entry["examples"] == []

    def test_round_trips_null_template_content(self, service_with_real):
        """Generate-mode entries come back with ``template_content=None``."""
        service_with_real.save_entry(PromptHistorySaveRequest(mode="generate", intent="x", focus="both"))
        page = service_with_real.list_history(limit=1)
        entry = service_with_real.get_entry(page.entries[0].id)

        assert entry is not None
        assert entry["template_content"] is None


# --- delete_entry ---


class TestDeleteEntry:
    def test_delegates_to_repo(self, service_with_mock):
        service_with_mock._repo.delete_entry.return_value = True
        assert service_with_mock.delete_entry(42) is True
        service_with_mock._repo.delete_entry.assert_called_once_with(42)


# --- helpers ---


def _b64(data: bytes) -> str:
    """Base64-encode bytes (helper to keep test data readable)."""
    import base64 as _b64

    return _b64.b64encode(data).decode("ascii")
