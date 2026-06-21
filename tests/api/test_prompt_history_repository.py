"""Unit tests for :class:`PromptHistoryRepository`.

Uses the shared :class:`DBConnectionFactory` fixture (migrations
applied to a temp-file DB) so the schema is created by the existing
migrations — including the ``0009_prompt_history_images`` BLOB
table. Builds state via the repo's public methods and asserts on
the repo's public methods.
"""

from __future__ import annotations

import time

import pytest

from yadc.api.modules.db_connection_factory import DBConnectionFactory
from yadc.api.modules.logging_factory import LoggingFactory
from yadc.api.services.prompt_history_repository import PromptExample, PromptHistoryEntry, PromptHistoryRepository


@pytest.fixture
def repo(
    db_connection_factory: DBConnectionFactory,
    logging_factory: LoggingFactory,
) -> PromptHistoryRepository:
    return PromptHistoryRepository(db=db_connection_factory, logging=logging_factory)


def _make_entry(
    repo: PromptHistoryRepository,
    *,
    mode: str = "generate",
    intent: str = "Caption each image as a single sentence.",
    focus: str = "both",
    template_content: str | None = None,
    created_t: float | None = None,
) -> int:
    return repo.insert_entry(
        mode=mode,
        intent=intent,
        focus=focus,
        template_content=template_content,
        created_t=created_t if created_t is not None else time.time(),
    )


def _make_example(
    entry_id: int,
    ordinal: int,
    *,
    subject: str = "a cat",
    caption: str = "a tabby cat",
    mime: str = "image/png",
    data: bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8,
) -> PromptExample:
    return PromptExample(
        entry_id=entry_id,
        ordinal=ordinal,
        subject=subject,
        caption=caption,
        mime=mime,
        data=data,
    )


class TestInsertAndGet:
    def test_round_trip(self, repo):
        """Insert + get returns the same fields (no example rows)."""
        _make_entry(
            repo,
            mode="refine",
            intent="Make the system prompt more concise.",
            focus="system",
            template_content="{% set system_prompt %}\nfoo\n{% endset %}",
        )
        rows = repo.list_entries_with_counts(limit=1)
        assert len(rows) == 1
        entry, count = rows[0]
        assert isinstance(entry, PromptHistoryEntry)
        assert entry.mode == "refine"
        assert entry.intent == "Make the system prompt more concise."
        assert entry.focus == "system"
        assert entry.template_content == "{% set system_prompt %}\nfoo\n{% endset %}"
        assert entry.created_t > 0
        assert count == 0  # no examples attached

    def test_template_content_null_for_generate(self, repo):
        """Generate-mode entries have ``template_content = None``."""
        _make_entry(repo, mode="generate", template_content=None)
        rows = repo.list_entries_with_counts(limit=1)
        assert rows[0][0].template_content is None


class TestExamples:
    def test_insert_and_get_examples(self, repo):
        """Insert example rows + fetch them back in ordinal order."""
        entry_id = _make_entry(repo)
        repo.insert_examples(
            [
                _make_example(entry_id, ordinal=0, subject="first", caption="one"),
                _make_example(entry_id, ordinal=1, subject="second", caption="two"),
            ],
        )

        result = repo.get_entry_with_examples(entry_id)
        assert result is not None
        entry, examples = result
        assert entry.id == entry_id
        assert len(examples) == 2
        assert examples[0].subject == "first"
        assert examples[1].subject == "second"
        assert examples[0].caption == "one"
        assert examples[1].caption == "two"
        assert examples[0].mime == "image/png"
        assert examples[0].data == b"\x89PNG\r\n\x1a\n" + b"\x00" * 8

    def test_insert_examples_empty_list_is_noop(self, repo):
        """An empty ``examples`` list is a no-op (no rows inserted)."""
        entry_id = _make_entry(repo)
        repo.insert_examples([])  # no error

        result = repo.get_entry_with_examples(entry_id)
        assert result is not None
        _, examples = result
        assert examples == []

    def test_insert_examples_preserves_ordinals(self, repo):
        """Ordinals are stored as-given — the caller (service) is responsible for ordering."""
        entry_id = _make_entry(repo)
        repo.insert_examples(
            [_make_example(entry_id, ordinal=2), _make_example(entry_id, ordinal=0), _make_example(entry_id, ordinal=1)],
        )

        _, examples = repo.get_entry_with_examples(entry_id)
        # The repo stores what was given; ordering on read happens via ORDER BY.
        ordinals = [ex.ordinal for ex in examples]
        assert ordinals == [0, 1, 2]  # sorted on read

    def test_get_entry_returns_none_for_missing(self, repo):
        """``get_entry_with_examples`` returns ``None`` if the entry doesn't exist."""
        assert repo.get_entry_with_examples(9999) is None


class TestList:
    def test_orders_newest_first(self, repo):
        """Inserts in order 1, 2, 3 → list returns 3, 2, 1 (DESC by id)."""
        ids = [_make_entry(repo, intent=f"intent-{i}") for i in range(1, 4)]

        rows = repo.list_entries_with_counts(limit=10)
        assert [r[0].id for r in rows] == [ids[2], ids[1], ids[0]]

    def test_respects_limit(self, repo):
        """``limit`` caps the number of rows returned."""
        for _ in range(5):
            _make_entry(repo)
        rows = repo.list_entries_with_counts(limit=3)
        assert len(rows) == 3

    def test_before_id_filters(self, repo):
        """``before_id`` returns only rows with id < the cursor (DESC order)."""
        ids = [_make_entry(repo, intent=f"intent-{i}") for i in range(5)]
        # Page 1: top 2 (ids 5, 4)
        page1 = repo.list_entries_with_counts(limit=2)
        assert [r[0].id for r in page1] == [ids[4], ids[3]]
        # Page 2: next 2 (ids 3, 2) — before_id = page1 last id
        page2 = repo.list_entries_with_counts(limit=2, before_id=page1[-1][0].id)
        assert [r[0].id for r in page2] == [ids[2], ids[1]]

    def test_example_count_aggregated_in_sql(self, repo):
        """``example_count`` is the SQL aggregate, not a per-row parse."""
        e1 = _make_entry(repo, intent="a")
        e2 = _make_entry(repo, intent="b")
        e3 = _make_entry(repo, intent="c")
        repo.insert_examples([_make_example(e1, ordinal=0), _make_example(e1, ordinal=1)])
        repo.insert_examples([_make_example(e2, ordinal=0)])
        # e3 has zero examples.

        rows = repo.list_entries_with_counts(limit=10)
        counts = {r[0].id: r[1] for r in rows}
        assert counts == {e1: 2, e2: 1, e3: 0}

    def test_empty_table(self, repo):
        """An empty table returns an empty list (not None / not an error)."""
        assert repo.list_entries_with_counts(limit=10) == []


class TestDelete:
    def test_returns_true_for_existing(self, repo):
        row_id = _make_entry(repo)
        assert repo.delete_entry(row_id) is True

    def test_returns_false_for_missing(self, repo):
        assert repo.delete_entry(9999) is False

    def test_actually_deletes(self, repo):
        row_id = _make_entry(repo)
        repo.delete_entry(row_id)
        assert repo.get_entry_with_examples(row_id) is None
        assert repo.list_entries_with_counts(limit=10) == []

    def test_cascade_deletes_example_rows(self, repo):
        """Deleting an entry cascades to its example rows (``ON DELETE CASCADE``)."""
        entry_id = _make_entry(repo)
        repo.insert_examples([_make_example(entry_id, ordinal=0), _make_example(entry_id, ordinal=1)])

        repo.delete_entry(entry_id)

        # The example rows must be gone (verified via a direct DB query
        # since ``get_entry_with_examples`` returns ``None`` for a
        # deleted entry and won't surface orphan rows).
        with repo._db.connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM prompt_example_images WHERE entry_id = ?", (entry_id,)).fetchone()[0]
        assert count == 0


class TestCount:
    def test_empty(self, repo):
        assert repo.count_entries() == 0

    def test_after_inserts(self, repo):
        for _ in range(3):
            _make_entry(repo)
        assert repo.count_entries() == 3


class TestPruneOld:
    def test_keeps_newest_n(self, repo):
        """Insert 5 with keep_count=3 → 3 newest survive (5 - 2 = 3)."""
        ids = [_make_entry(repo, intent=f"intent-{i}") for i in range(1, 6)]
        deleted = repo.prune_old(keep_count=3)
        assert deleted == 2

        survivors = {r[0].id for r in repo.list_entries_with_counts(limit=10)}
        # Newest 3 (by id) survive
        assert survivors == {ids[4], ids[3], ids[2]}

    def test_no_op_under_cap(self, repo):
        """Insert 3 with keep_count=5 → 0 deletes."""
        for _ in range(3):
            _make_entry(repo)
        assert repo.prune_old(keep_count=5) == 0
        assert repo.count_entries() == 3

    def test_at_cap(self, repo):
        """Insert exactly keep_count → 0 deletes (boundary)."""
        for _ in range(5):
            _make_entry(repo)
        assert repo.prune_old(keep_count=5) == 0

    def test_keep_count_zero_deletes_all(self, repo):
        """``keep_count=0`` deletes every row (no rows are ``id IN ()``)."""
        for _ in range(3):
            _make_entry(repo)
        deleted = repo.prune_old(keep_count=0)
        assert deleted == 3
        assert repo.count_entries() == 0
