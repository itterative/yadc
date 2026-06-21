"""SQL access for the ``prompt_history`` table + ``prompt_example_images`` child table.

The repository owns:

- The data models: :class:`PromptHistoryEntry` (entry-level row) and
  :class:`PromptExample` (per-image row, child of an entry) — both
  shaped by the SQL.
- All SQL: query text, parameter binding, row-to-dataclass mapping.

Per-image data lives in the ``prompt_example_images`` BLOB child
table (one row per image, ordered by ``ordinal``), not inline in
``prompt_history``. Image bytes are stored as raw ``BLOB`` (no
base64 inflation); the service layer encodes / decodes at the wire
boundary so the on-the-wire format stays ``data:<mime>;base64,...``.
Image rows are tied to their parent entry via FK with ``ON DELETE
CASCADE`` — deleting an entry cleans up its images automatically.

Connection management is internal to each public method — callers do
not pass a connection. Every method uses
:meth:`DBConnectionFactory.connection`, which auto-enrolls in any
active ``with db.transaction():`` block, so multi-statement
orchestration (insert entry + insert examples + prune) can be done
at the service level by wrapping several repo calls in a single
transaction.
"""

from __future__ import annotations

from dataclasses import dataclass
from logging import Logger
from typing import Any

from ..modules.db_connection_factory import DBConnectionFactory
from ..modules.logging_factory import LoggingFactory
from ..modules.service import Service


@dataclass
class PromptHistoryEntry:
    """A single prompt-history row (entry-level metadata only).

    The example list lives in the sibling ``prompt_example_images``
    table — see :class:`PromptExample`. Use
    :meth:`PromptHistoryRepository.get_entry_with_examples` or
    :meth:`PromptHistoryRepository.list_entries_with_counts` to
    fetch the examples alongside the entry.
    """

    id: int
    mode: str  # 'generate' | 'refine'
    intent: str
    focus: str  # 'system' | 'user' | 'both'
    template_content: str | None
    created_t: float


@dataclass
class PromptExample:
    """A single few-shot example row (one image + its metadata).

    ``data`` is the raw image bytes (no base64). The service layer
    encodes / decodes at the boundary. ``mime`` lets the service
    rebuild a ``data:<mime>;base64,...`` URL on read.
    """

    entry_id: int
    ordinal: int
    subject: str
    caption: str
    mime: str
    data: bytes


class PromptHistoryRepository(Service):
    """Read/write operations on the SQLite ``prompt_history`` table and its ``prompt_example_images`` child table."""

    def __init__(self, db: DBConnectionFactory, logging: LoggingFactory) -> None:
        self._db: DBConnectionFactory = db
        self._logger: Logger = logging.get_logger(__name__)

    # --- Reads ---

    def list_entries_with_counts(self, *, limit: int, before_id: int | None = None) -> list[tuple[PromptHistoryEntry, int]]:
        """List history entries (newest first) with their example counts.

        Returns a list of ``(entry, example_count)`` tuples. Uses a
        single ``LEFT JOIN ... GROUP BY`` so the count comes back
        attached to each entry — one round-trip, no per-entry image
        lookup.
        """
        with self._db.connection() as conn:
            if before_id is not None:
                rows: list[Any] = conn.execute(
                    """
                    SELECT h.id, h.mode, h.intent, h.focus, h.template_content, h.created_t,
                           COUNT(i.entry_id) AS example_count
                    FROM prompt_history h
                    LEFT JOIN prompt_example_images i ON i.entry_id = h.id
                    WHERE h.id < ?
                    GROUP BY h.id
                    ORDER BY h.id DESC
                    LIMIT ?
                    """,
                    (before_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT h.id, h.mode, h.intent, h.focus, h.template_content, h.created_t,
                           COUNT(i.entry_id) AS example_count
                    FROM prompt_history h
                    LEFT JOIN prompt_example_images i ON i.entry_id = h.id
                    GROUP BY h.id
                    ORDER BY h.id DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()

        return [
            (
                PromptHistoryEntry(
                    id=row[0],
                    mode=row[1],
                    intent=row[2],
                    focus=row[3],
                    template_content=row[4],
                    created_t=row[5],
                ),
                int(row[6]),
            )
            for row in rows
        ]

    def get_entry_with_examples(self, entry_id: int) -> tuple[PromptHistoryEntry, list[PromptExample]] | None:
        """Fetch a single entry + its example rows (in ordinal order).

        Two queries inside the same connection: one for the entry,
        one for the examples. They share the same connection so
        they observe the same transactional snapshot — safe to call
        from inside a ``with db.transaction():`` block.
        """
        with self._db.connection() as conn:
            row = conn.execute(
                "SELECT id, mode, intent, focus, template_content, created_t FROM prompt_history WHERE id = ?",
                (entry_id,),
            ).fetchone()
            if row is None:
                return None
            entry = PromptHistoryEntry(
                id=row[0],
                mode=row[1],
                intent=row[2],
                focus=row[3],
                template_content=row[4],
                created_t=row[5],
            )

            img_rows = conn.execute(
                """
                SELECT entry_id, ordinal, subject, caption, mime, data
                FROM prompt_example_images
                WHERE entry_id = ?
                ORDER BY ordinal
                """,
                (entry_id,),
            ).fetchall()
            examples = [
                PromptExample(
                    entry_id=r[0],
                    ordinal=r[1],
                    subject=r[2],
                    caption=r[3],
                    mime=r[4],
                    data=bytes(r[5]),  # sqlite3 returns memoryview — wrap to bytes
                )
                for r in img_rows
            ]

        return entry, examples

    def count_entries(self) -> int:
        """Return the total number of history entries."""
        with self._db.connection() as conn:
            row = conn.execute("SELECT COUNT(*) FROM prompt_history").fetchone()
        return row[0]

    # --- Writes ---

    def insert_entry(
        self,
        mode: str,
        intent: str,
        focus: str,
        template_content: str | None,
        created_t: float,
    ) -> int:
        """Insert an entry-level row and return its ID. Examples are inserted separately via :meth:`insert_examples`."""
        with self._db.connection() as conn:
            cursor = conn.execute(
                "INSERT INTO prompt_history (mode, intent, focus, template_content, created_t) VALUES (?, ?, ?, ?, ?)",
                (mode, intent, focus, template_content, created_t),
            )
            row_id = cursor.lastrowid
            assert row_id is not None
            return row_id

    def insert_examples(self, examples: list[PromptExample]) -> None:
        """Insert multiple example rows in one ``executemany``.

        Each :class:`PromptExample` carries its own ``entry_id`` (the
        service sets it before calling, from the freshly-inserted
        ``prompt_history`` row). ``examples`` must already be sorted
        by ``ordinal`` (the caller preserves the order it received
        from the wire). Empty lists are a no-op.
        """
        if not examples:
            return
        params = [(ex.entry_id, ex.ordinal, ex.subject, ex.caption, ex.mime, ex.data) for ex in examples]
        with self._db.connection() as conn:
            conn.executemany(
                "INSERT INTO prompt_example_images (entry_id, ordinal, subject, caption, mime, data) VALUES (?, ?, ?, ?, ?, ?)",
                params,
            )

    def delete_entry(self, entry_id: int) -> bool:
        """Delete a single history entry. Returns True if found and deleted.

        Image rows are removed automatically by the ``ON DELETE
        CASCADE`` FK constraint — no service-level cleanup needed.
        """
        with self._db.connection() as conn:
            cursor = conn.execute("DELETE FROM prompt_history WHERE id = ?", (entry_id,))
            return cursor.rowcount > 0

    def prune_old(self, keep_count: int) -> int:
        """Delete entries beyond the ``keep_count`` newest. Returns the number deleted.

        Single statement using a subquery: ``DELETE WHERE id NOT IN
        (SELECT id ORDER BY id DESC LIMIT keep_count)`` keeps the
        ``keep_count`` newest rows (id is monotonic — it's the
        primary key autoincrement) and deletes everything else.
        Image rows go with the cascade.
        """
        with self._db.connection() as conn:
            cursor = conn.execute(
                "DELETE FROM prompt_history WHERE id NOT IN (SELECT id FROM prompt_history ORDER BY id DESC LIMIT ?)",
                (keep_count,),
            )
            deleted = cursor.rowcount
            if deleted > 0:
                self._logger.debug("Pruned %d old prompt-history entries (keeping %d).", deleted, keep_count)
            return deleted
