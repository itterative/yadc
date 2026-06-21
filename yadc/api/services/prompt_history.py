"""Prompt-history service — save/load the prompt-generator artifact.

The repo (:class:`PromptHistoryRepository`) owns the SQL. This service
owns:

- Public Python API: Pydantic models for the save args + the list /
  full response shapes (the wire-format model lives in the controller;
  the service-layer model is the same shape, used internally and
  for testing).
- Base64 ↔ BLOB translation at the boundary (the wire format uses
  ``data:<mime>;base64,...`` URLs; the repo stores raw image bytes
  in a sibling ``prompt_example_images`` table — no base64 inflation
  in the DB).
- Server-derived list-view fields (``intent_preview``,
  ``example_count``, ``had_template``) so the frontend doesn't have to
  compute them.
- The prune-to-cap policy on save.
- Transaction boundary for the multi-statement save (insert entry +
  insert examples + prune).
"""

from __future__ import annotations

import base64
import re
import time
from dataclasses import dataclass
from logging import Logger
from typing import ClassVar, Literal

import pydantic

from ..modules.db_connection_factory import DBConnectionFactory
from ..modules.logging_factory import LoggingFactory
from ..modules.service import Service
from .prompt_generation import ExamplePair, PromptGenerationFocus
from .prompt_history_repository import PromptExample, PromptHistoryEntry, PromptHistoryRepository

__all__ = [
    "PROMPT_HISTORY_MAX_ENTRIES",
    "PromptHistoryListItem",
    "PromptHistoryPage",
    "PromptHistorySaveRequest",
    "PromptHistoryService",
]

# Hard cap on the number of history entries. On save, the service
# prunes the oldest entries beyond this count. Mirrors the
# config_history pattern (``DEFAULT_MAX_ENTRIES = 50`` there — picked
# 20 here because each entry can hold raw image blobs and we don't
# need a long tail).
PROMPT_HISTORY_MAX_ENTRIES = 20

# ``data:<mime>;base64,<...>`` — used to parse the wire-format
# ``image_data_url`` back into ``(mime, raw_bytes)`` for storage. The
# regex is intentionally tolerant: it captures the mime as ``[^;]+``
# (anything up to the first ``;``) and the payload as everything
# after ``;base64,``. Trailing whitespace in the payload is not
# allowed by ``base64.b64decode`` but doesn't arise in practice —
# browser ``FileReader.readAsDataURL`` produces clean base64.
_DATA_URL_RE = re.compile(r"^data:([^;]+);base64,(.*)$", re.DOTALL)


def _decode_data_url(url: str) -> tuple[str, bytes] | None:
    """Parse a ``data:<mime>;base64,...`` URL into ``(mime, bytes)``.

    Returns ``None`` if the URL doesn't match the expected shape
    or the base64 decode fails. Callers should skip the example
    silently in that case — a malformed example shouldn't block a
    save.
    """
    match = _DATA_URL_RE.match(url)
    if match is None:
        return None
    mime, payload = match.group(1), match.group(2)
    try:
        raw = base64.b64decode(payload, validate=True)
    except ValueError:
        # ``base64.b64decode`` raises ``binascii.Error`` (a ``ValueError``
        # subclass) for invalid base64. Returning ``None`` lets the
        # caller skip the example silently.
        return None
    return mime, raw


def _encode_data_url(mime: str, data: bytes) -> str:
    """Build a ``data:<mime>;base64,...`` URL from raw bytes."""
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


class PromptHistorySaveRequest(pydantic.BaseModel):
    """Service-layer Pydantic model for ``save_entry`` inputs.

    Matches the wire-format save body in the controller (with
    ``examples`` as Pydantic models instead of raw dicts, so the
    service can validate the shape before encoding).
    """

    mode: Literal["generate", "refine"]
    intent: str
    focus: PromptGenerationFocus
    examples: list[ExamplePair] = pydantic.Field(default_factory=list)
    template_content: str | None = None

    model_config: ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(extra="forbid")


@dataclass
class PromptHistoryListItem:
    """A summary row for the list view — no examples payload, no full intent.

    ``intent_preview`` is the first ~2 lines of the intent (truncated
    to ``INTENT_PREVIEW_MAX_CHARS`` so a long intent doesn't blow up
    the list UI). Server-computed so the frontend doesn't have to
    know how to slice a multi-line string.

    ``had_template`` is derived from ``template_content is not None``
    — included separately so the frontend doesn't have to carry
    the nullable field around in the list shape.

    ``example_count`` is derived server-side from the JOIN/GROUP BY
    in the list query, so the frontend doesn't have to know how the
    examples are stored.
    """

    id: int
    mode: str
    had_template: bool
    focus: str
    intent_preview: str
    example_count: int
    created_t: float


@dataclass
class PromptHistoryPage:
    """A paginated page of history entries (newest first).

    ``next_token`` is the opaque cursor to pass back for the next
    (older) page. ``None`` means this is the last page. The
    encoding is an implementation detail — the client treats it as
    a string token.
    """

    entries: list[PromptHistoryListItem]
    next_token: str | None = None


# How many leading characters of the intent to expose in the list
# view. Long intents get truncated with an ellipsis. 200 chars ≈
# 2-3 lines at typical textarea widths.
INTENT_PREVIEW_MAX_CHARS = 200


def _make_intent_preview(intent: str) -> str:
    """First ``INTENT_PREVIEW_MAX_CHARS`` characters of ``intent``, with an ellipsis on truncation."""
    if len(intent) <= INTENT_PREVIEW_MAX_CHARS:
        return intent
    return intent[:INTENT_PREVIEW_MAX_CHARS].rstrip() + "…"


def _list_item_from_entry(entry: PromptHistoryEntry, example_count: int) -> PromptHistoryListItem:
    """Build a :class:`PromptHistoryListItem` from a :class:`PromptHistoryEntry`."""
    return PromptHistoryListItem(
        id=entry.id,
        mode=entry.mode,
        had_template=entry.template_content is not None,
        focus=entry.focus,
        intent_preview=_make_intent_preview(entry.intent),
        example_count=example_count,
        created_t=entry.created_t,
    )


class PromptHistoryService(Service):
    """Saves and retrieves prompt-history entries.

    The prune policy (``PROMPT_HISTORY_MAX_ENTRIES``) runs inside the
    same transaction as the insert, so a partially-applied save (e.g.
    crash mid-write) can never leave the table over the cap. Mirrors
    :meth:`ConfigHistoryService.save_snapshot`'s pattern.

    Base64 ↔ BLOB translation happens here at the service boundary
    so the wire format (``ExamplePair.image_data_url``) stays the
    same while the DB stores raw bytes in the
    ``prompt_example_images`` BLOB table.
    """

    def __init__(
        self,
        db: DBConnectionFactory,
        logging: LoggingFactory,
        repo: PromptHistoryRepository,
    ) -> None:
        self._db: DBConnectionFactory = db
        self._logger: Logger = logging.get_logger(__name__)
        self._repo: PromptHistoryRepository = repo

    def save_entry(self, request: PromptHistorySaveRequest) -> int:
        """Save a history entry. Returns the new row ID.

        Decodes each ``ExamplePair.image_data_url`` into raw bytes,
        inserts the entry row + the example rows + the prune in one
        transaction. Malformed data URLs (regex miss / bad base64)
        are silently skipped — a single corrupt example shouldn't
        block a save.
        """
        examples: list[PromptExample] = []
        for ordinal, ex in enumerate(request.examples):
            decoded = _decode_data_url(ex.image_data_url)
            if decoded is None:
                # Defensive: in practice the frontend always produces
                # clean data URLs, so this branch is rarely hit.
                continue
            mime, data = decoded
            examples.append(
                PromptExample(
                    entry_id=0,  # overwritten below once we know the row id
                    ordinal=ordinal,
                    subject=ex.subject,
                    caption=ex.caption,
                    mime=mime,
                    data=data,
                )
            )

        with self._db.transaction():
            row_id = self._repo.insert_entry(
                mode=request.mode,
                intent=request.intent,
                focus=request.focus,
                template_content=request.template_content,
                created_t=time.time(),
            )
            # Bind the freshly-assigned entry_id to every example
            # before inserting them.
            for ex in examples:
                ex.entry_id = row_id
            self._repo.insert_examples(examples)
            self._repo.prune_old(PROMPT_HISTORY_MAX_ENTRIES)

        self._logger.debug(
            "Saved prompt-history entry (id=%d, mode=%s, examples=%d).",
            row_id,
            request.mode,
            len(examples),
        )
        return row_id

    def list_history(self, *, limit: int = 50, next_token: str | None = None) -> PromptHistoryPage:
        """List history entries (newest first) as :class:`PromptHistoryListItem`\\s.

        Fetches ``limit + 1`` rows to detect "is there a next page"
        without an extra count query. The example count comes back
        attached (single SQL with ``LEFT JOIN ... GROUP BY``) — no
        per-row JSON parse like the pre-BLOB implementation.
        """
        before_id = self._decode_next_token(next_token)
        rows = self._repo.list_entries_with_counts(limit=limit + 1, before_id=before_id)
        has_more = len(rows) > limit
        page_rows = rows[:limit]

        items = [_list_item_from_entry(entry, count) for entry, count in page_rows]
        next_token = str(page_rows[-1][0].id) if has_more and page_rows else None
        return PromptHistoryPage(entries=items, next_token=next_token)

    def get_entry(self, entry_id: int) -> dict[str, object] | None:
        """Return the full entry as a JSON-friendly dict (with ``examples`` decoded).

        ``None`` if the entry doesn't exist. The shape mirrors the
        ``PromptHistorySaveRequest`` (plus ``id`` + ``created_t``)
        so the frontend can use it as a drop-in for the restore
        action. ``template_content`` is preserved (may be ``None``).

        Examples are re-encoded from raw BLOB bytes back to
        ``data:<mime>;base64,...`` URLs so the wire shape matches
        the save body.
        """
        result = self._repo.get_entry_with_examples(entry_id)
        if result is None:
            return None
        entry, examples = result
        return {
            "id": entry.id,
            "mode": entry.mode,
            "intent": entry.intent,
            "focus": entry.focus,
            "examples": [
                {
                    "subject": ex.subject,
                    "caption": ex.caption,
                    "image_data_url": _encode_data_url(ex.mime, ex.data),
                }
                for ex in examples
            ],
            "template_content": entry.template_content,
            "created_t": entry.created_t,
        }

    def delete_entry(self, entry_id: int) -> bool:
        """Delete a single history entry. Returns True if found and deleted.

        Image rows are removed automatically by the FK ``ON DELETE
        CASCADE`` — no service-level cleanup.
        """
        return self._repo.delete_entry(entry_id)

    @staticmethod
    def _decode_next_token(token: str | None) -> int:
        """Decode an opaque ``next`` token into the repo's ``before_id`` cursor.

        ``None`` or an empty token means "no upper bound" — use a
        value larger than any real id so all rows pass the
        ``id < ?`` filter. SQLite's INTEGER is 64-bit; ``2**63 - 1``
        is the max signed value and won't be reached by
        AUTOINCREMENT for billions of years. Mirrors
        :meth:`ConfigHistoryService._decode_next_token`.
        """
        if not token:
            return 2**63 - 1
        return int(token)
