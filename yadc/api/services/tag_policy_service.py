"""Per-dataset tag policy — always-add / banned lists used during tag result filtering.

A simple dataclass returned by the service, plus a thin wrapper around
:class:`DatasetSettingsRepository` that decodes the two stored rows
(``policy_always_add`` / ``policy_banned`` JSON arrays) into a
:class:`TagPolicy`. The dataset name → ``dataset_id`` resolution goes
through :class:`DatasetRepository` (one small SQL read) so the public
API stays dataset-name-keyed — same as the rest of the API surface.

Behaviour:

- A never-configured dataset (no rows) decodes to an **empty**
  :class:`TagPolicy` (``always_add=[]``, ``banned=[]``). This matches
  the original "frontend sends a default ``TagPolicy()`` if unset"
  semantic — a stored-but-empty policy is observationally identical
  to no policy.
- A missing row decodes to ``[]``; a row whose JSON is unparseable
  logs and falls back to ``[]`` rather than raising, mirroring
  :meth:`SettingsService.get`.
- Both keys are independently upsertable. Toggling one list in the
  UI doesn't overwrite the other.

The wire shape is mirrored by a Pydantic model so the JSON-decoded
``Any`` parses into a typed structure (and basedpyright stays clean
on the per-row accessors).
"""

from __future__ import annotations

import json
from logging import Logger

import pydantic

from yadc.taggers.postprocessing import TagPolicy

from ..modules.db_connection_factory import DBConnectionFactory
from ..modules.logging_factory import LoggingFactory
from ..modules.service import Service
from .dataset_repository import DatasetRepository
from .dataset_settings_repository import DatasetSettingsRepository


class TagPolicyPayload(pydantic.BaseModel):
    """Wire shape for one row of the per-dataset tag policy.

    Mirrors what we accept on the wire (PUT body) and what we emit
    (GET response). Each row in :class:`DatasetSettingsRepository`
    carries one of these (the two lists are stored separately so
    mutating one doesn't rewrite the other).
    """

    tags: list[str] = pydantic.Field(default_factory=list)


class TagPolicyService(Service):
    """Per-dataset always-add / banned policy accessor.

    Public API is keyed by ``dataset_name`` (string) so the caller —
    usually :class:`TaggingService` or a controller — doesn't have to
    thread the SQL ``dataset_id`` through. Each call resolves the id
    via :class:`DatasetRepository.get_dataset_row`; missing datasets
    return an empty policy rather than raising (consumers gate on
    image existence / dataset registration upstream).
    """

    #: SQLite row key for the always-add JSON array.
    _KEY_ALWAYS_ADD: str = "policy_always_add"
    #: SQLite row key for the banned JSON array.
    _KEY_BANNED: str = "policy_banned"

    def __init__(
        self,
        db: DBConnectionFactory,
        logging: LoggingFactory,
        repo: DatasetSettingsRepository,
        dataset_repo: DatasetRepository,
    ) -> None:
        self._db: DBConnectionFactory = db
        self._logger: Logger = logging.get_logger(__name__)
        self._repo: DatasetSettingsRepository = repo
        self._dataset_repo: DatasetRepository = dataset_repo

    def get(self, dataset_name: str) -> TagPolicy:
        """Return the stored policy for ``dataset_name``.

        Missing rows, missing dataset, or unparseable JSON all fall
        back to the default :class:`TagPolicy` (both lists empty).
        A missing dataset is not an error: callers that have already
        resolved the dataset upstream (image detail UI, batch job
        preflight) keep their existing 404 / 409 semantics, and
        callers that haven't resolved it yet get a no-op policy
        rather than a stack trace.
        """
        dataset_row = self._dataset_repo.get_dataset_row(dataset_name)
        if dataset_row is None:
            return TagPolicy()
        dataset_id = dataset_row[0]
        always_add = self._decode_list(dataset_id, self._KEY_ALWAYS_ADD)
        banned = self._decode_list(dataset_id, self._KEY_BANNED)
        return TagPolicy(always_add=always_add, banned=banned)

    def set(self, dataset_name: str, policy: TagPolicy) -> bool:
        """Persist ``policy`` for ``dataset_name``. Returns ``True`` on success.

        Each list is encoded independently — a missing/empty list is
        also persisted (as ``[]``) so a cleared policy is observable
        in the storage layer and survives a restart. Returns ``False``
        when the dataset is not registered (silently: callers don't
        need to special-case this since the controller maps to 404).
        """
        dataset_row = self._dataset_repo.get_dataset_row(dataset_name)
        if dataset_row is None:
            return False
        dataset_id = dataset_row[0]
        self._repo.upsert(dataset_id, self._KEY_ALWAYS_ADD, _dump_list(policy.always_add))
        self._repo.upsert(dataset_id, self._KEY_BANNED, _dump_list(policy.banned))
        return True

    def _decode_list(self, dataset_id: int, key: str) -> list[str]:
        """Decode a stored JSON list, falling back to ``[]`` on miss / parse failure."""
        raw = self._repo.get(dataset_id, key)
        if raw is None:
            return []
        try:
            payload = TagPolicyPayload.model_validate(json.loads(raw))
        except (json.JSONDecodeError, pydantic.ValidationError, ValueError):
            # ``ValueError`` covers ``model_validate``'s pre-2.x
            # fallback path; pydantic 2 raises ``ValidationError``,
            # but older yadc docs / tests reference ``ValueError``.
            self._logger.warning(
                "Failed to decode dataset_settings row, falling back to empty list. [dataset_id=%d, key=%s]",
                dataset_id,
                key,
            )
            return []
        return list(payload.tags)


def _dump_list(tags: list[str]) -> str:
    """JSON-serialize one policy list through the wire payload shape."""
    return TagPolicyPayload(tags=list(tags)).model_dump_json()
