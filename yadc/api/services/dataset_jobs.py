"""Dataset job coordinator — the single source of truth for "who holds a dataset".

At most one sidecar-writing job may run per dataset at a time. Captioning
and tagging both mutate image sidecars, so letting them overlap on the
same dataset would race on writes (a caption run could clobber a tag
extras-merge, or vice versa). This service is the cross-cutting claim
registry both consult before starting; the individual services keep
their own job dicts (status snapshots, cleanup) but the *right to start*
flows through here.

Adding a new sidecar-writing job type later means claiming through this
service — no need for the existing services to know about each other.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from logging import Logger
from typing import Literal

from yadc.api.modules import LoggingFactory, Service

type JobKind = Literal["captioning", "tagging"]


@dataclass(frozen=True)
class JobClaim:
    """Opaque handle returned by :meth:`DatasetJobService.try_acquire`.

    Treated as a bearer token: pass it back to :meth:`release`. Identity
    is compared by the coordinator so a stale release (e.g. from a job
    whose cleanup ran late) can't evict a newer owner.
    """

    dataset_name: str
    kind: JobKind
    job_id: str


class DatasetBusyError(ValueError):
    """Raised by :meth:`DatasetJobService.try_acquire` when the dataset is held.

    Subclasses :class:`ValueError` so existing controllers that map
    ``ValueError → 409`` keep working; catch this subclass directly when
    you want to surface *which* kind is blocking.
    """

    def __init__(self, dataset_name: str, held_by: JobClaim) -> None:
        self.dataset_name: str = dataset_name
        self.held_by: JobClaim = held_by
        super().__init__(f"A {held_by.kind} job is already running for dataset '{dataset_name}'")


class DatasetJobService(Service):
    """Cross-service registry of per-dataset job ownership.

    All mutations go through one :class:`asyncio.Lock`, so ``try_acquire``
    is the decisive atomic point across every job kind — there is no
    window in which two concurrent cross-type starts both succeed.
    """

    def __init__(self, logging: LoggingFactory):
        self._logger: Logger = logging.get_logger(__name__)
        self._lock: asyncio.Lock = asyncio.Lock()
        self._owners: dict[str, JobClaim] = {}

    async def try_acquire(self, dataset_name: str, kind: JobKind, job_id: str) -> JobClaim:
        """Atomically claim *dataset_name* for *kind*/*job_id*.

        Raises:
            DatasetBusyError: if any job already holds this dataset.
        """
        async with self._lock:
            existing = self._owners.get(dataset_name)
            if existing is not None:
                raise DatasetBusyError(dataset_name, existing)
            claim = JobClaim(dataset_name=dataset_name, kind=kind, job_id=job_id)
            self._owners[dataset_name] = claim
            return claim

    async def release(self, claim: JobClaim) -> None:
        """Release a claim. No-op if the dataset is now held by a different claim.

        The identity check protects against a late cleanup releasing a
        stale handle after a newer job has already taken ownership.
        """
        async with self._lock:
            if self._owners.get(claim.dataset_name) is claim:
                del self._owners[claim.dataset_name]

    def current(self, dataset_name: str) -> JobClaim | None:
        """Peek at the current owner without waiting on the lock.

        Safe for status reads from the event-loop thread; the result can
        be stale by the time it's acted on, which is fine for advisory
        display (the authoritative gate is always :meth:`try_acquire`).
        """
        return self._owners.get(dataset_name)

    def is_busy(self, dataset_name: str) -> bool:
        """Whether any job currently holds this dataset."""
        return dataset_name in self._owners
