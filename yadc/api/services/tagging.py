"""Tagging service — manages the tagger subprocess for the API backend.

The subprocess is spawned **lazily** on the first tagging request and
torn down after ``Configuration.tagger_idle_timeout_seconds`` of
inactivity (default 15 minutes). A periodic background job
(``IDLE_CHECK_INTERVAL_SECONDS``) checks the idle timer and stops the
subprocess when it expires. The next request after teardown respawns
it.

Lifecycle transitions (start / stop / idle-stop) are serialized
through ``self._lifecycle_lock`` (``threading.Lock``) so the sync
idle check, async request handlers, and the shutdown event handler
never race on the subprocess state.

The service dispatches ``TaggerStatusEvent`` (subprocess state
changes), ``ImageTaggedEvent`` (per-image success), and
``ImageTagErrorEvent`` (per-image failure) via the
:class:`EventDispatcher` so SSE clients see what happened.
"""

from __future__ import annotations

import asyncio
import math
import threading
import time
import uuid
from dataclasses import dataclass, field
from logging import Logger
from pathlib import Path
from typing import Any, Literal

import pydantic
import tomlkit

from yadc.api.configuration import Configuration
from yadc.api.events import (
    ImageTagErrorEvent,
    ImageTaggedEvent,
    ImageTagStartedEvent,
    ShutdownEvent,
    StartupEvent,
    TaggerStatusEvent,
    TagJobStatusEvent,
)
from yadc.api.modules import DatasetWatcherService, EventDispatcher, JobScheduler, LoggingFactory, Service
from yadc.api.modules.dataset_watcher import SELF_JOB_ID
from yadc.api.modules.event_dispatcher import event_handler
from yadc.api.modules.tagger_catalog import KNOWN_TAGGER_MODELS, ActiveTagger
from yadc.api.services.dataset_jobs import DatasetJobService, JobClaim
from yadc.api.services.dataset_repository import ImageInfo
from yadc.api.services.datasets import DatasetService
from yadc.api.services.settings import SettingsService
from yadc.api.services.tag_policy_service import TagPolicyService
from yadc.taggers import OnnxTagger, TaggerResult, apply_thresholds, extras_tags, format_draft
from yadc.taggers.base import TagCustomizations, tamer_result_size
from yadc.taggers.client import TaggerClient
from yadc.taggers.postprocessing import TagPolicy, apply_policy
from yadc.taggers.postprocessing import replace_underscores as replace_underscores_in
from yadc.utils import MemoryLRU, size_units

# How often the idle-check job wakes up. Worst-case shutdown latency
# after the timeout elapses is one interval (e.g. 30s for a 15min
# timeout). Small enough not to waste CPU; large enough to be
# negligible relative to the timeout.
IDLE_CHECK_INTERVAL_SECONDS: float = 30.0

# Step size used to floor tagger thresholds into cache buckets. The
# cache key stores the bucket floor (e.g. ``general=0.4``) and the
# cached value is post-thresholded at that floor, so requests with
# effective thresholds greater than or equal to the floor can be
# served from the slot after re-filtering. Bumping the step trades
# cache hit rate (larger steps → more requests per slot) for storage
# cost (smaller steps → fewer tags dropped at write time).
THRESHOLD_BUCKET_STEP: float = 0.2


def bucket_threshold(value: float, step: float = THRESHOLD_BUCKET_STEP) -> float:
    """Round a tagger threshold DOWN to the nearest ``step`` boundary.

    Used to coarsen ``TaggerResultKey`` so requests at nearby
    threshold values share a slot. The cached value is post-thresholded
    at the bucket floor (already a subset of any stricter request), so
    the request's effective filters are reapplied at retrieval.
    Loose requests (``effective_threshold < bucket_floor``) miss
    because the cached set has been filtered tighter.

    Examples (default step ``0.2``):

    - ``0.0``  → ``0.0``
    - ``0.05`` → ``0.0``
    - ``0.35`` → ``0.2``
    - ``0.5``  → ``0.4``
    - ``0.85`` → ``0.8``
    - ``1.0``  → ``1.0``
    """
    if value <= 0:
        return 0.0
    # ``+ 1e-9`` dodges float-precision ``0.6 / 0.2 = 2.999999...``
    # which would otherwise floor to ``1.2`` instead of ``0.6``.
    return math.floor(value / step + 1e-9) * step


TaggerState = Literal["starting", "ready", "stopping", "stopped", "failed"]


type TagJobStatus = Literal["idle", "running", "stopping", "error", "done", "cancelled"]


# Outcome of a cancel request (graceful-first, kill-as-fallback). "stopped"
# means a job's stop_event was set (or nothing was running) without needing to
# kill the subprocess; "killed" means the subprocess was force-terminated
# because an inference was still in flight after the grace window.
type CancelOutcome = Literal["stopped", "killed", "stale_job", "nothing_running"]


@dataclass
class TaggingThresholds:
    """Per-category score thresholds applied to a tagging result."""

    rating: float = 0.0
    general: float = 0.35
    character: float = 0.85


class TagSaveOptions(pydantic.BaseModel):
    """Where and how to persist a tag result for an image.

    - ``none`` — tag only (emit events, write nothing). Default.
    - ``draft`` — write formatted text to a named draft file
      (``<image>.<name>.draft~``). Reuses the caption draft sidecar.
    - ``extras`` — merge a ``[tags]`` sub-table into the image TOML
      (``general`` / ``character`` lists + a single ``rating`` string).

    ``overwrite`` governs the batch save path only: when ``False`` (the
    default), images that already carry the target artifact (a draft of
    the same name, or a ``[tags]`` sub-table) are skipped entirely — no
    tag, no save. The interactive save endpoint always writes.
    """

    mode: Literal["none", "draft", "extras"] = "none"
    draft_name: str = "tags"
    draft_format: str = "comma"
    overwrite: bool = False


class TagJobOptions(pydantic.BaseModel):
    """Body for ``POST /datasets/<name>/tag`` — start a batch tagging job.

    ``image_ids`` omitted/empty → tag the whole dataset; otherwise tag
    just the listed images. Thresholds fall back to the server config
    when unset. ``save`` controls per-image persistence. The
    always-add / banned policy is resolved server-side from
    :class:`TagPolicyService` against the dataset — request bodies
    do not carry an override.
    """

    image_ids: list[int] | None = None
    rating_threshold: float | None = None
    general_threshold: float | None = None
    character_threshold: float | None = None
    replace_underscores: bool | None = None
    save: TagSaveOptions = pydantic.Field(default_factory=TagSaveOptions)
    source: str | None = None


@dataclass(frozen=True)
class TaggerResultKey:
    """Cache key for a single tagged image result.

    Two keys compare equal iff they name the same image AND were
    produced against the same model AND fall under the same bucketed
    thresholds. When any of those change the new key hashes to a
    different slot.

    Threshold bucketing coarsens the key so requests at nearby
    threshold values share a slot:

    - ``rating_threshold`` is **always** ``0.0``. We never pre-filter
      rating tags at write time, so the cached set always carries all
      four rating categories. The request's effective rating is
      reapplied at retrieval.
    - ``general_threshold`` / ``character_threshold`` are the
      :func:`bucket_threshold` floors of the request's effective
      thresholds. The cached value is post-thresholded at those
      floors, so it's a subset of any stricter request and the
      effective thresholds can be reapplied at retrieval.

    ``replace_underscores`` and the always-add / banned policy are
    **not** in the key. Both are near-free transforms applied post-hoc
    at retrieval (see :meth:`TaggingService._refilter`), so a single
    cache slot serves every variant of either — toggling a policy tag
    shows up on the next read without re-tagging.

    Frozen so the dataclass-generated ``__hash__`` and ``__eq__`` are
    reliable: ``LRU`` uses the key as an ``OrderedDict`` key, which
    requires hashable + equality support.
    """

    dataset_name: str
    image_id: int
    # The configured model source: ``tagger_repo_id`` (HF download) or
    # ``tagger_model_path`` (local file). Only one is non-empty per
    # :attr:`TaggingService.is_configured`.
    model_id: str
    rating_threshold: float
    general_threshold: float
    character_threshold: float


@dataclass
class TagJobInfo:
    """Snapshot of a running (or recently finished) tagging job."""

    status: TagJobStatus
    dataset_name: str
    job_id: str = ""
    processed: int = 0
    total: int = 0
    errors: int = 0
    error: str | None = None
    error_messages: list[str] = field(default_factory=list)
    source: str = ""
    elapsed: float = 0.0


@dataclass
class CancelResult:
    """Outcome of a cancel request (see :meth:`TaggingService.cancel_async`)."""

    outcome: CancelOutcome
    killed: bool = False
    job_id: str = ""


class TaggerBusyError(Exception):
    """Raised when a swap is attempted while a batch job is running. Maps to HTTP 409.

    The UI surfaces the ``detail`` payload and points the user at the
    stop button for the offending dataset(s).
    """

    def __init__(self, detail: str) -> None:
        self.detail: str = detail
        super().__init__(detail)


class TaggerSwapInProgressError(Exception):
    """Raised when a swap is attempted while another swap is already in flight. Maps to HTTP 429.

    The optional ``retry_after_s`` hints at how long the second
    caller should wait before retrying; the UI surfaces it as the
    toast's retry guidance.
    """

    def __init__(self, detail: str, retry_after_s: float = 2.0) -> None:
        self.detail: str = detail
        self.retry_after_s: float = retry_after_s
        super().__init__(detail)


class TaggingService(Service):
    """Manages the tagger subprocess lifecycle and tagging requests.

    The subprocess is spawned on demand and torn down after
    ``Configuration.tagger_idle_timeout_seconds`` of inactivity. Set
    the timeout to ``0`` to keep the subprocess up indefinitely once
    it has been started.
    """

    def __init__(
        self,
        configuration: Configuration,
        logging: LoggingFactory,
        event_dispatcher: EventDispatcher,
        dataset_service: DatasetService,
        dataset_watcher: DatasetWatcherService,
        dataset_jobs: DatasetJobService,
        settings_service: SettingsService,
        tag_policy_service: TagPolicyService | None = None,
        job_scheduler: JobScheduler | None = None,
    ):
        self._configuration: Configuration = configuration
        self._logger: Logger = logging.get_logger(__name__)
        self._event_dispatcher: EventDispatcher = event_dispatcher
        self._dataset_service: DatasetService = dataset_service
        self._dataset_watcher: DatasetWatcherService = dataset_watcher
        self._dataset_jobs: DatasetJobService = dataset_jobs
        self._settings_service: SettingsService = settings_service
        # Optional so unit tests that build the service directly (no
        # DI) without a settings-backed policy store keep working:
        # ``None`` falls back to an empty policy at every call site.
        # Production wiring (auto-discovery) injects the real service.
        self._tag_policy_service: TagPolicyService | None = tag_policy_service
        self._job_scheduler: JobScheduler | None = job_scheduler

        # The user's persisted active-tagger selection (set via the
        # swap flow). Hydrated from ``SettingsService`` at construction
        # so a running subprocess picks up the saved choice on the
        # next spawn. ``None`` when no selection has been persisted
        # (fresh install or after a corrupt-settings warning). Future
        # phases wire this into ``_ensure_running_locked`` /
        # ``_source_label``; the fallback to the flat ``Configuration``
        # fields also lands in that phase.
        self._active_tagger: ActiveTagger | None = None

        self._tagger_client: TaggerClient | None = None
        # ``threading.Lock`` (not ``asyncio.Lock``) so the sync idle
        # check thread can acquire it without going through
        # ``run_coroutine_threadsafe``. Async request handlers acquire
        # it across an ``await``; that's acceptable because the actual
        # tag inference happens off-thread via ``asyncio.to_thread``,
        # so the event loop stays free for non-locking work.
        self._lifecycle_lock: threading.Lock = threading.Lock()
        # Atomic gate so concurrent swap requests don't both drain +
        # respawn. Distinct from ``_lifecycle_lock`` because the swap
        # holds the lifecycle lock for the full drain+spawn, while the
        # swap-in-progress lock is only for the HTTP-visible window.
        self._swap_in_progress_lock: threading.Lock = threading.Lock()
        # Holds a reference to the in-flight background swap task so the
        # event loop can't garbage-collect it mid-flight (a task that
        # isn't referenced elsewhere can be collected and cancelled). The
        # swap-in-progress lock is the real mutual-exclusion gate; this is
        # purely a keep-alive handle.
        self._swap_task: asyncio.Task[None] | None = None
        # ``None`` until the first request arrives. Used as the
        # "last touched" timestamp by the idle-check job.
        self._last_used_t: float | None = None
        # Monotonic timestamp set when a swap respawns the subprocess.
        # ``_idle_check_tick`` uses it to apply the shorter post-swap
        # timeout when no request has touched the subprocess yet
        # (``last_used_t <= swapped_at``); the first request clears it
        # so the normal idle timeout takes over.
        self._swapped_at: float | None = None

        # Batch tagging jobs (one per dataset). ``asyncio.Lock`` guards
        # the dict; the jobs themselves run as ``asyncio.Task``s.
        self._tag_jobs_lock: asyncio.Lock = asyncio.Lock()
        self._tag_jobs: dict[str, _TagJobState] = {}
        # Holds in-flight deferred expected-changes clears so the event
        # loop can't GC them mid-sleep. Entries self-remove on completion.
        self._pending_clears: set[asyncio.Task[None]] = set()

        # Bounded *bytes*-limited LRU cache for tag results. Keyed by
        # :class:`TaggerResultKey` (image identity + model + bucketed
        # thresholds + replace_underscores post-hoc). Persists across
        # requests and browser sessions so a fresh Tags tab loaded
        # after a batch job sees the existing result without re-running
        # the model. The fingerprint baked into the key means model /
        # threshold changes naturally hash to a different bucket — old
        # entries age out via LRU eviction without needing explicit
        # invalidation. ``tag_image`` is a read-through on this LRU:
        # a hit short-circuits the model run entirely (no subprocess
        # spawn, no byte read, no idle-timer reset), still dispatching
        # ``ImageTaggedEvent`` so SSE clients see the same success
        # signal a real run would emit (just with ``duration_ms=0``).
        # The byte budget (``tagger_result_max_memory_bytes``) keeps
        # the working set bounded for large-vocab models like the
        # animetimm ConvNeXt (~12k tags per result).
        self._tag_lock: asyncio.Lock = asyncio.Lock()
        self._tag_results: MemoryLRU[TaggerResultKey, TaggerResult] = MemoryLRU(
            self._configuration.tagger_result_max_memory_bytes,
            size_fn=tamer_result_size,
        )

        # Hydrate the persisted active-tagger selection last so the
        # LRU and other fields above are ready, and so a malformed
        # row can't take the service down. Warnings are logged; on
        # failure ``self._active_tagger`` stays ``None`` and the
        # service still operates against the flat ``Configuration``
        # fallback once Phase 2 wires that path.
        self._hydrate_active_tagger()

    # --- event handlers ---------------------------------------------------

    @event_handler(StartupEvent)
    async def on_startup(self, event: StartupEvent) -> None:  # pyright: ignore[reportUnusedParameter]
        if not self.is_configured:
            self._logger.debug("No tagger model configured (both tagger_model_path and tagger_repo_id empty); tagging endpoints will return 503.")
        if self._job_scheduler is not None:
            self._job_scheduler.new_scheduled_job(IDLE_CHECK_INTERVAL_SECONDS, self._idle_check_tick)

    @event_handler(ShutdownEvent)
    async def on_shutdown(self, event: ShutdownEvent) -> None:  # pyright: ignore[reportUnusedParameter]
        await self._acquire_lifecycle()
        try:
            await self._stop_locked_async()
        finally:
            self._lifecycle_lock.release()

    # --- public API -------------------------------------------------------

    @property
    def is_configured(self) -> bool:
        """Whether the tagger has a model source configured (i.e. can serve requests at all).

        True when either the persisted :class:`ActiveTagger` (whose
        validators guarantee a non-empty ``repo_id`` / ``model_path``)
        or the flat ``Configuration`` fallback fields provide a model.
        The 503/ready split is driven by this property in the
        controller.
        """
        if self._active_tagger is not None:
            return True
        return bool(self._configuration.tagger_model_path.strip()) or bool(self._configuration.tagger_repo_id.strip())

    @property
    def is_available(self) -> bool:
        """Whether the tagger subprocess is currently running."""
        return self._tagger_client is not None and self._tagger_client.is_alive

    # --- persisted active-tagger selection --------------------------------

    async def swap_active_model(self, selection: ActiveTagger) -> ActiveTagger:
        """Swap the active tagger selection end-to-end (validation up front, drain+respawn in the background).

        The full flow:

        1. Refuse with :class:`TaggerBusyError` (→ 409) if any batch
           job is currently running on any dataset. The active
           datasets are discovered via ``_tag_jobs``; a job is
           "running" while its ``asyncio.Task`` is unfinished, matching
           :meth:`is_tagging`.
        2. Refuse with :class:`TaggerSwapInProgressError` (→ 429 with
           a ``retry_after_s`` hint) if another swap is already in
           flight. ``_swap_in_progress_lock`` is an atomic gate
           distinct from ``_lifecycle_lock`` so a failing swap doesn't
           hold the lifecycle lock open.
        3. Return the intended selection immediately (HTTP 202) and
           hand off to :meth:`_swap_background`, which:
           a. Drains any in-flight single-image call (acquiring
              ``_lifecycle_lock`` naturally waits for it to release;
              ``_acquire_lifecycle`` polls with ``asyncio.sleep`` so the
              loop keeps running).
           b. Tears down the current subprocess via ``_stop_locked_async``,
              emitting stopping + stopped events with the **old** source
              label.
           c. Flips ``_active_tagger`` to the new selection.
           d. Respawns. On failure the in-memory selection is rolled
              back to the previous one and a best-effort rollback respawn
              runs so the service stays operative. If the rollback
              respawn also fails (a ``failed`` event is dispatched), the
              service is degraded but not wedged.
           e. Persists via ``SettingsService.set`` after a successful
              respawn. A persist failure logs and continues.

        Returning immediately (rather than awaiting the whole swap) keeps
        a slow first-run model download from blocking — and being
        cancelled by — the HTTP handler. The frontend follows progress via
        the existing ``tagger_status`` SSE stream (stopping/stopped → old
        label, then starting/ready/failed → new label) and re-fetches the
        active selection on a terminal state.
        """

        if self._active_tagger is not None and self._active_tagger == selection:
            self._logger.debug(
                "Swap is a no-op; active selection already matches. [source=%s]",
                selection.source_label,
            )
            return self._active_tagger

        busy = [name for name, state in self._tag_jobs.items() if state.task is not None and not state.task.done()]
        if busy:
            raise TaggerBusyError(f"a batch tagging job is running on {busy!r}; stop it before swapping the model")
        if not self._swap_in_progress_lock.acquire(blocking=False):
            raise TaggerSwapInProgressError(
                "another swap is already in progress",
                retry_after_s=2.0,
            )
        # Run the drain → stop → respawn → persist in the background so a
        # slow first-run model download (HuggingFace fetch inside the
        # worker's ``load_model``) can't block — and get cancelled by — the
        # HTTP request handler. The swap-in-progress lock stays held until
        # the background task finishes; progress is reported over the
        # existing ``tagger_status`` SSE stream (stopping/stopped with the
        # *old* label, then starting/ready/failed with the *new* one), so
        # the picker refreshes on state change without a long-hanging POST.
        # Hold the task reference on the instance so the event loop can't
        # garbage-collect a mid-flight swap.
        old_active = self._active_tagger
        self._swap_task = asyncio.create_task(self._swap_background(selection, old_active))
        return selection

    async def _swap_background(self, selection: ActiveTagger, old_active: ActiveTagger | None) -> None:
        """Background half of :meth:`swap_active_model`.

        Drains any in-flight single-image call, stops the old subprocess
        (SSE carries the *old* label), flips ``_active_tagger`` to the new
        selection, respawns (SSE carries the *new* label), and persists.
        On respawn failure the in-memory selection rolls back to the
        previous one and a best-effort rollback respawn runs; the outcome
        reaches the UI via the ``failed`` SSE event rather than a raised
        exception, since this runs detached from any HTTP handler.
        """
        try:
            await self._acquire_lifecycle()
            try:
                if self._tagger_client is not None:
                    await self._stop_locked_async()
                # Flip the selection *after* the old subprocess is torn down
                # so the stopping/stopped events carry the old label, then the
                # starting/ready events from the respawn carry the new one.
                self._active_tagger = selection
                # Mark the spawn as post-swap so the idle check applies
                # ``tagger_post_swap_idle_timeout_seconds`` until the first
                # request uses the subprocess. Spawn does NOT reset
                # ``_last_used_t`` — it keeps its prior value (``None`` on
                # first swap, or an older timestamp otherwise), which is
                # always ``<= swapped_at``, so the post-swap branch in
                # ``_idle_reference`` matches until a request bumps
                # ``_last_used_t`` past ``_swapped_at``.
                self._swapped_at = time.monotonic()
                try:
                    await self._ensure_running_locked()
                except Exception as exc:
                    self._active_tagger = old_active
                    self._swapped_at = None
                    self._logger.exception("Respawn with new model failed; rolling back.")
                    if old_active is not None:
                        try:
                            await self._ensure_running_locked()
                        except Exception:
                            self._logger.exception("Rollback respawn also failed.")
                            self._emit_status("failed", error=str(exc))
                    else:
                        self._emit_status("failed", error=str(exc))
                    return
                try:
                    self._settings_service.set("tagger.active_model", selection.model_dump())
                except Exception:
                    self._logger.exception("Failed to persist active tagger; in-memory state retained.")
                self._logger.info(
                    "Active tagger swapped. [kind=%s, source=%s]",
                    selection.kind,
                    selection.source_label,
                )
            finally:
                self._lifecycle_lock.release()
        except Exception:
            # Defensive catch-all: this task is detached from any request
            # handler, so an unhandled exception would only surface as a
            # logged ``task exception was never retrieved`` warning and leave
            # the swap-in-progress lock held. Surface it as a ``failed`` SSE
            # event and let the finally release the lock.
            self._logger.exception("Unexpected error during tagger swap.")
            self._emit_status("failed", error="unexpected error during tagger swap")
        finally:
            self._swap_in_progress_lock.release()

    @property
    def active_tagger(self) -> ActiveTagger | None:
        """The user's persisted active-tagger selection, or ``None``.

        ``None`` means no selection has been persisted yet (fresh
        install, after a schema-mismatch warning, or after the
        settings row was deleted). The SettingsDialog picker uses
        :attr:`effective_active_tagger` instead so legacy
        Configuration-only setups still see their model reflected.
        """
        return self._active_tagger

    @property
    def effective_active_tagger(self) -> ActiveTagger | None:
        """The active selection the SettingsDialog picker should display.

        Returns the persisted :class:`ActiveTagger` when one has been
        written via :meth:`set_active_tagger`. Otherwise synthesizes
        one from the flat ``Configuration`` fields so legacy setups
        (e.g. ``tagger_model_path`` set with no prior swap) show the
        right model in the picker without requiring the user to do a
        no-op swap to "activate" it. Returns ``None`` only when
        neither source is configured.
        """
        if self._active_tagger is not None:
            return self._active_tagger
        cfg = self._configuration
        repo_id = cfg.tagger_repo_id.strip()
        if repo_id:
            return ActiveTagger(
                kind="hf",
                repo_id=repo_id,
                repo_model_filename=cfg.tagger_repo_model_filename,
                repo_label_filename=cfg.tagger_repo_label_filename,
                preproc_profile=cfg.tagger_preproc_profile,
                default_size=cfg.tagger_default_input_size,
            )
        path = cfg.tagger_model_path.strip()
        if path:
            return ActiveTagger(
                kind="local",
                model_path=path,
                label_path=cfg.tagger_label_path,
                preproc_profile=cfg.tagger_preproc_profile,
                default_size=cfg.tagger_default_input_size,
            )
        return None

    def set_active_tagger(self, selection: ActiveTagger) -> None:
        """Persist the user's active-tagger selection.

        Stores the JSON form of *selection* under the
        ``tagger.active_model`` settings key, then updates the cached
        model in-place. Future phases (swap infrastructure + endpoints)
        drive this from ``POST /api/tagger/swap`` after draining any
        in-flight single-image call and refusing concurrent swaps.

        A subsequent ``TaggingService`` construction reads the same
        key back via :meth:`_hydrate_active_tagger` so the selection
        survives process restarts. The legacy flat ``Configuration``
        fields are **not** modified — they remain the initial defaults
        that this method overrides.
        """
        # Persist *first* so a crash mid-write doesn't leave the
        # cached model ahead of the on-disk row. ``SettingsService``
        # logs its own errors and returns silently on failure; we
        # still update the cached model so the in-process state is
        # usable, and log a warning so the cause is visible.
        self._settings_service.set("tagger.active_model", selection.model_dump())
        self._active_tagger = selection
        self._logger.info(
            "Active tagger persisted. [kind=%s, source=%s]",
            selection.kind,
            selection.source_label,
        )

    def _hydrate_active_tagger(self) -> None:
        """Read the persisted active-tagger selection from ``SettingsService``.

        Called at the end of :meth:`__init__`. Schema mismatches or
        I/O errors leave ``self._active_tagger`` as ``None`` and log
        a warning rather than aborting construction — the service can
        still operate against the flat ``Configuration`` fallback in
        later phases, and the user can re-save the selection from
        the UI.
        """
        try:
            raw = self._settings_service.get("tagger.active_model")
        except Exception as exc:
            self._logger.warning("Failed to read active tagger from settings: %s", exc)
            return
        if not isinstance(raw, dict):
            return
        try:
            self._active_tagger = ActiveTagger.model_validate(raw)
            self._logger.info(
                "Hydrated active tagger from settings. [kind=%s, source=%s]",
                self._active_tagger.kind,
                self._active_tagger.source_label,
            )
        except pydantic.ValidationError as exc:
            self._logger.warning(
                "Discarding malformed active tagger from settings: %s. The legacy Configuration fields will be used until a new selection is saved.",
                exc,
            )

    def _source_label(self) -> str:
        """Describe the configured model source for diagnostics / SSE payloads.

        Reads from the persisted :class:`ActiveTagger` selection when
        present, so the SSE ``source`` reflects whatever the user
        picked (and naturally updates when they swap). Falls back to
        the flat ``Configuration`` fields so legacy setups without a
        persisted selection keep emitting the standard format.

        Returns ``"hf:<repo_id>"`` for HuggingFace Hub downloads,
        ``"local:<path>"`` for local files, ``""`` when unconfigured.
        """
        if self._active_tagger is not None:
            return self._active_tagger.source_label
        repo_id = self._configuration.tagger_repo_id.strip()
        if repo_id:
            return f"hf:{repo_id}"
        path = self._configuration.tagger_model_path.strip()
        if path:
            return f"local:{path}"
        return ""

    def _emit_status(self, state: TaggerState, error: str | None = None) -> None:
        """Dispatch a ``TaggerStatusEvent``. Safe to call without a running event loop."""
        self._event_dispatcher.dispatch(TaggerStatusEvent(state=state, source=self._source_label(), error=error))

    def _tag_result_key(
        self,
        dataset_name: str,
        image_id: int,
        thresholds: TaggingThresholds,
    ) -> TaggerResultKey:
        """Build a :class:`TaggerResultKey` from the current config + effective thresholds.

        ``thresholds`` is the *effective* value (per-request override or
        config default) — the caller's responsibility. The general /
        character thresholds are floored to the cache-bucket step (see
        :func:`bucket_threshold`); rating is always recorded as ``0``
        because we never pre-filter ratings at write time (the request's
        effective rating is reapplied at retrieval instead).

        The always-add / banned policy is intentionally **not** part of
        the key — it's a read-time transform (see :meth:`_refilter`),
        so a single slot serves every policy variant.

        Two keys with different effective values floor to different
        buckets (and never collide); keys with equal floors and the
        same model collide and reuse the slot.
        """
        cfg = self._configuration
        if self._active_tagger is not None:
            # Same string the SSE event will carry (see
            # :meth:`_source_label`); collapses to ``hf:<id>`` or
            # ``local:<path>``. Swapping the model therefore
            # invalidates existing cache slots naturally because the
            # new selection hashes to a different bucket.
            model_id = self._active_tagger.source_label
        else:
            model_id = cfg.tagger_repo_id.strip() or cfg.tagger_model_path.strip()
        return TaggerResultKey(
            dataset_name=dataset_name,
            image_id=image_id,
            model_id=model_id,
            # Always 0 — we never pre-filter ratings at write time;
            # the request's effective rating is reapplied at retrieval.
            rating_threshold=0.0,
            # Floored so nearby request thresholds share a slot.
            general_threshold=bucket_threshold(thresholds.general),
            character_threshold=bucket_threshold(thresholds.character),
        )

    @staticmethod
    def _refilter(
        cached: TaggerResult,
        eff_thresholds: TaggingThresholds,
        eff_replace: bool,
        policy: TagPolicy | None = None,
    ) -> TaggerResult:
        """Re-apply the request's effective thresholds + policy + ``replace_underscores`` to a cached value.

        The cached value was filtered at the bucket floors
        (``rating=0``, ``general=bucket_general``, ``character=bucket_character``)
        at write time, which is always a permissive superset of any
        stricter request. Re-applying ``apply_thresholds`` at the
        request's effective values can therefore only drop more tags,
        never add any. The always-add / banned policy is then applied
        via :func:`apply_policy` (injects missing always-add tags at
        1.0, drops banned ones) and ``replace_underscores`` is a
        string transform applied last so the cache can serve both
        with / without it and every policy variant from one slot.
        """
        re_filtered = apply_thresholds(
            cached,
            rating_threshold=eff_thresholds.rating,
            general_threshold=eff_thresholds.general,
            character_threshold=eff_thresholds.character,
        )
        if policy is not None:
            re_filtered = apply_policy(re_filtered, policy)
        if eff_replace:
            re_filtered = replace_underscores_in(re_filtered)
        # User-edited selection is independent of thresholds / underscores / policy —
        # carry it through the refilter so reads don't wipe it. (The
        # transform helpers above build fresh results and omit it.)
        re_filtered.customizations = cached.customizations
        return re_filtered

    async def get_tag_result(
        self,
        dataset_name: str,
        image_id: int,
        thresholds: TaggingThresholds | None = None,
        replace_underscores: bool | None = None,
        policy: TagPolicy | None = None,
    ) -> TaggerResult | None:
        """Return the cached tag result for an image, re-applied at the caller's effective settings.

        ``thresholds`` and ``replace_underscores`` are the effective
        values the caller will display (so it lands in the same cache
        bucket as the entry that was originally written). When omitted
        the config defaults are used — matching what :meth:`tag_image`
        falls back to when no per-request override is supplied.

        ``policy`` is the request's intent for always-add / banned
        tags; it's applied as a read-time transform (see
        :func:`apply_policy`) and is **not** part of the cache key, so
        toggling a policy tag shows up on the next read of an existing
        slot without re-tagging. When ``None`` (the controller path
        after the policy move), :class:`TagPolicyService` resolves the
        per-dataset stored policy. When ``policy`` is supplied
        explicitly (test / preview path), it wins.

        The cached value was filtered at the bucket floors, so we
        re-apply the *effective* thresholds (which are always ≥ bucket
        floors for a matching key) on the way out. A request whose
        effective threshold is below the bucket floor (would want to
        keep tags the cache has already filtered out) misses entirely.

        Returns ``None`` when no entry matches the bucketed key (either
        never tagged, evicted by LRU, or tagged under different
        bucketed thresholds / model).
        """
        eff_thresholds = thresholds or TaggingThresholds(
            rating=self._configuration.tagger_rating_threshold,
            general=self._configuration.tagger_general_threshold,
            character=self._configuration.tagger_character_threshold,
        )
        eff_replace = replace_underscores if replace_underscores is not None else self._configuration.tagger_replace_underscores
        eff_policy = self._resolve_policy(dataset_name, override=policy)
        key = self._tag_result_key(dataset_name, image_id, eff_thresholds)
        async with self._tag_lock:
            cached = self._tag_results.get(key)
        if cached is None:
            return None
        return self._refilter(cached, eff_thresholds, eff_replace, eff_policy)

    async def evict_tag_result(
        self,
        dataset_name: str,
        image_id: int,
        thresholds: TaggingThresholds | None = None,
    ) -> bool:
        """Drop the cached tag result for an image (best-effort).

        Returns ``True`` when an entry was removed, ``False`` when no
        matching entry was present (no error). The fingerprint rules
        match :meth:`get_tag_result` (modulo the bucketing — evicting
        drops only the bucketed-key slot, leaving any sibling slots
        for the same image under different bucketed thresholds
        alone).
        """
        eff_thresholds = thresholds or TaggingThresholds(
            rating=self._configuration.tagger_rating_threshold,
            general=self._configuration.tagger_general_threshold,
            character=self._configuration.tagger_character_threshold,
        )
        key = self._tag_result_key(dataset_name, image_id, eff_thresholds)
        async with self._tag_lock:
            try:
                del self._tag_results[key]
            except KeyError:
                return False
            return True

    async def set_tag_customizations(
        self,
        dataset_name: str,
        image_id: int,
        customizations: TagCustomizations,
        thresholds: TaggingThresholds | None = None,
    ) -> bool:
        """Attach a user-edited selection to the cached result for an image (best-effort).

        Customizations ride on the cached :class:`TaggerResult` under
        the same key as the model output (image + model + bucketed
        thresholds), so a model / settings change naturally orphans
        them — the user's edits apply only to the tagged result they
        were made against. The always-add / banned policy is not part
        of the key, so policy changes don't orphan customizations.
        Returns ``True`` when an entry was updated, ``False`` when no
        slot matches (never tagged / evicted / stale settings) — a
        miss is silent: there is simply nothing to remember the
        selection against yet.
        """
        eff_thresholds = thresholds or TaggingThresholds(
            rating=self._configuration.tagger_rating_threshold,
            general=self._configuration.tagger_general_threshold,
            character=self._configuration.tagger_character_threshold,
        )
        key = self._tag_result_key(dataset_name, image_id, eff_thresholds)
        async with self._tag_lock:
            cached = self._tag_results.get(key)
            if cached is None:
                return False
            cached.customizations = customizations
            # Re-insert so the size function re-measures (customizations
            # changed the entry's byte cost) and the entry is promoted.
            self._tag_results[key] = cached
            return True

    async def tag_image(
        self,
        dataset_name: str,
        image_info: ImageInfo,
        thresholds: TaggingThresholds | None = None,
        replace_underscores: bool | None = None,
        source: str | None = None,
        job_id: str | None = None,
        policy: TagPolicy | None = None,
    ) -> TaggerResult:
        """Tag a single image.

        The subprocess is started on demand if not already running.
        The idle timer is reset at the start of the request so an
        in-flight request can't be torn down mid-flight. When the same
        image + model + bucketed thresholds already have a cached
        result, the LRU hit short-circuits the model run (no subprocess
        spawn, no byte read, no idle-timer reset) and returns the
        cached value re-applied at the request's effective thresholds
        and ``replace_underscores``.

        Per-image SSE events are paired with the actual model run on
        the cold path only — ``ImageTagStartedEvent`` before the
        subprocess call and ``ImageTaggedEvent`` after. Cache hits
        dispatch neither (the data is unchanged from the cold-path
        write that populated the cache; firing them would flood the
        SSE queue at the rate the loop can produce results for an
        all-cache-hit batch). The HTTP response carries the
        (effective-filtered) result for the initiating request, and
        ``ImageTagErrorEvent`` still fires on cold-path failure.

        The caller is responsible for resolving ``image_info`` (e.g.
        via ``DatasetService.get_image``) and verifying the on-disk
        file is present — this service trusts its inputs and focuses
        on the tag lifecycle.

        Args:
            source: Optional frontend-supplied label for the
                tagging source (e.g. a model name chosen from a
                dropdown). Included in the dispatched events so other
                tabs / clients can display what the requester used.
                Falls back to the server's configured model when not
                provided.
            job_id: Optional batch-job id carried into
                ``ImageTagStartedEvent`` so the frontend can
                correlate per-image events with the originating job.
                ``None`` (default) for single-image callers.

        Raises:
            RuntimeError: If the tagger is not configured, or the
                subprocess fails to start.
        """
        if not self.is_configured:
            raise RuntimeError("Tagger is not configured on this server")

        # Resolve effective thresholds + replace_underscores once —
        # the cache key is the *bucketed* thresholds, but the read-
        # through path re-applies the *effective* (request) thresholds
        # on the cached value before returning.
        eff_thresholds = thresholds or TaggingThresholds(
            rating=self._configuration.tagger_rating_threshold,
            general=self._configuration.tagger_general_threshold,
            character=self._configuration.tagger_character_threshold,
        )
        eff_replace = replace_underscores if replace_underscores is not None else self._configuration.tagger_replace_underscores
        eff_policy = self._resolve_policy(dataset_name, override=policy)
        cache_key = self._tag_result_key(dataset_name, image_info.id, eff_thresholds)

        # --- Read-through cache: skip the model when the LRU already
        # holds a result for this bucketed key. The cached value was
        # post-thresholded at the bucket floors (≥ the request's
        # effective floors) at write time, so re-applying the request's
        # effective thresholds + ``replace_underscores`` on the way out
        # yields the same data the cold path would produce. Still
        # dispatch ``ImageTaggedEvent`` so SSE clients see the same
        # success signal a real model run would emit.
        async with self._tag_lock:
            cached = self._tag_results.get(cache_key)
        if cached is not None:
            re_filtered = self._refilter(cached, eff_thresholds, eff_replace, eff_policy)
            # Yield once so the event loop can run other ready tasks
            # (idle check, in-flight cancellations) between back-to-back
            # cache hits in a batch. ``_refilter`` is the closest thing
            # we do to CPU-blocking when the model isn't running — for
            # animetimm-sized caches it's milliseconds per call, which
            # compounds across a batch of all-hit images.
            await asyncio.sleep(0)
            # No per-image SSE dispatch on hits: a burst of all-cache
            # re-tags (e.g. 41 images in a few ms) would saturate
            # per-client SSE queues at the rate the loop can produce
            # results. Frontends consume the data through the HTTP
            # response (single-image POST) or
            # via ``fetchCachedTagResult`` (cold-mount on Tags tab); the
            # job terminal state (``TagJobStatusEvent`` with terminal
            # status) sweeps any leftover inflight per dataset.
            return re_filtered

        # The ``source`` label for SSE events. Frontend-supplied
        # value wins; otherwise fall back to the configured model.
        event_source = source if source else self._source_label()

        image_bytes = Path(image_info.path).read_bytes()
        start_t = time.monotonic()

        try:
            # Acquire without blocking the loop (see ``_acquire_lifecycle``)
            # so cancelling an in-flight tag can't deadlock the unwind.
            await self._acquire_lifecycle()
            try:
                await self._ensure_running_locked()
                assert self._tagger_client is not None  # ensured above
                client = self._tagger_client
                # Reset the idle timer while holding the lock so the
                # periodic idle check sees an in-use server.
                self._last_used_t = time.monotonic()
                # First request after a swap: drop the post-swap
                # timestamp so the idle check uses the normal timeout
                # from this point on. ``last_used_t`` is already
                # past ``_swapped_at`` (the request happened after
                # spawn), so the check would switch anyway — clearing
                # keeps the state honest for future respawns.
                if self._swapped_at is not None:
                    self._swapped_at = None
                # Dispatch ``ImageTagStartedEvent`` only on the cold
                # path so cache hits don't fire per-image events (which
                # would flood the SSE queue at the rate the loop can
                # produce results for an all-cache-hit batch).
                self._event_dispatcher.dispatch(
                    ImageTagStartedEvent(
                        dataset_name=dataset_name,
                        job_id=job_id or "",
                        image_id=image_info.id,
                        file_name=image_info.file_name,
                    )
                )
                result = await client.tag(image_bytes)
            finally:
                self._lifecycle_lock.release()
        except Exception as exc:
            duration_ms = int((time.monotonic() - start_t) * 1000)
            self._event_dispatcher.dispatch(
                ImageTagErrorEvent(
                    dataset_name=dataset_name,
                    image_id=image_info.id,
                    error=str(exc),
                    source=event_source,
                    duration_ms=duration_ms,
                )
            )
            raise

        # Cache filtered at the bucket floors — a superset of any
        # stricter request (whose effective floors are ≥ bucket floors).
        # ``replace_underscores`` and the always-add / banned policy are
        # NOT applied to the cache value: both are read-time transforms
        # applied post-hoc at retrieval so the slot can serve every
        # variant of either without re-tagging.
        bucketed = apply_thresholds(
            result,
            rating_threshold=cache_key.rating_threshold,
            general_threshold=cache_key.general_threshold,
            character_threshold=cache_key.character_threshold,
        )

        # Effective (request-specific) for caller + dispatched event.
        # Re-apply on top of the bucketed cache value; since effective
        # thresholds are always ≥ the bucket floors for a matching key,
        # this only drops more tags, never adds any. The policy is
        # applied here too (injects always-add, drops banned) so the
        # returned result and SSE event match what the caller asked for.
        effective = self._refilter(bucketed, eff_thresholds, False, eff_policy)  # replace applied below

        duration_ms = int((time.monotonic() - start_t) * 1000)
        self._event_dispatcher.dispatch(
            ImageTaggedEvent(
                dataset_name=dataset_name,
                image_id=image_info.id,
                file_name=image_info.file_name,
                path=image_info.path,
                tags=effective.tags,
                categories=effective.categories,
                source=event_source,
                duration_ms=duration_ms,
            )
        )
        # Cache the bucketed value for next read (Tags tab on a fresh
        # page load, or any subsequent POST under the same bucketed
        # thresholds — different per-request thresholds within the same
        # bucket land in the same slot and re-filter on the way out).
        async with self._tag_lock:
            self._tag_results[cache_key] = bucketed

        if eff_replace:
            effective = replace_underscores_in(effective)
        return effective

    async def tag_single_image_async(
        self,
        dataset_name: str,
        image_info: ImageInfo,
        thresholds: TaggingThresholds | None = None,
        replace_underscores: bool | None = None,
        source: str | None = None,
    ) -> TaggerResult:
        """Tag a single image, gated by the cross-service dataset coordinator.

        Wraps :meth:`tag_image` (which is also called internally by batch
        jobs) with a transient claim so a single-image tag can't overlap a
        captioning run — or vice versa — on the same dataset. The
        always-add / banned policy is resolved by :meth:`tag_image` from
        :class:`TagPolicyService` for the dataset — there is no per-call
        policy override at this layer.

        Raises:
            DatasetBusyError: if another job kind holds the dataset (→ 409).
            RuntimeError: if the tagger is not configured / fails to start.
        """
        claim = await self._dataset_jobs.try_acquire(dataset_name, "tagging", uuid.uuid4().hex[:12])
        try:
            return await self.tag_image(
                dataset_name,
                image_info,
                thresholds=thresholds,
                replace_underscores=replace_underscores,
                source=source,
            )
        finally:
            await self._dataset_jobs.release(claim)

    # --- lifecycle helpers (callers must hold ``_lifecycle_lock``) -------

    async def _ensure_running_locked(self) -> None:
        """Spawn the tagger subprocess if not already running.

        The subprocess kwargs are sourced from
        :meth:`_subprocess_spawn_args`, which prefers the persisted
        :class:`ActiveTagger` selection and falls back to the flat
        ``Configuration`` fields.

        Raises:
            RuntimeError: if the subprocess fails to start, or if the
                configured preproc profile name is not recognised.
        """
        if self._tagger_client is not None and self._tagger_client.is_alive:
            return

        # Either no client yet, or the previous one died. Build a
        # fresh client and start it. We do NOT reuse the dead client
        # — its multiprocessing queues are stale.
        tagger_kwargs, model_path = self._subprocess_spawn_args()

        self._emit_status("starting")

        client = TaggerClient(
            OnnxTagger,
            model_path,
            tagger_kwargs,
            heartbeat_interval=self._configuration.tagger_heartbeat_interval_seconds,
            response_timeout=self._configuration.tagger_response_timeout_seconds,
            start_timeout=self._configuration.tagger_startup_timeout_seconds,
            poll_interval=self._configuration.tagger_liveness_poll_seconds,
        )
        try:
            await client.start()
        except Exception as exc:
            self._logger.exception("Failed to start tagger process.")
            self._emit_status("failed", error=str(exc))
            raise RuntimeError("Failed to start tagger process — check server logs") from None

        self._tagger_client = client
        labels_desc = tagger_kwargs.get("label_path") or ("<downloaded from HF>" if "repo_id" in tagger_kwargs else "<none>")
        self._logger.info(
            "Tagger process started. [source=%s, labels=%s]",
            self._source_label(),
            labels_desc,
        )
        self._emit_status("ready")

    def _subprocess_spawn_args(self) -> tuple[dict[str, Any], str]:
        """Build the spawn kwargs + ``model_path`` for ``_ensure_running_locked``.

        Prefers the persisted :class:`ActiveTagger` selection when
        one has been written via :meth:`set_active_tagger`. Falls
        back to the flat ``Configuration`` fields so setups without a
        persisted selection (fresh installs, after a corrupt-settings
        warning) keep working until the user picks a model from the
        UI.

        Both paths resolve the preprocessing profile by name (raising
        ``RuntimeError`` on an unknown name) and forward a positive
        ``default_size`` override. The HF path leaves ``model_path``
        empty — ``OnnxTagger.load_model`` ignores it once ``repo_id``
        is set in kwargs.

        For known catalog repos, ``repo_sidecar_filenames`` is set
        from :data:`yadc.api.modules.tagger_catalog.KNOWN_TAGGER_MODELS`
        so extra files (typically the ONNX external-data file for
        >2 GB models) are downloaded best-effort alongside the model.
        """
        from yadc.taggers.onnx_preprocess import get_profile

        tagger_kwargs: dict[str, Any] = {}
        model_path = ""
        label_path: str | None = None
        profile_name: str
        default_size: int

        if self._active_tagger is not None:
            selection = self._active_tagger
            profile_name = selection.preproc_profile
            default_size = selection.default_size
            if selection.kind == "hf":
                tagger_kwargs["repo_id"] = selection.repo_id
                tagger_kwargs["repo_model_filename"] = selection.repo_model_filename
                tagger_kwargs["repo_label_filename"] = selection.repo_label_filename
                sidecars = self._catalog_sidecars(selection.repo_id)
                if sidecars:
                    tagger_kwargs["repo_sidecar_filenames"] = sidecars
            else:
                model_path = selection.model_path
                explicit = selection.label_path.strip()
                label_path = explicit or self._resolve_label_path(model_path)
        else:
            # Legacy fallback — flat Configuration fields. Kept
            # verbatim so pre-swap installations behave identically.
            model_path = self._configuration.tagger_model_path
            profile_name = self._configuration.tagger_preproc_profile
            default_size = self._configuration.tagger_default_input_size
            repo_id = self._configuration.tagger_repo_id.strip()
            if repo_id:
                tagger_kwargs["repo_id"] = repo_id
                tagger_kwargs["repo_model_filename"] = self._configuration.tagger_repo_model_filename
                tagger_kwargs["repo_label_filename"] = self._configuration.tagger_repo_label_filename
                sidecars = self._catalog_sidecars(repo_id)
                if sidecars:
                    tagger_kwargs["repo_sidecar_filenames"] = sidecars
            else:
                label_path = self._resolve_label_path(model_path)

        if label_path:
            tagger_kwargs["label_path"] = label_path
        try:
            tagger_kwargs["preproc_profile"] = get_profile(profile_name)
        except ValueError as exc:
            raise RuntimeError(f"Invalid tagger_preproc_profile: {exc}") from None
        if default_size > 0:
            tagger_kwargs["default_size"] = default_size
        return tagger_kwargs, model_path

    async def _stop_locked_async(self) -> None:
        """Stop the tagger subprocess (async). Used from ``on_shutdown``."""
        if self._tagger_client is None:
            return
        self._emit_status("stopping")
        client = self._tagger_client
        # Clear the state first so a new request arriving while
        # ``client.stop()`` is in flight sees "no client" and spawns
        # a fresh subprocess. The lock (held by the caller) prevents
        # a concurrent spawn from racing this stop.
        self._tagger_client = None
        self._last_used_t = None
        self._swapped_at = None
        try:
            await client.stop()
        except Exception:
            self._logger.exception("Error stopping tagger process.")
            self._emit_status("failed", error="stop failed")
        else:
            self._logger.info("Tagger process stopped.")
            self._emit_status("stopped")

    def _stop_locked_sync(self) -> None:
        """Stop the tagger subprocess (sync). Used from the idle check thread.

        Blocks the calling thread until the subprocess exits (up to
        ~15s with the default ``TaggerServer.stop`` join/terminate
        timeouts). Fine because the caller is a daemon thread.

        SSE dispatch is still safe from this thread because
        ``EventDispatcher.dispatch`` walks subscribers synchronously;
        the SSE queue push is sync.
        """
        if self._tagger_client is None:
            return
        self._emit_status("stopping")
        client = self._tagger_client
        self._tagger_client = None
        self._last_used_t = None
        self._swapped_at = None
        try:
            # ``TaggerServer.stop`` is sync; calling it via
            # ``client._server`` skips the ``asyncio.to_thread`` in
            # ``TaggerClient.stop``. This is safe here because we
            # are already in a non-event-loop thread.
            client._server.stop()
        except Exception as exc:
            self._logger.exception("Error stopping tagger process.")
            self._emit_status("failed", error=str(exc))
        else:
            self._logger.info("Tagger process stopped.")
            self._emit_status("stopped")

    async def _acquire_lifecycle(self) -> None:
        """Acquire the lifecycle lock without blocking the event loop.

        ``threading.Lock.acquire`` is a blocking call; doing it directly on
        the loop thread would freeze the loop and deadlock the unwind of an
        in-flight tag we're cancelling (the loop must run that coroutine so
        it can release the lock). Poll with a non-blocking acquire plus
        ``await asyncio.sleep`` so the loop keeps running while we wait.
        """
        while not self._lifecycle_lock.acquire(blocking=False):
            await asyncio.sleep(0.05)

    def _resolve_label_path(self, model_path: str) -> str | None:
        """Pick the label file, preferring the configured one and falling back to ``<model_dir>/selected_tags.csv``."""
        configured = self._configuration.tagger_label_path.strip()
        if configured:
            return configured
        candidate = Path(model_path).parent / "selected_tags.csv"
        return str(candidate) if candidate.exists() else None

    def _catalog_sidecars(self, repo_id: str) -> list[str]:
        """Sidecar filenames to fetch alongside a known catalog repo's model.

        Returns the row's ``sidecars`` list when ``repo_id`` matches a
        curated entry in :data:`KNOWN_TAGGER_MODELS`; ``[]`` otherwise
        (custom HF repos, legacy Configuration-driven setups). The
        catalog is the single source of truth for which extra files a
        curated model needs — typically the ONNX external-data file
        (``model.onnx_data``) for models that exceed protobuf's 2 GB
        size limit.
        """
        for entry in KNOWN_TAGGER_MODELS:
            if entry.id == repo_id:
                return list(entry.sidecars)
        return []

    # --- idle check (called by JobScheduler in a daemon thread) ----------

    def _idle_reference(self) -> tuple[float, float] | None:
        """Pick the reference timestamp + timeout for the idle check.

        Returns ``(reference_timestamp, timeout_seconds)``, or ``None``
        if there's nothing to time against (no request and no swap
        since startup). The shorter ``tagger_post_swap_idle_timeout_seconds``
        applies while a post-swap subprocess hasn't been touched by a
        request yet — i.e. ``_last_used_t`` is ``None`` or still older
        than ``_swapped_at``. The first request clears ``_swapped_at``,
        after which the normal ``tagger_idle_timeout_seconds`` governs.
        """
        last_used = self._last_used_t
        swapped_at = self._swapped_at
        # Normal window: a request has bumped the idle timer past the
        # swap point (or there was no swap). The ``is not None`` guard
        # narrows ``last_used`` for the checker.
        if last_used is not None and (swapped_at is None or last_used > swapped_at):
            return last_used, self._configuration.tagger_idle_timeout_seconds
        # Post-swap window: spawn happened but no request has used the
        # subprocess yet (``_last_used_t`` is still ``None`` or predates
        # the swap).
        if swapped_at is not None:
            return swapped_at, self._configuration.tagger_post_swap_idle_timeout_seconds
        return None

    def _idle_check_tick(self) -> None:
        """Stop the subprocess if it has been idle longer than the configured timeout.

        In-flight requests and recent activity reset the idle timer,
        so this never tears down a server that's actively serving.
        ``tagger_idle_timeout_seconds <= 0`` disables teardown
        entirely (the subprocess stays up once started).

        After a swap, the shorter ``tagger_post_swap_idle_timeout_seconds``
        applies — but only while the subprocess hasn't been touched
        by a request yet. The first request clears ``_swapped_at`` so
        the normal timeout takes over. See :meth:`_idle_reference`
        for how the reference timestamp + timeout are selected.
        """
        ref = self._idle_reference()
        if ref is None:
            return  # never used since startup, never swapped
        reference, timeout = ref
        if timeout <= 0 or time.monotonic() - reference < timeout:
            return

        # Non-blocking acquire: if an async request is currently
        # holding the lock, skip this tick — the request will reset
        # the timer and the next tick will see the fresh timestamp.
        if not self._lifecycle_lock.acquire(blocking=False):
            return
        try:
            # Re-check everything under the lock — state may have
            # changed while we were waiting.
            if self._tagger_client is None:
                return
            # Recompute the reference under the lock (the request
            # path may have just bumped ``_last_used_t``).
            ref = self._idle_reference()
            if ref is None:
                return
            reference, timeout = ref
            if timeout <= 0 or time.monotonic() - reference < timeout:
                return
            if not self._tagger_client.is_alive:
                # Subprocess died but the client object lingered.
                # Clean up state; next request will respawn.
                self._tagger_client = None
                self._last_used_t = None
                self._swapped_at = None
                return
            idle_for = time.monotonic() - reference
            self._logger.info("Tagger idle for %.0fs, stopping. [timeout=%ss]", idle_for, timeout)
            self._stop_locked_sync()
        finally:
            self._lifecycle_lock.release()

    # --- batch tagging jobs ----------------------------------------------

    async def start_tag_job_async(self, dataset_name: str, options: TagJobOptions) -> TagJobInfo:
        """Start a batch tagging job for *dataset_name*.

        At most one tagging job per dataset may run at a time. Raises
        :class:`ValueError` (→ 409) if one is already running. The job
        runs as an ``asyncio.Task``; per-image results reach clients via
        ``ImageTaggedEvent`` / ``ImageTagErrorEvent`` and aggregate
        progress via ``TagJobStatusEvent``.
        """
        if not self.is_configured:
            raise RuntimeError("Tagger is not configured on this server")

        job_id = uuid.uuid4().hex[:12]
        # Resolve the image set synchronously so the initial status is
        # accurate and config/lookup errors surface as 4xx before the
        # background task starts (mirrors captioning preflight).
        images = self._resolve_job_images(dataset_name, options.image_ids)

        async with self._tag_jobs_lock:
            existing = self._tag_jobs.get(dataset_name)
            if existing is not None and existing.task is not None and not existing.task.done():
                raise ValueError(f"A tagging job is already running for dataset '{dataset_name}'")

            # Cross-service gate: at most one sidecar-writing job per
            # dataset. Raises DatasetBusyError (a ValueError) if a
            # captioning job — or any future job kind — holds it.
            claim = await self._dataset_jobs.try_acquire(dataset_name, "tagging", job_id)

            # Preflight: partition out images the overwrite=False skip
            # would drop anyway (mirrors captioning preflight).
            to_do, skipped = self._filter_skipped_images(dataset_name, images, options.save)

            # Nothing to tag (e.g. overwrite=False and every image
            # already has saved tags). Return done immediately WITHOUT
            # spawning a task or registering expected changes — the
            # HTTP response carries "done", so there's no SSE event to
            # race with the response. Mirrors captioning's no-op path.
            if not to_do:
                await self._dataset_jobs.release(claim)
                return TagJobInfo(
                    status="done",
                    dataset_name=dataset_name,
                    job_id=job_id,
                    processed=skipped,
                    total=skipped,
                    errors=0,
                    source=options.source or self._source_label(),
                )

            state = _TagJobState(
                job_id=job_id,
                dataset_name=dataset_name,
                source=options.source or self._source_label(),
                total=len(to_do) + skipped,
                thresholds=self._thresholds_from_options(options),
                replace_underscores=options.replace_underscores,
                save=options.save,
                # Resolve the per-dataset policy once at job start —
                # the policy is read-time-only at the cache layer, so
                # capturing a snapshot here keeps behaviour consistent
                # across all images in the batch even if the user edits
                # the policy mid-run.
                policy=self._resolve_policy(dataset_name),
                claim=claim,
                # Seed progress with the preflight skip count so the bar
                # starts at the right position (e.g. 40/50 when 40 were
                # already tagged).
                processed=skipped,
            )
            self._tag_jobs[dataset_name] = state
            # Tag filesystem change events with the job_id so the
            # initiating frontend tab can suppress the redundant
            # dataset_changed refresh (mirrors captioning). Cleared by a
            # deferred task scheduled in ``_run_tag_job``'s finally — see
            # ``_deferred_clear_expected_changes`` for why the clear is
            # delayed past the loop end.
            self._dataset_watcher.expect_changes(dataset_name, job_id)
            state.task = asyncio.create_task(self._run_tag_job(dataset_name, state, to_do))

        return await self.get_tag_job_status_async(dataset_name)

    async def stop_tag_job_async(self, dataset_name: str) -> bool:
        """Request graceful cancellation of the running tagging job.

        Returns ``False`` if no job is running.
        """
        async with self._tag_jobs_lock:
            state = self._tag_jobs.get(dataset_name)
            if state is None or state.task is None or state.task.done():
                return False
            state.stop_event.set()
        # Don't await the task — the controller returns immediately and
        # the job transitions to "cancelled" via its own finally block.
        return True

    async def cancel_async(self, job_id: str | None = None) -> CancelResult:
        """Cancel a tagging job and/or interrupt an in-flight tag.

        Graceful-first: when *job_id* matches a running batch job, set its
        ``stop_event`` (the loop breaks at the next image boundary). If an
        inference is still in flight after ``tagger_cancel_grace_seconds``,
        escalate to force-killing the subprocess — ONNX ``session.run``
        can't be interrupted mid-call except by ending the process.

        With *job_id* omitted (the single-image synchronous-tag case),
        there is no job to stop — cancel goes straight to the escalation
        check and kills the process if a request is mid-inference.

        Returns the outcome; see :class:`CancelResult`.
        """
        # 1. Graceful — set the matched job's stop_event.
        stopped_job_id = ""
        if job_id is not None:
            async with self._tag_jobs_lock:
                match: _TagJobState | None = None
                for state in self._tag_jobs.values():
                    if state.job_id == job_id:
                        match = state
                        break
                if match is None:
                    return CancelResult(outcome="stale_job")
                if match.task is None or match.task.done():
                    return CancelResult(outcome="nothing_running", job_id=match.job_id)
                match.stop_event.set()
                stopped_job_id = match.job_id

        # 2. Escalation — is an inference in flight right now? The
        #    lifecycle lock is held across ``await client.tag()``, so
        #    held == mid-inference. Free means the job is between images
        #    (or nothing is running): the stop_event catches it; no kill.
        if self._lifecycle_lock.acquire(blocking=False):
            self._lifecycle_lock.release()
            return CancelResult(outcome="stopped" if stopped_job_id else "nothing_running", job_id=stopped_job_id)

        # Held — give it a grace window to finish on its own (small images
        # may complete), then force-kill.
        await asyncio.sleep(self._configuration.tagger_cancel_grace_seconds)
        if self._lifecycle_lock.acquire(blocking=False):
            self._lifecycle_lock.release()
            return CancelResult(outcome="stopped", job_id=stopped_job_id)

        killed = await self._force_kill_async()
        return CancelResult(outcome="killed" if killed else "stopped", killed=killed, job_id=stopped_job_id)

    async def _force_kill_async(self) -> bool:
        """Terminate the subprocess to interrupt an in-flight inference, then
        finalize state before returning.

        Returns ``True`` if a live subprocess was killed; ``False`` if there
        was nothing live to kill. Emits the stopping/stopped status events
        on a successful kill. After terminating, waits (off the event loop,
        bounded) for the in-flight request to unwind and release the
        lifecycle lock, then clears the dead client ref under it. Waiting
        off-loop is essential: the unwind coroutine runs on the loop, so
        the loop must stay free for it to proceed.
        """
        client = self._tagger_client
        if client is None or not client.is_alive:
            return False
        self._emit_status("stopping")
        # ``client.kill`` is async (it off-threads the sync ``server.kill``);
        # await it directly rather than wrapping in ``to_thread``.
        await client.kill()  # process is dead after this

        def _wait_and_clear() -> bool:
            # Block (in a worker thread) until the killed worker's in-flight
            # ``tag()`` surfaces "worker died" (~poll_interval) and its
            # handler releases the lock. Bounded so a truly wedged worker
            # can't hang cancel forever; the self-healing in
            # ``_ensure_running_locked`` handles a leftover stale ref.
            timeout = max(5.0, self._configuration.tagger_liveness_poll_seconds * 10)
            acquired = self._lifecycle_lock.acquire(timeout=timeout)
            if not acquired:
                return False
            try:
                # Only clear if it's still the client we killed — a request
                # that arrived after the kill may have respawned a fresh one.
                if self._tagger_client is client:
                    self._tagger_client = None
                    self._last_used_t = None
                    return True
                return False
            finally:
                self._lifecycle_lock.release()

        cleared = await asyncio.to_thread(_wait_and_clear)
        if cleared:
            self._emit_status("stopped")
        else:
            # Killed, but the in-flight request didn't unwind within the
            # timeout. The (dead) client ref is left for self-healing.
            self._logger.warning("Tagger killed but in-flight request did not unwind within timeout; leaving stale client ref.")
        return True

    async def get_tag_job_status_async(self, dataset_name: str) -> TagJobInfo:
        """Return a snapshot of the tagging job status for *dataset_name*."""
        async with self._tag_jobs_lock:
            state = self._tag_jobs.get(dataset_name)
            if state is None:
                return TagJobInfo(status="idle", dataset_name=dataset_name)
            return state.snapshot()

    async def list_tag_jobs_async(self) -> list[TagJobInfo]:
        """Return a snapshot of every tracked tagging job (active or finished)."""
        async with self._tag_jobs_lock:
            states = list(self._tag_jobs.values())
        return [s.snapshot() for s in states]

    def is_tagging(self, dataset_name: str) -> bool:
        """Return ``True`` if a tagging job is actively running for *dataset_name*."""
        state = self._tag_jobs.get(dataset_name)
        return state is not None and state.task is not None and not state.task.done()

    def job_id_for_dataset(self, dataset_name: str) -> str | None:
        """Return the job_id of the running job for *dataset_name*, or ``None``."""
        state = self._tag_jobs.get(dataset_name)
        if state is None or state.task is None or state.task.done():
            return None
        return state.job_id

    async def save_tags(self, dataset_name: str, image_info: ImageInfo, result: TaggerResult, save_options: TagSaveOptions, *, source: str = "tagger") -> None:
        """Persist a tag *result* for an image per *save_options*.

        Shared by the batch job (auto-save per image) and the interactive
        save endpoint (user-pruned tags). All writes go through
        :class:`DatasetService` so the file watcher's expected-change
        suppression and history-saving are uniform.
        """
        if save_options.mode == "draft":
            text = format_draft(save_options.draft_format, result)
            self._dataset_service.write_draft(dataset_name, image_info.id, save_options.draft_name, text, source=source)
        elif save_options.mode == "extras":
            self._merge_extras_tags(dataset_name, image_info.id, extras_tags(result), source=source)
        # mode == "none" → write nothing.

    def _merge_extras_tags(self, dataset_name: str, image_id: int, tags: dict[str, Any], *, source: str = SELF_JOB_ID) -> bool:
        """Write the tagger's categorized output as a ``[tags]`` sub-table on the image's TOML extras.

        Overwrites the existing ``[tags]`` sub-table (authoritative
        tagger output) while leaving other top-level keys untouched.
        Goes through :meth:`DatasetService.update_extras` so the
        watcher's expected-change suppression and history-saving are
        uniform. Returns ``False`` when the image is missing.
        """
        if self._dataset_service.get_image(dataset_name, image_id) is None:
            return False
        # Load the existing extras as a tomlkit doc (plain=False) so
        # comments and formatting on other keys survive the round-trip.
        # None means the image has no sidecar (or it's unparseable) —
        # start from an empty doc so the write path repairs it.
        extras = self._dataset_service.load_extras(dataset_name, image_id) or tomlkit.document()
        extras["tags"] = tags
        return self._dataset_service.update_extras(dataset_name, image_id, tomlkit.dumps(extras), source=source)

    def _has_tags_table(self, dataset_name: str, image_id: int) -> bool:
        """Whether the image's TOML extras already carries a ``[tags]`` sub-table.

        Drives the batch ``overwrite=False`` skip for the extras path.
        Returns ``False`` when the image, the sidecar, or the ``[tags]``
        key is missing, or when the sidecar is unparseable.
        """
        extras = self._dataset_service.load_extras(dataset_name, image_id)
        return extras is not None and "tags" in extras

    def preview_save_tags(self, result: TaggerResult, save_options: TagSaveOptions) -> str:
        """Format what :meth:`save_tags` would write, without touching disk.

        ``draft`` → the formatter text; ``extras`` → the ``[tags]`` sub-table
        as TOML (only the tagger's contribution, not the merged sidecar);
        ``none`` → empty string. Used by the interactive save bar's preview.
        """
        if save_options.mode == "draft":
            return format_draft(save_options.draft_format, result)
        if save_options.mode == "extras":
            doc = tomlkit.document()
            doc["tags"] = extras_tags(result)
            return tomlkit.dumps(doc).strip()
        return ""

    def _image_already_has_tags(self, dataset_name: str, image_info: ImageInfo, save_options: TagSaveOptions) -> bool:
        """Whether the image already carries the target save artifact.

        Drives the batch ``overwrite=False`` skip. Draft mode checks the
        DB-backed ``draft_names`` list (cheap); extras mode reads the
        ``[tags]`` sub-table from the TOML sidecar via
        :meth:`_has_tags_table`. ``none`` mode never skips (nothing is
        written).
        """
        if save_options.mode == "draft":
            return save_options.draft_name in image_info.draft_names
        if save_options.mode == "extras":
            return self._has_tags_table(dataset_name, image_info.id)
        return False

    def _filter_skipped_images(self, dataset_name: str, images: list[ImageInfo], save_options: TagSaveOptions) -> tuple[list[ImageInfo], int]:
        """Preflight partition into ``(to_do, skipped)``.

        Mirrors captioning's preflight filter: when ``overwrite=False``
        and the save mode is real (not ``none``), images already
        carrying the target artifact are dropped here rather than
        skipped inside the loop. This lets the start path return
        ``done`` immediately when nothing remains (no background task,
        no SSE event to race the HTTP response) and seeds accurate
        ``total``/``processed`` accounting. Returns the full list with
        ``skipped=0`` when skipping doesn't apply.
        """
        if save_options.mode == "none" or save_options.overwrite:
            return images, 0
        to_do = [img for img in images if not self._image_already_has_tags(dataset_name, img, save_options)]
        return to_do, len(images) - len(to_do)

    async def _run_tag_job(self, dataset_name: str, state: "_TagJobState", images: list[ImageInfo]) -> None:
        """Background task: iterate images, tag each, optionally save, emit status."""
        state.started_at = time.monotonic()
        await self._emit_tag_status(dataset_name, state)
        self._logger.info(
            "Tagger job starting. [dataset=%s, job_id=%s, images=%d, save_mode=%s]",
            dataset_name,
            state.job_id,
            len(images),
            state.save.mode,
        )

        try:
            for image_info in images:
                if state.stop_event.is_set():
                    break

                self._logger.debug(
                    "Tagger processing image. [dataset=%s, job_id=%s, image_id=%d, file_name=%s]",
                    dataset_name,
                    state.job_id,
                    image_info.id,
                    image_info.file_name,
                )
                image_start = time.monotonic()
                try:
                    result = await self.tag_image(
                        dataset_name,
                        image_info,
                        thresholds=state.thresholds,
                        replace_underscores=state.replace_underscores,
                        source=state.source,
                        job_id=state.job_id,
                        policy=state.policy,
                    )

                    if state.save.mode != "none":
                        try:
                            await self.save_tags(dataset_name, image_info, result, state.save, source=f"tagger:{state.job_id}")
                        except Exception:
                            # Saving is best-effort per image — a failed write
                            # shouldn't abort the run.
                            self._logger.exception("Failed to save tags for image. [dataset=%s, image_id=%s]", dataset_name, image_info.id)
                except Exception as exc:
                    self._logger.debug(
                        "Tagger image failed. [dataset=%s, job_id=%s, image_id=%d, error=%s]",
                        dataset_name,
                        state.job_id,
                        image_info.id,
                        exc,
                    )
                    async with state.lock:
                        state.errors += 1
                        state.error_messages.append(f"image {image_info.id}: tag failed: {exc}")
                    continue
                self._logger.debug(
                    "Tagger image done. [dataset=%s, job_id=%s, image_id=%d, tags=%d, saved=%s, duration_ms=%d]",
                    dataset_name,
                    state.job_id,
                    image_info.id,
                    len(result.tags),
                    state.save.mode != "none",
                    int((time.monotonic() - image_start) * 1000),
                )
                async with state.lock:
                    state.processed += 1
                await self._emit_tag_status(dataset_name, state)

            async with state.lock:
                if state.stop_event.is_set():
                    state.status = "cancelled"
                elif state.errors > 0 and state.processed == 0:
                    state.status = "error"
                else:
                    state.status = "done"
        except asyncio.CancelledError:
            async with state.lock:
                state.status = "cancelled"
            raise
        except Exception as exc:
            async with state.lock:
                state.status = "error"
                state.error = str(exc)
                state.error_messages.append(str(exc))
        finally:
            await self._dataset_jobs.release(state.claim)
            self._logger.info(
                "Tagger job finished. [dataset=%s, job_id=%s, status=%s, processed=%d, errors=%d, elapsed=%.1fs]",
                dataset_name,
                state.job_id,
                state.status,
                state.processed,
                state.errors,
                time.monotonic() - state.started_at,
            )
            if state.errors > 0:
                self._logger.warning(
                    "Tagger job had %d error(s). [dataset=%s, job_id=%s, messages=%s]",
                    state.errors,
                    dataset_name,
                    state.job_id,
                    "; ".join(state.error_messages),
                )
            self._logger.info(
                "Tagger result cache at end of job. [dataset=%s, job_id=%s, entries=%d, size=%s/%s]",
                dataset_name,
                state.job_id,
                len(self._tag_results),
                size_units(self._tag_results.total_bytes),
                size_units(self._tag_results.max_bytes),
            )
            # Defer clearing the expected-changes tag: residual inotify
            # events from the last writes land after the loop exits, and
            # the watcher would re-tag them with the per-file ``source``
            # ("tagger") instead of the job_id — leaking through as a
            # toast. ``clear_expected_changes_for_job`` only fires if the
            # job_id still matches, so a newer job's tag is safe.
            clear_task = asyncio.create_task(self._deferred_clear_expected_changes(dataset_name, state.job_id))
            self._pending_clears.add(clear_task)
            clear_task.add_done_callback(self._pending_clears.discard)
            await self._emit_tag_status(dataset_name, state)

    async def _deferred_clear_expected_changes(self, dataset_name: str, job_id: str) -> None:
        """Clear the job's expected-changes tag after a grace window.

        Residual inotify events from the last batch writes land slightly
        after the loop exits (kernel buffering + the watcher debounce).
        Sleeping before the clear keeps ``_expected_sources`` populated
        with the real job_id long enough for those events to be captured
        and suppressed by the originating tab. Mirrors captioning's
        ``_cleanup_async``.
        """
        try:
            await asyncio.sleep(self._configuration.tagger_expected_changes_grace_seconds)
            self._dataset_watcher.clear_expected_changes_for_job(dataset_name, job_id)
        finally:
            task = asyncio.current_task()
            if task is not None:
                self._pending_clears.discard(task)

    async def _emit_tag_status(self, dataset_name: str, state: "_TagJobState") -> None:
        snap = state.snapshot()
        self._event_dispatcher.dispatch(
            TagJobStatusEvent(
                status=snap.status,
                dataset_name=dataset_name,
                processed=snap.processed,
                total=snap.total,
                errors=snap.errors,
                job_id=snap.job_id,
                error=snap.error,
                source=snap.source,
                elapsed=snap.elapsed,
            )
        )

    def _thresholds_from_options(self, options: TagJobOptions) -> TaggingThresholds:
        """Resolve per-request threshold overrides against the server config."""
        cfg = self._configuration
        return TaggingThresholds(
            rating=options.rating_threshold if options.rating_threshold is not None else cfg.tagger_rating_threshold,
            general=options.general_threshold if options.general_threshold is not None else cfg.tagger_general_threshold,
            character=options.character_threshold if options.character_threshold is not None else cfg.tagger_character_threshold,
        )

    def _resolve_job_images(self, dataset_name: str, image_ids: list[int] | None) -> list[ImageInfo]:
        """Resolve the image set for a job: explicit ids, or the whole dataset.

        ``image_ids`` omitted/empty → paginate every image in the dataset
        (newest first, matching the browse order). Explicit ids resolve
        one ``get_image`` each; unknown ids are silently dropped.
        """
        if image_ids:
            resolved: list[ImageInfo] = []
            for image_id in image_ids:
                info = self._dataset_service.get_image(dataset_name, image_id)
                if info is not None:
                    resolved.append(info)
            return resolved

        all_images: list[ImageInfo] = []
        next_token: str | None = None
        while True:
            page = self._dataset_service.list_images(dataset_name, limit=500, next=next_token)
            all_images.extend(page.images)
            next_token = page.next_token
            if not next_token:
                break
        return all_images

    def _resolve_policy(self, dataset_name: str, *, override: TagPolicy | None = None) -> TagPolicy:
        """Return the effective always-add / banned policy for ``dataset_name``.

        Resolution order:

        1. ``override`` if supplied — explicit caller intent, used by
           service-level unit tests and any future preview path that
           wants to try a policy without persisting.
        2. The per-dataset stored policy via
           :meth:`TagPolicyService.get` (production). An unregistered
           dataset returns an empty :class:`TagPolicy` from the
           service, so a missing policy defaults cleanly without a
           404 leaking into the tag call.
        3. Empty :class:`TagPolicy` — the fallback for unit tests
           that construct the service without injecting
           ``TagPolicyService``.
        """
        if override is not None:
            return override
        if self._tag_policy_service is not None:
            return self._tag_policy_service.get(dataset_name)
        return TagPolicy()


@dataclass
class _TagJobState:
    """Mutable state for one batch tagging job (guarded by ``lock`` for counts)."""

    job_id: str
    dataset_name: str
    source: str
    total: int
    thresholds: TaggingThresholds
    replace_underscores: bool | None
    save: TagSaveOptions
    policy: TagPolicy
    claim: JobClaim
    status: TagJobStatus = "running"
    processed: int = 0
    errors: int = 0
    error: str | None = None
    error_messages: list[str] = field(default_factory=list)
    started_at: float | None = None
    stop_event: asyncio.Event = field(default_factory=asyncio.Event)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    task: asyncio.Task[Any] | None = None

    def snapshot(self) -> TagJobInfo:
        elapsed = 0.0
        if self.started_at is not None:
            elapsed = time.monotonic() - self.started_at
        return TagJobInfo(
            status=self.status,
            dataset_name=self.dataset_name,
            job_id=self.job_id,
            processed=self.processed,
            total=self.total,
            errors=self.errors,
            error=self.error,
            error_messages=list(self.error_messages),
            source=self.source,
            elapsed=elapsed,
        )
