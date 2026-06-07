"""Event types and dataclasses dispatched on the :class:`EventDispatcher`.

Each event is a ``@dataclass`` subclass of :class:`Event` with a
``TYPE: ClassVar[str]`` carrying the wire name (e.g. ``"captioning_status"``).

- ``SetupAppEvent`` — fired during ``Application.configure_app()``,
  *before* blueprints are registered on the Quart app. Carries the
  ``Quart`` instance so services can register request/response hooks
  (e.g. CORS attaches ``after_request`` to the ``ApiBlueprint``).
  Fired synchronously, while the blueprints are still unwired — at
  this point there is no running event loop, so handlers should be
  sync.
- ``StartupEvent`` / ``ShutdownEvent`` — application lifecycle, used to
  bring services up/down. ``StartupEvent`` is dispatched from Quart's
  ``before_serving`` callback, after the event loop is running, so
  async handlers can use ``asyncio.create_task()`` directly.
- ``PingEvent`` — periodic keepalive (every 5s); not stored in the SSE
  history ring buffer.
- ``CaptioningStatusEvent`` — per-job progress (status, processed, total,
  errors, elapsed, max_concurrent) for the UI's progress bar.
- ``DatasetChangedEvent`` — emitted by the dataset watcher and on manual
  rescan; ``job_id`` carries the originating source id so the
  originating UI tab can suppress its own reload.
- ``ResumptionFailedEvent`` — returned to a single SSE client when its
  ``Last-Event-ID`` is older than the oldest buffered event.
- ``ImageCaptionedEvent`` / ``ImageRefinedEvent`` /
  ``ImageCaptionStartedEvent`` / ``ImageCaptionErrorEvent`` — per-image
  notifications, consumed by the UI to update the image grid.
- ``EnvironmentsChangedEvent`` / ``TemplatesChangedEvent`` — emitted by
  the env / template watchers so the UI can refresh the env/template
  pickers.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import ClassVar, Literal

from quart import Quart


class Event:
    TYPE: ClassVar[str]


@dataclass
class SetupAppEvent(Event):
    TYPE: ClassVar[str] = "setup_app"
    app: Quart


class StartupEvent(Event):
    TYPE: ClassVar[str] = "startup"


class ShutdownEvent(Event):
    TYPE: ClassVar[str] = "shutdown"


@dataclass
class PingEvent(Event):
    TYPE: ClassVar[str] = "ping"
    time: str


@dataclass
class CaptioningStatusEvent(Event):
    TYPE: ClassVar[str] = "captioning_status"
    status: Literal["idle", "running", "stopping", "error", "done", "cancelled"]
    dataset_name: str
    processed: int
    total: int
    errors: int
    job_id: str = ""
    error: str | None = None
    error_messages: list[str] = dataclasses.field(default_factory=list)
    api_url: str = ""
    api_model_name: str = ""
    # Seconds since the job started. 0 before the job has actually
    # started.  Used by the frontend for the "elapsed" display and as
    # a sanity check on per-image timing under concurrency.
    elapsed: float = 0.0
    # Configured concurrency for the job. The frontend divides the
    # per-image ETA by this so the estimate reflects actual wall-clock
    # throughput (max_concurrent requests in flight).
    max_concurrent: int = 1


@dataclass
class DatasetChangedEvent(Event):
    TYPE: ClassVar[str] = "dataset_changed"
    dataset_name: str
    job_id: str | None = None


@dataclass
class ResumptionFailedEvent(Event):
    TYPE: ClassVar[str] = "resumption_failed"
    requested_event_id: int


@dataclass
class ImageCaptionedEvent(Event):
    TYPE: ClassVar[str] = "image_captioned"
    dataset_name: str
    job_id: str
    id: int
    file_name: str
    path: str
    has_caption: bool = False
    has_toml: bool = False
    width: int = 0
    height: int = 0
    draft_names: list[str] = dataclasses.field(default_factory=list)
    last_modified_t: float | None = None
    caption: str = ""
    duration_ms: int = 0
    api_url: str = ""
    api_model_name: str = ""


@dataclass
class ImageRefinedEvent(Event):
    TYPE: ClassVar[str] = "image_refined"
    dataset_name: str
    job_id: str
    image_id: int
    caption: str = ""
    source: Literal["caption", "draft"] = "caption"
    draft_name: str = ""


@dataclass
class EnvironmentsChangedEvent(Event):
    TYPE: ClassVar[str] = "environments_changed"
    envs: list[str] = dataclasses.field(default_factory=list)


@dataclass
class TemplatesChangedEvent(Event):
    TYPE: ClassVar[str] = "templates_changed"
    templates: list[str] = dataclasses.field(default_factory=list)


@dataclass
class ImageCaptionStartedEvent(Event):
    TYPE: ClassVar[str] = "image_caption_started"
    dataset_name: str
    job_id: str
    image_id: int
    file_name: str


@dataclass
class ImageCaptionErrorEvent(Event):
    TYPE: ClassVar[str] = "image_caption_error"
    dataset_name: str
    job_id: str
    image_id: int
    error: str
    duration_ms: int = 0
    api_url: str = ""
    api_model_name: str = ""
