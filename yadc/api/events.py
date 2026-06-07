from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import ClassVar, Literal


class Event:
    TYPE: ClassVar[str]


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
