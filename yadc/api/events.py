from dataclasses import dataclass
from typing import ClassVar, Literal


class Event:
    TYPE: ClassVar[str]


@dataclass
class PingEvent(Event):
    TYPE: ClassVar[str] = "ping"
    time: str


@dataclass
class CaptioningStatusEvent(Event):
    TYPE: ClassVar[str] = "captioning_status"
    status: Literal["idle", "running", "stopping", "error", "done"]
    dataset_name: str
    processed: int
    total: int
    errors: int
    error: str | None = None


@dataclass
class DatasetChangedEvent(Event):
    TYPE: ClassVar[str] = "dataset_changed"
    dataset_name: str
