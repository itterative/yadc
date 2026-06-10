"""Data classes and type aliases for the captioning subsystem.

- ``RefineOptions`` — parameters for a feedback-driven caption refinement job.
- ``JobStatus`` — literal union of possible job states.
- ``JobInfo`` — snapshot of a running or recently finished job.
"""

from dataclasses import dataclass, field
from typing import Literal

from yadc.core import ReplyRound


@dataclass
class RefineOptions:
    """Parameters for a dry-run refine job (feedback-driven caption refinement).

    Running with refine options active implies a dry run."""

    extra_messages: list[ReplyRound] | None = None
    refine_source: Literal["caption", "draft"] = "caption"
    refine_draft_name: str = ""


type JobStatus = Literal["idle", "running", "stopping", "error", "done", "cancelled"]


@dataclass
class JobInfo:
    """Snapshot of a running (or recently finished) captioning job."""

    status: JobStatus
    dataset_name: str
    job_id: str = ""
    processed: int = 0
    total: int = 0
    errors: int = 0
    error: str | None = None
    error_messages: list[str] = field(default_factory=list)
    api_url: str = ""
    api_model_name: str = ""
    # Seconds since the job started. 0 before the job has actually
    # started.  Returned to the frontend on the job-start / status
    # endpoints for the "elapsed" display.
    elapsed: float = 0.0
    # Configured concurrency for the job.  Returned to the frontend so
    # it can correctly factor the ETA under parallel runs.
    max_concurrent: int = 1
