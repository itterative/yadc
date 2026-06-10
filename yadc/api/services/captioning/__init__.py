"""Captioning service package — manages background captioning jobs for datasets.

Re-exports the public API: ``CaptioningService``, ``AsyncCaptionJob``,
``AsyncCaptionJobRunner``, ``RefineOptions``, ``JobStatus``, and ``JobInfo``.
"""

from .job import AsyncCaptionJob
from .job_runner import AsyncCaptionJobRunner
from .models import JobInfo, JobStatus, RefineOptions
from .service import CaptioningService

__all__ = [
    "AsyncCaptionJob",
    "AsyncCaptionJobRunner",
    "CaptioningService",
    "JobInfo",
    "JobStatus",
    "RefineOptions",
]
