"""Re-exports all services and repositories auto-discovered by the DI container."""

from .captioning import CaptioningService
from .config_history import ConfigHistoryService
from .config_history_repository import ConfigHistoryRepository
from .dataset_jobs import DatasetBusyError, DatasetJobService, JobClaim, JobKind
from .dataset_loader import DatasetLoader
from .dataset_repository import DatasetRepository
from .dataset_scanner import DatasetScanner
from .dataset_upload import DatasetUploadResult, DatasetUploadService, UploadProgressEvent
from .datasets import DatasetService
from .managed_datasets import ManagedDatasetsService
from .prompt_generation import ExamplePair, PromptGenerationFocus, PromptGenerationRequest, PromptGenerationService
from .prompt_history import (
    PROMPT_HISTORY_MAX_ENTRIES,
    PromptHistoryListItem,
    PromptHistoryPage,
    PromptHistorySaveRequest,
    PromptHistoryService,
)
from .prompt_history_repository import PromptExample, PromptHistoryEntry, PromptHistoryRepository
from .settings import SettingsService
from .settings_repository import SettingsRepository
from .tagging import TaggingService

__all__ = [
    "CaptioningService",
    "TaggingService",
    "ConfigHistoryRepository",
    "ConfigHistoryService",
    "DatasetLoader",
    "DatasetRepository",
    "DatasetScanner",
    "DatasetService",
    "DatasetBusyError",
    "DatasetJobService",
    "JobClaim",
    "JobKind",
    "DatasetUploadResult",
    "DatasetUploadService",
    "ExamplePair",
    "ManagedDatasetsService",
    "PROMPT_HISTORY_MAX_ENTRIES",
    "PromptGenerationFocus",
    "PromptGenerationRequest",
    "PromptGenerationService",
    "PromptExample",
    "PromptHistoryEntry",
    "PromptHistoryListItem",
    "PromptHistoryPage",
    "PromptHistoryRepository",
    "PromptHistorySaveRequest",
    "PromptHistoryService",
    "SettingsRepository",
    "SettingsService",
    "UploadProgressEvent",
]
