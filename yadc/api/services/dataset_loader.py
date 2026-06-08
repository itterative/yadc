"""``DatasetLoader`` — read-only access to a dataset's TOML config and declared paths.

Pure: no DB, no filesystem writes, no side effects beyond a logger
warning on parse failure. The scanner uses it to read the config
before walking image directories; the service uses it to re-register
filesystem watches after a scan finds path changes.

Stateless (no instance state beyond a logger) but kept as a service
so it follows the project's DI conventions and can grow cached /
reactive config loading later without churning callers.
"""

from __future__ import annotations

from logging import Logger
from pathlib import Path
from typing import Any, cast

from yadc.core.config import Config, parse_config
from yadc.utils.dict_utils import load_toml_file

from ..modules.logging_factory import LoggingFactory
from ..modules.service import Service


class DatasetLoader(Service):
    """Read a dataset's TOML config and extract declared image paths.

    All methods are safe to call without DB or watcher dependencies.
    Parse failures are logged at WARN and ``load_config`` returns
    ``None`` rather than raising — callers (scanner, service) treat
    that as "dataset is broken, skip" rather than crashing the
    surrounding workflow.
    """

    def __init__(self, logging: LoggingFactory):
        self._logger: Logger = logging.get_logger(__name__)

    def load_config(self, config_path: Path) -> Config | None:
        """Load and parse a dataset config file with relaxed validation.

        Uses ``strict=False`` so webui TOMLs (missing api_url/model/template)
        parse cleanly. Those fields are validated when creating the captioner.
        Returns ``None`` on any parse failure (logged at WARN).
        """
        try:
            with open(config_path) as f:
                raw = load_toml_file(f)
            return parse_config(raw, strict=False)
        except Exception as e:
            self._logger.warning("Failed to parse config at %s: %s", config_path, e)
            return None

    def load_raw_config(self, config_path: Path) -> dict[str, Any]:
        """Load a TOML config as a raw dict (no v2 validation).

        Used by :meth:`DatasetService.import_dataset` to read the
        original file before resolving relative paths and writing it
        back. We don't go through :meth:`load_config` here because
        the caller is about to mutate the result.
        """
        with open(config_path) as f:
            return load_toml_file(f, plain=False)

    def resolve_relative_paths(self, raw: dict[str, Any], base_dir: Path) -> dict[str, Any]:
        """Resolve relative dataset paths in a raw config dict to absolute paths.

        Handles both v2 ``[[dataset]]`` and v1 ``[dataset]`` formats,
        rewriting each ``path`` (or ``paths``) entry in place.
        """
        # Handle v2 [[dataset]]
        dataset_entries: Any = raw.get("dataset")
        if isinstance(dataset_entries, list):
            dataset_entries = cast(list[dict[str, Any]], dataset_entries)
            for entry in dataset_entries:
                if isinstance(entry, dict) and "path" in entry:
                    p = Path(cast(str, entry["path"]))
                    if not p.is_absolute():
                        entry["path"] = str((base_dir / p).resolve())

        # Handle v1 [dataset] paths
        dataset_v1: Any = raw.get("dataset")
        if isinstance(dataset_v1, dict) and "paths" in dataset_v1:
            dataset_v1 = cast(dict[str, Any], dataset_v1)
            resolved: list[str] = []
            for p in cast(list[str], dataset_v1["paths"]):
                pp = Path(p)
                if not pp.is_absolute():
                    resolved.append(str((base_dir / pp).resolve()))
                else:
                    resolved.append(p)
            dataset_v1["paths"] = resolved

        return raw

    def get_dataset_paths(self, config: Config | None, config_path: Path | None = None) -> list[str]:
        """Extract image directory paths from a parsed config.

        Relative paths are resolved against ``config_path.parent`` if
        provided. Paths that don't resolve to an existing directory
        are dropped (the dataset author may have left a stale entry).
        """
        if config is None:
            return []
        paths: list[str] = []
        for entry in config.dataset:
            if entry.path:
                p = Path(entry.path)
                if not p.is_absolute() and config_path is not None:
                    p = config_path.parent / p
                p = p.resolve()
                if p.is_dir():
                    paths.append(str(p))
        return paths
