import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import requests

from yadc.core.env import DEBUG_CAPTION_REQUESTS_BODY, DEBUG_CAPTION_RESPONSES

# Accept both dict and CaseInsensitiveDict
_ResponseHeaders = Mapping[str, str]


class ResponseLogger:
    """Logs API request/response pairs to JSONL files for debugging.

    Created when ``env.DEBUG_API_RESPONSES`` is ``True``.
    Logging is only active inside a :meth:`Session.capture_response` block.
    """

    _SENSITIVE_HEADERS: frozenset[str] = frozenset({"authorization", "x-goog-api-key"})
    _DATASET_PREFIXES_TO_STRIP: tuple[str, ...] = ("dataset_",)

    _run_dir: Path
    _counter: int

    def __init__(self, run_dir: Path, *, log_body: bool = False):
        self._run_dir = run_dir
        self._log_body = log_body
        self._counter = self._resume_counter()

    @classmethod
    def from_env(cls, cache_path: Path, dataset_paths: list[str] | None = None) -> "ResponseLogger | None":
        """Create a ResponseLogger if debug logging is enabled, otherwise return None."""
        if not DEBUG_CAPTION_RESPONSES:
            return None
        run_dir = cls._build_run_dir(cache_path, dataset_paths)
        return cls(run_dir, log_body=DEBUG_CAPTION_REQUESTS_BODY)

    # ---- run directory naming ----

    @classmethod
    def _build_run_dir(cls, cache_path: Path, dataset_paths: list[str] | None) -> Path:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        name = cls._derive_dataset_name(dataset_paths)
        path_hash = cls._hash_paths(dataset_paths)
        return cache_path / "api-debug" / f"{timestamp}_{name}_{path_hash}"

    @classmethod
    def _derive_dataset_name(cls, dataset_paths: list[str] | None) -> str:
        if not dataset_paths:
            return "unknown"
        raw = Path(dataset_paths[0]).name if dataset_paths[0] else ""
        for prefix in cls._DATASET_PREFIXES_TO_STRIP:
            if raw.startswith(prefix):
                raw = raw[len(prefix) :]
        return raw or "unknown"

    @classmethod
    def _hash_paths(cls, dataset_paths: list[str] | None) -> str:
        payload = "\0".join(dataset_paths) if dataset_paths else ""
        return hashlib.sha256(payload.encode()).hexdigest()[:8]

    # ---- counter ----

    def _resume_counter(self) -> int:
        """Find the next counter value by scanning existing files."""
        if not self._run_dir.exists():
            return 1

        max_n = 0
        for f in self._run_dir.iterdir():
            if f.suffix == ".jsonl":
                match = re.match(r"(\d+)", f.name)
                if match:
                    max_n = max(max_n, int(match.group(1)))

        return max_n + 1 if max_n else 1

    def _next_filename(self) -> str:
        n = self._counter
        self._counter += 1
        return f"{n:03d}.jsonl"

    # ---- header sanitization ----

    def _sanitize_headers(self, headers: dict[str, str]) -> dict[str, str]:
        sanitized: dict[str, str] = {}
        for k, v in headers.items():
            if k.lower() in self._SENSITIVE_HEADERS:
                sanitized[k] = "[REDACTED]"
            else:
                sanitized[k] = v
        return sanitized

    # ---- logging ----

    def log(
        self,
        *,
        method: str,
        url: str,
        request_headers: dict[str, str],
        request_body: dict[str, Any] | list[Any] | str | None,
        response_status: int,
        response_headers: _ResponseHeaders,
        response_body: str,
        stream: bool = False,
    ):
        entry: dict[str, Any] = {
            "request": {
                "method": method,
                "url": url,
                "headers": self._sanitize_headers(request_headers),
            },
            "response": {
                "status": response_status,
                "headers": dict(response_headers) if response_headers is not None else {},
            },
            "stream": stream,
        }

        if self._log_body:
            if isinstance(request_body, (dict, list)):
                entry["request"]["body"] = request_body
            elif request_body is not None:
                entry["request"]["body_raw"] = str(request_body)
        else:
            entry["request"]["body_raw"] = "[REDACTED]"

        entry["response"]["body"] = response_body

        self._run_dir.mkdir(parents=True, exist_ok=True)
        filepath = self._run_dir / self._next_filename()

        tmp_path = filepath.with_suffix(".tmp")
        with open(tmp_path, "w") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        tmp_path.rename(filepath)


class DebugStreamProxy:
    """Wraps a streaming response to accumulate raw lines for debug logging."""

    __slots__: tuple[str, ...] = ("response", "accumulator")

    def __init__(self, response: requests.Response, accumulator: list[bytes | str]):
        self.response = response
        self.accumulator = accumulator

    def __getattr__(self, name: str) -> Any:
        return getattr(self.response, name)

    def iter_lines(self, *args: Any, **kwargs: Any) -> Any:  # returns Generator[bytes | str, None, None]
        for line in self.response.iter_lines(*args, **kwargs):
            self.accumulator.append(line)
            yield line
