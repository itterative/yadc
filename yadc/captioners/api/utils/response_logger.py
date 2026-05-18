import json
import re
from pathlib import Path
from typing import Any, Mapping

import requests

# Accept both dict and CaseInsensitiveDict
_ResponseHeaders = Mapping[str, str]


class ResponseLogger:
    """Logs API request/response pairs to JSONL files for debugging.

    Accepts a resolved ``log_dir`` and writes one JSONL file per request/response
    pair. The counter is initialized from existing files in the directory.
    """

    _SENSITIVE_HEADERS: frozenset[str] = frozenset({"authorization", "x-goog-api-key"})

    def __init__(self, log_dir: Path, *, log_body: bool = False):
        self._log_dir = log_dir
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._counter = self._resume_counter(log_dir)
        self._log_body = log_body

    @property
    def log_dir(self) -> Path:
        """The run directory for debug log files."""
        return self._log_dir

    # ---- counter ----

    @staticmethod
    def _resume_counter(run_dir: Path) -> int:
        """Find the next counter value by scanning existing files."""
        if not run_dir.exists():
            return 1

        max_n = 0
        for f in run_dir.iterdir():
            if f.suffix == ".jsonl":
                match = re.match(r"(\d+)", f.name)
                if match:
                    max_n = max(max_n, int(match.group(1)))

        return max_n + 1 if max_n else 1

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
        image_name: str,
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

        n = self._counter
        self._counter += 1
        filepath = self._log_dir / f"{n:06d}_{image_name}.jsonl"

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
