"""Shared JSON utilities for the yadc API controllers."""

import dataclasses
import json
from enum import StrEnum
from typing import Any, override

from quart import Response

from .models_errors import APIErrorDetail, APIErrorResponse


class ErrorCode(StrEnum):
    """Stable top-level error codes returned by the API."""

    VALIDATION_ERROR = "VALIDATION_ERROR"
    BAD_REQUEST = "BAD_REQUEST"
    PASSWORD_REQUIRED = "PASSWORD_REQUIRED"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    NOT_FOUND = "NOT_FOUND"
    CONFLICT = "CONFLICT"
    UPSTREAM_ERROR = "UPSTREAM_ERROR"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class DataclassJSONEncoder(json.JSONEncoder):
    """JSON encoder that serializes dataclass instances as dicts."""

    @override
    def default(self, o: Any):
        if dataclasses.is_dataclass(o) and not isinstance(o, type):
            return dataclasses.asdict(o)  # type: ignore[call-overload]
        return super().default(o)


def jsonify_dataclass(obj: Any) -> Response:
    """JSON-serialize a dataclass or list of dataclasses as a Quart Response."""
    return Response(json.dumps(obj, cls=DataclassJSONEncoder), mimetype="application/json")


def jsonify_error(message: str, details: list[APIErrorDetail] | None = None, status: int = 400, code: str | None = None) -> tuple[Response, int]:
    """Build a consistent JSON error response with an HTTP status code."""
    return jsonify_dataclass(APIErrorResponse(error=message, details=details, code=code)), status
