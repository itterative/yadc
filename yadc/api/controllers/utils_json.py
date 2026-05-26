"""Shared JSON utilities for the yadc API controllers."""

import dataclasses
import json
from typing import Any, override

from flask import Response

from .models_errors import APIErrorDetail, APIErrorResponse


class DataclassJSONEncoder(json.JSONEncoder):
    """JSON encoder that serializes dataclass instances as dicts."""

    @override
    def default(self, o: Any):
        if dataclasses.is_dataclass(o) and not isinstance(o, type):
            return dataclasses.asdict(o)  # type: ignore[call-overload]
        return super().default(o)


def jsonify_dataclass(obj: Any) -> Response:
    """JSON-serialize a dataclass or list of dataclasses as a Flask Response."""
    return Response(json.dumps(obj, cls=DataclassJSONEncoder), mimetype="application/json")


def jsonify_error(message: str, details: list[APIErrorDetail] | None = None, status: int = 400) -> tuple[Response, int]:
    """Build a consistent JSON error response with an HTTP status code."""
    return jsonify_dataclass(APIErrorResponse(error=message, details=details)), status
