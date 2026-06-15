"""Shared JSON utilities for the yadc API controllers."""

import dataclasses
import json
from enum import StrEnum
from typing import Any, override

import pydantic
from quart import Response
from werkzeug.exceptions import HTTPException

from .models_errors import APIErrorDetail, APIErrorResponse


class ErrorCode(StrEnum):
    """Stable top-level error codes returned by the API."""

    VALIDATION_ERROR = "VALIDATION_ERROR"
    BAD_REQUEST = "BAD_REQUEST"
    PASSWORD_REQUIRED = "PASSWORD_REQUIRED"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    NOT_FOUND = "NOT_FOUND"
    CONFLICT = "CONFLICT"
    PAYLOAD_TOO_LARGE = "PAYLOAD_TOO_LARGE"
    UPSTREAM_ERROR = "UPSTREAM_ERROR"
    GATEWAY_TIMEOUT = "GATEWAY_TIMEOUT"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class DataclassJSONEncoder(json.JSONEncoder):
    """JSON encoder that serializes dataclass instances as dicts."""

    @override
    def default(self, o: Any):
        if dataclasses.is_dataclass(o) and not isinstance(o, type):
            return dataclasses.asdict(o)  # type: ignore[call-overload]
        return super().default(o)


def validate_body[T: pydantic.BaseModel](model: type[T], raw_body: Any) -> T:
    """Validate *raw_body* against *model*.

    Returns the validated *model* instance on success. Raises
    :class:`HTTPException` (with a 400 JSON body) on validation failure.
    """
    try:
        return model.model_validate(raw_body)
    except pydantic.ValidationError as e:
        raise HTTPException(
            response=jsonify_error(
                "Invalid request body",
                details=[APIErrorDetail.from_pydantic_error(err) for err in e.errors()],
                status=400,
                code=ErrorCode.BAD_REQUEST,
            )
        )


def jsonify_dataclass(obj: Any, status: int | None = 200) -> Response:
    """JSON-serialize a dataclass or list of dataclasses as a Quart Response."""
    return Response(json.dumps(obj, cls=DataclassJSONEncoder), status=status, mimetype="application/json")


def jsonify_error(message: str, details: list[APIErrorDetail] | None = None, status: int = 400, code: str | None = None) -> Response:
    """Build a consistent JSON error response with an HTTP status code."""
    return jsonify_dataclass(APIErrorResponse(error=message, details=details, code=code), status=status)
