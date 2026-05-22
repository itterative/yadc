"""Shared JSON utilities for the yadc API."""

import dataclasses
import json
from typing import Any, override

from flask import Response


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
