"""Error response models for the API."""

import dataclasses

from pydantic_core import ErrorDetails


@dataclasses.dataclass
class APIErrorDetail:
    """A single validation or application error detail."""

    msg: str
    loc: list[str] | None = None
    type: str | None = None

    @classmethod
    def from_pydantic_error(cls, error: ErrorDetails) -> "APIErrorDetail":
        raw_loc = error.get("loc", ())
        loc = [str(x) for x in raw_loc] if raw_loc else []
        return cls(
            msg=error.get("msg", "Unknown error"),
            loc=loc,
            type=error.get("type"),
        )


@dataclasses.dataclass
class APIErrorResponse:
    """Standard error response envelope returned by the API."""

    error: str
    details: list[APIErrorDetail] | None = None
    code: str | None = None
