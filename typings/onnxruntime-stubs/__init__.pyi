# Minimal stubs for the onnxruntime symbols used by yadc/taggers/onnx.py.
# Hand-written because onnxruntime ships no py.typed marker and its core
# lives in a C extension that pyright/mypy cannot introspect, so the
# auto-generated stubs leave re-exports unresolvable.
from collections.abc import Sequence
from typing import Any

import numpy as np

class SessionOptions:
    intra_op_num_threads: int
    inter_op_num_threads: int

class NodeArg:
    name: str
    # Shape is a list mixing ints with symbolic strings ("batch") and
    # None for unknown ranks — callers handle the fallback themselves.
    shape: Any

class InferenceSession:
    def __init__(
        self,
        path_or_bytes: str | bytes,
        sess_options: SessionOptions | None = ...,
        providers: Sequence[str] | None = ...,
        **kwargs: Any,
    ) -> None: ...
    def get_inputs(self) -> list[NodeArg]: ...
    def get_outputs(self) -> list[NodeArg]: ...
    def run(
        self,
        output_names: list[str],
        input_feed: dict[str, np.ndarray],
    ) -> list[np.ndarray]: ...

def get_available_providers() -> list[str]: ...
