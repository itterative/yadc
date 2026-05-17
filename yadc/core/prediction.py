class PredictionContext:
    """
    Mutable container for prediction metadata, passed into predict/predict_stream calls.

    The captioner populates this during prediction so the caller can inspect
    reasoning and other side-band data without querying global state on the
    captioner instance.

    Attributes:
        reasoning (str | None): Raw reasoning/thinking content returned by the API.
        reasoning_summary (str | None): Displayable summary of reasoning (when provided by the API).
        reasoning_encrypted (list[dict] | None): Encrypted reasoning data to pass back in subsequent turns.
    """

    __slots__ = ("reasoning", "reasoning_summary", "reasoning_encrypted")

    def __init__(self):
        self.reasoning: str | None = None
        self.reasoning_summary: str | None = None
        self.reasoning_encrypted: list[dict] | None = None
