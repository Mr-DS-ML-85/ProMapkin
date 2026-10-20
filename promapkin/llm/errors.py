"""LLM error hierarchy. Callers fall back to deterministic analysis on any LLMError."""


class LLMError(RuntimeError):
    pass


class LLMUnavailable(LLMError):
    """Missing SDK, credentials or connectivity: no point retrying this invocation."""


class LLMRefusal(LLMError):
    pass


class LLMTruncated(LLMError):
    pass


class LLMBudgetExceeded(LLMError):
    pass


class LLMInvalidOutput(LLMError):
    """A structured call returned something that is not the requested JSON. `kind` names the failure
    (empty, truncated, wrapped, malformed, not_object, schema); `raw` keeps the reply for the local debug log."""

    def __init__(self, kind: str, message: str, raw: str = "", finish_reason: str = ""):
        super().__init__(message)
        self.kind = kind
        self.raw = raw
        self.finish_reason = finish_reason
