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
