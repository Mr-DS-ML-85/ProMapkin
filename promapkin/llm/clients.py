"""Provider clients built on the official SDKs (`anthropic`, `openai`).

Both expose the same small surface to the rest of Promapkin:

* `json(system, user, schema, ...)` - one structured-output call, returns a dict;
* `text(system, user, ...)`         - one free-text call;
* `chat(system, tools)`             - a multi-turn conversation with tool calling.

Provider-specific message formats never leave this module.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from promapkin.llm.errors import LLMError, LLMRefusal, LLMTruncated, LLMUnavailable

DEFAULT_MODELS = {"anthropic": "claude-opus-5-5", "openai": "gpt-5"}


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_calls: int = 0

    def add(self, i: int, o: int) -> None:
        self.calls += 1
        self.input_tokens += i or 0
        self.output_tokens += o or 0


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Turn:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = ""


def strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Make a JSON schema acceptable to strict structured outputs on both providers:
    every object closed (additionalProperties false) with all properties required, and
    numeric/string/array bound keywords removed."""
    s = json.loads(json.dumps(schema))

    def fix(node: Any) -> None:
        if isinstance(node, dict):
            for k in ("minimum", "maximum", "minLength", "maxLength", "minItems", "maxItems", "multipleOf",
                      "exclusiveMinimum", "exclusiveMaximum", "pattern", "format"):
                node.pop(k, None)
            if node.get("type") == "object" or "properties" in node:
                node["type"] = "object"
                node.setdefault("properties", {})
                node["required"] = list(node["properties"])
                node["additionalProperties"] = False
            for v in list(node.values()):
                fix(v)
        elif isinstance(node, list):
            for v in node:
                fix(v)

    fix(s)
    return s


class LLMClient:
    provider = ""

    def __init__(self, model: str, usage: Usage):
        self.model = model
        self.usage = usage

    def json(self, system: str, user: str, schema: dict[str, Any], *, max_tokens: int = 16000,
             effort: str = "medium", name: str = "result") -> dict[str, Any]:
        raise NotImplementedError

    def text(self, system: str, user: str, *, max_tokens: int = 16000, effort: str = "medium") -> str:
        raise NotImplementedError

    def chat(self, system: str, tools: list[dict[str, Any]]) -> "Chat":
        raise NotImplementedError


class Chat:
    def user(self, text: str) -> None:
        raise NotImplementedError

    def send(self, *, max_tokens: int = 16000, effort: str = "high") -> Turn:
        raise NotImplementedError

    def tool_results(self, results: list[tuple[ToolCall, str, bool]]) -> None:
        raise NotImplementedError


# ===================================================================================== Anthropic

class AnthropicClient(LLMClient):
    provider = "anthropic"

    def __init__(self, model: str, usage: Usage, api_key: str | None = None, base_url: str | None = None,
                 timeout: float = 300.0, max_retries: int = 3, fallbacks: bool = True):
        super().__init__(model or DEFAULT_MODELS["anthropic"], usage)
        try:
            import anthropic
        except ModuleNotFoundError as exc:
            raise LLMUnavailable("the Anthropic SDK is not installed: pip install 'promapkin[llm]'") from exc
        self._sdk = anthropic
        kwargs: dict[str, Any] = {"timeout": timeout, "max_retries": max_retries}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        try:
            # With no explicit key the SDK resolves ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN / `ant auth` profiles.
            self.client = anthropic.Anthropic(**kwargs)
        except Exception as exc:  # noqa: BLE001 - surface any credential problem uniformly
            raise LLMUnavailable(f"could not create Anthropic client: {exc}") from exc
        # Server-side refusal fallback is a first-party API feature; skip it behind custom gateways.
        self.fallbacks = fallbacks and not base_url

    def _extra(self) -> dict[str, Any]:
        if not self.fallbacks:
            return {}
        return {"extra_headers": {"anthropic-beta": "server-side-fallback-2026-07-01"},
                "extra_body": {"fallbacks": "default"}}

    def _create(self, **kwargs: Any) -> Any:
        sdk = self._sdk
        try:
            resp = self.client.messages.create(model=self.model, **kwargs, **self._extra())
        except sdk.AuthenticationError as exc:
            raise LLMUnavailable("Anthropic rejected the API key (promapkin auth status)") from exc
        except sdk.NotFoundError as exc:
            raise LLMError(f"unknown model or endpoint: {self.model}") from exc
        except sdk.BadRequestError as exc:
            if self.fallbacks and "fallback" in str(exc).lower():
                self.fallbacks = False   # gateway without the beta: retry once without it
                return self._create(**kwargs)
            raise LLMError(f"Anthropic request rejected: {exc}") from exc
        except sdk.RateLimitError as exc:
            raise LLMError("Anthropic rate limit exceeded after retries") from exc
        except sdk.APIStatusError as exc:
            raise LLMError(f"Anthropic API error {exc.status_code}") from exc
        except sdk.APIConnectionError as exc:
            raise LLMUnavailable(f"cannot reach the Anthropic API: {exc}") from exc
        usage = getattr(resp, "usage", None)
        self.usage.add(getattr(usage, "input_tokens", 0), getattr(usage, "output_tokens", 0))
        if resp.stop_reason == "refusal":
            details = getattr(resp, "stop_details", None)
            raise LLMRefusal(f"model declined the request ({getattr(details, 'category', None) or 'policy'})")
        return resp

    @staticmethod
    def _text_of(resp: Any) -> str:
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")

    def json(self, system: str, user: str, schema: dict[str, Any], *, max_tokens: int = 16000,
             effort: str = "medium", name: str = "result") -> dict[str, Any]:
        resp = self._create(max_tokens=max_tokens, system=system, messages=[{"role": "user", "content": user}],
                            output_config={"effort": effort,
                                           "format": {"type": "json_schema", "schema": strict_schema(schema)}})
        if resp.stop_reason == "max_tokens":
            raise LLMTruncated("structured output was cut off by max_tokens")
        try:
            return json.loads(self._text_of(resp))
        except json.JSONDecodeError as exc:
            raise LLMError("model returned invalid JSON") from exc

    def text(self, system: str, user: str, *, max_tokens: int = 16000, effort: str = "medium") -> str:
        resp = self._create(max_tokens=max_tokens, system=system, messages=[{"role": "user", "content": user}],
                            output_config={"effort": effort})
        return self._text_of(resp)

    def chat(self, system: str, tools: list[dict[str, Any]]) -> "AnthropicChat":
        return AnthropicChat(self, system, tools)


class AnthropicChat(Chat):
    def __init__(self, client: AnthropicClient, system: str, tools: list[dict[str, Any]]):
        self.c = client
        self.system = system
        self.tools = [{"name": t["name"], "description": t["description"], "input_schema": t["input_schema"]}
                      for t in tools]
        self.messages: list[dict[str, Any]] = []

    def user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def send(self, *, max_tokens: int = 16000, effort: str = "high") -> Turn:
        resp = self.c._create(max_tokens=max_tokens, system=self.system, messages=self.messages,
                              tools=self.tools, output_config={"effort": effort})
        # Keep the full content (including thinking blocks) so the conversation stays append-only.
        self.messages.append({"role": "assistant", "content": resp.content})
        calls = [ToolCall(b.id, b.name, dict(b.input)) for b in resp.content if getattr(b, "type", "") == "tool_use"]
        return Turn(AnthropicClient._text_of(resp), calls, resp.stop_reason or "")

    def tool_results(self, results: list[tuple[ToolCall, str, bool]]) -> None:
        # All results for one assistant turn go back in a single user message.
        self.messages.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": call.id, "content": content, **({"is_error": True} if err else {})}
            for call, content, err in results]})


# ===================================================================================== OpenAI

class OpenAIClient(LLMClient):
    provider = "openai"

    def __init__(self, model: str, usage: Usage, api_key: str | None = None, base_url: str | None = None,
                 timeout: float = 300.0, max_retries: int = 3, reasoning_effort: str | None = None):
        super().__init__(model or DEFAULT_MODELS["openai"], usage)
        try:
            import openai
        except ModuleNotFoundError as exc:
            raise LLMUnavailable("the OpenAI SDK is not installed: pip install 'promapkin[llm]'") from exc
        self._sdk = openai
        if not api_key and not base_url:
            raise LLMUnavailable("no OpenAI API key: set OPENAI_API_KEY or run `promapkin auth set openai`")
        # OpenAI-compatible servers (Azure gateways, vLLM, Ollama, LM Studio) only need base_url.
        self.client = openai.OpenAI(api_key=api_key or "not-needed", base_url=base_url or None, timeout=timeout,
                                    max_retries=max_retries)
        self.official = not base_url or "api.openai.com" in base_url
        self.reasoning_effort = reasoning_effort

    def _create(self, **kwargs: Any) -> Any:
        sdk = self._sdk
        max_tokens = kwargs.pop("max_tokens")
        kwargs["max_completion_tokens" if self.official else "max_tokens"] = max_tokens
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        try:
            resp = self.client.chat.completions.create(model=self.model, **kwargs)
        except sdk.AuthenticationError as exc:
            raise LLMUnavailable("OpenAI rejected the API key (promapkin auth status)") from exc
        except sdk.NotFoundError as exc:
            raise LLMError(f"unknown model or endpoint: {self.model}") from exc
        except sdk.BadRequestError as exc:
            raise LLMError(f"OpenAI request rejected: {exc}") from exc
        except sdk.RateLimitError as exc:
            raise LLMError("OpenAI rate limit exceeded after retries") from exc
        except sdk.APIStatusError as exc:
            raise LLMError(f"OpenAI API error {exc.status_code}") from exc
        except sdk.APIConnectionError as exc:
            raise LLMUnavailable(f"cannot reach {self.client.base_url}: {exc}") from exc
        usage = getattr(resp, "usage", None)
        self.usage.add(getattr(usage, "prompt_tokens", 0), getattr(usage, "completion_tokens", 0))
        choice = resp.choices[0]
        if getattr(choice.message, "refusal", None):
            raise LLMRefusal(f"model declined the request: {choice.message.refusal[:200]}")
        return choice

    def json(self, system: str, user: str, schema: dict[str, Any], *, max_tokens: int = 16000,
             effort: str = "medium", name: str = "result") -> dict[str, Any]:
        choice = self._create(max_tokens=max_tokens, messages=[{"role": "system", "content": system},
                                                               {"role": "user", "content": user}],
                              response_format={"type": "json_schema", "json_schema": {
                                  "name": name, "strict": True, "schema": strict_schema(schema)}})
        if choice.finish_reason == "length":
            raise LLMTruncated("structured output was cut off by the token limit")
        try:
            return json.loads(choice.message.content or "")
        except json.JSONDecodeError as exc:
            raise LLMError("model returned invalid JSON") from exc

    def text(self, system: str, user: str, *, max_tokens: int = 16000, effort: str = "medium") -> str:
        choice = self._create(max_tokens=max_tokens, messages=[{"role": "system", "content": system},
                                                               {"role": "user", "content": user}])
        return choice.message.content or ""

    def chat(self, system: str, tools: list[dict[str, Any]]) -> "OpenAIChat":
        return OpenAIChat(self, system, tools)


class OpenAIChat(Chat):
    def __init__(self, client: OpenAIClient, system: str, tools: list[dict[str, Any]]):
        self.c = client
        self.tools = [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                                        "parameters": t["input_schema"]}} for t in tools]
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": system}]

    def user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def send(self, *, max_tokens: int = 16000, effort: str = "high") -> Turn:
        choice = self.c._create(max_tokens=max_tokens, messages=self.messages, tools=self.tools or None)
        msg = choice.message
        calls = []
        raw_calls = []
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_invalid_json": tc.function.arguments}
            calls.append(ToolCall(tc.id, tc.function.name, args))
            raw_calls.append({"id": tc.id, "type": "function",
                              "function": {"name": tc.function.name, "arguments": tc.function.arguments or "{}"}})
        entry: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
        if raw_calls:
            entry["tool_calls"] = raw_calls
        self.messages.append(entry)
        return Turn(msg.content or "", calls, choice.finish_reason or "")

    def tool_results(self, results: list[tuple[ToolCall, str, bool]]) -> None:
        for call, content, err in results:
            self.messages.append({"role": "tool", "tool_call_id": call.id,
                                  "content": ("ERROR: " if err else "") + content})
