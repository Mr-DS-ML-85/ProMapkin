"""Provider clients built on the official SDKs (`anthropic`, `openai`).

Both expose the same small surface to the rest of Promapkin:

* `json(system, user, schema, ...)` - one structured-output call, returns a dict;
* `text(system, user, ...)`         - one free-text call;
* `chat(system, tools)`             - a multi-turn conversation with tool calling.

Provider-specific message formats never leave this module.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from promapkin.llm.errors import LLMError, LLMInvalidOutput, LLMRefusal, LLMTruncated, LLMUnavailable

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


STRUCTURED_MODES = ("json_schema", "json_object", "prompt")
_FENCE = re.compile(r"\A```(?:json)?[ \t]*\n(.*)\n```\Z", re.S)


def schema_instructions(schema: dict[str, Any]) -> str:
    """For endpoints that do not enforce response_format: the schema goes into the system prompt."""
    return ("Reply with a single JSON object and nothing else: no prose, no markdown. It must match this JSON "
            "schema exactly (every property present, no extra properties):\n" +
            json.dumps(strict_schema(schema), separators=(",", ":")))


def parse_structured(content: str | None, schema: dict[str, Any], *, finish_reason: str = "",
                     allow_fence: bool = False) -> dict[str, Any]:
    """Parse and validate a structured reply. Never repairs JSON: a reply is either exactly a JSON object
    matching the schema (optionally inside one ```json block when allow_fence) or an LLMInvalidOutput that
    says what went wrong."""
    raw = content or ""
    body = raw.strip()
    where = f"finish_reason={finish_reason or 'unknown'}, {len(raw)} chars"
    if not body:
        raise LLMInvalidOutput("empty", f"model returned empty content ({where})", raw, finish_reason)
    if allow_fence:
        m = _FENCE.match(body)
        if m:
            body = m.group(1).strip()
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        leading = not body.startswith(("{", "[")) and "{" in body
        if leading or exc.msg.startswith("Extra data"):
            raise LLMInvalidOutput(
                "wrapped", f"model wrapped the JSON in other text ({where}; reply starts {body[:12]!r}); the "
                "endpoint is probably ignoring response_format - try openai_structured_output: \"prompt\"",
                raw, finish_reason) from exc
        if exc.pos >= len(body) - 1 or "Unterminated" in exc.msg:
            raise LLMInvalidOutput("truncated", f"JSON ends before it is complete ({where}); the endpoint may "
                                   "cut long replies without reporting it", raw, finish_reason) from exc
        raise LLMInvalidOutput("malformed", f"invalid JSON at line {exc.lineno} col {exc.colno}: {exc.msg} "
                               f"({where})", raw, finish_reason) from exc
    if not isinstance(data, dict):
        raise LLMInvalidOutput("not_object", f"expected a JSON object, got {type(data).__name__} ({where})",
                               raw, finish_reason)
    problems = schema_errors(data, schema)
    if problems:
        raise LLMInvalidOutput("schema", f"reply does not match the schema: {'; '.join(problems[:3])} ({where})",
                               raw, finish_reason)
    return data


_TYPES = {"object": dict, "array": list, "string": str, "boolean": bool, "integer": int, "number": (int, float)}


def schema_errors(value: Any, schema: dict[str, Any], path: str = "$") -> list[str]:
    """The subset of JSON Schema Promapkin uses: type, properties (all required), items, enum."""
    kind = schema.get("type") or ("object" if "properties" in schema else None)
    if kind:
        ok = isinstance(value, _TYPES[kind]) and not (kind in ("integer", "number") and isinstance(value, bool))
        if not ok:
            return [f"{path} should be {kind}, got {type(value).__name__}"]
    if "enum" in schema and value not in schema["enum"]:
        return [f"{path} = {str(value)[:40]!r} is not one of {schema['enum']}"]
    out: list[str] = []
    if isinstance(value, dict):
        props = schema.get("properties", {})
        out += [f"{path}.{k} is missing" for k in props if k not in value]
        for k, sub in props.items():
            if k in value:
                out += schema_errors(value[k], sub, f"{path}.{k}")
    elif isinstance(value, list) and "items" in schema:
        for i, v in enumerate(value):
            out += schema_errors(v, schema["items"], f"{path}[{i}]")
            if len(out) >= 5:
                break
    return out


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
        return parse_structured(self._text_of(resp), schema, finish_reason=resp.stop_reason or "")

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
                 timeout: float = 300.0, max_retries: int = 3, reasoning_effort: str | None = None,
                 structured_output: str = "json_schema"):
        super().__init__(model or DEFAULT_MODELS["openai"], usage)
        if structured_output not in STRUCTURED_MODES:
            raise LLMUnavailable(f"openai_structured_output must be one of {STRUCTURED_MODES}")
        # json_schema: strict server-side schema; json_object: JSON mode + schema in the prompt; prompt: schema
        # in the prompt only, for endpoints that ignore response_format (web-session proxies).
        self.structured_output = structured_output
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
        mode = self.structured_output
        extra: dict[str, Any] = {}
        if mode == "json_schema":
            extra["response_format"] = {"type": "json_schema", "json_schema": {
                "name": name, "strict": True, "schema": strict_schema(schema)}}
        else:
            system = f"{system}\n\n{schema_instructions(schema)}"
            if mode == "json_object":
                extra["response_format"] = {"type": "json_object"}
        choice = self._create(max_tokens=max_tokens, messages=[{"role": "system", "content": system},
                                                               {"role": "user", "content": user}], **extra)
        if choice.finish_reason == "length":
            raise LLMTruncated("structured output was cut off by the token limit")
        return parse_structured(choice.message.content, schema, finish_reason=choice.finish_reason or "",
                                allow_fence=mode == "prompt")

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
