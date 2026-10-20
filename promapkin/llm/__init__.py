"""LLM integration (Anthropic and OpenAI, including OpenAI-compatible endpoints)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from promapkin import credentials
from promapkin.llm.clients import DEFAULT_MODELS, AnthropicClient, LLMClient, OpenAIClient, Usage
from promapkin.llm.engine import LLMEngine
from promapkin.llm.errors import LLMError, LLMUnavailable

__all__ = ["LLMSettings", "make_client", "make_engine", "LLMEngine", "LLMError", "LLMUnavailable"]


@dataclass
class LLMSettings:
    provider: str
    model: str = ""
    base_url: str = ""
    api_key: str | None = None
    key_source: str = "none"

    def describe(self) -> dict[str, Any]:
        return {"provider": self.provider, "model": self.model, "base_url": self.base_url or "default",
                "key_source": self.key_source}


def resolve_settings(cfg: Any, provider: str | None = None, model: str | None = None, base_url: str | None = None,
                     api_key: str | None = None) -> LLMSettings:
    provider = (provider or cfg.llm_provider or "anthropic").lower()
    if provider not in ("anthropic", "openai"):
        raise LLMUnavailable(f"unknown LLM provider {provider!r} (use anthropic or openai)")
    if provider == "anthropic":
        model = model or cfg.anthropic_model or DEFAULT_MODELS["anthropic"]
        base = base_url or cfg.anthropic_base_url or credentials.stored_base_url("anthropic") or ""
        key, source = credentials.resolve("anthropic", api_key, cfg.anthropic_api_key_env)
    else:
        model = model or cfg.openai_model or DEFAULT_MODELS["openai"]
        base = base_url or cfg.openai_base_url or credentials.stored_base_url("openai") or ""
        key, source = credentials.resolve("openai", api_key, cfg.openai_api_key_env)
    return LLMSettings(provider, model, base, key, source)


def make_client(cfg: Any, settings: LLMSettings, usage: Usage | None = None) -> LLMClient:
    usage = usage or Usage()
    if settings.provider == "anthropic":
        # No key found: the SDK still resolves ANTHROPIC_AUTH_TOKEN and `ant auth login` profiles itself.
        return AnthropicClient(settings.model, usage, api_key=settings.api_key, base_url=settings.base_url or None,
                               timeout=cfg.llm_timeout, fallbacks=cfg.llm_fallbacks)
    return OpenAIClient(settings.model, usage, api_key=settings.api_key, base_url=settings.base_url or None,
                        timeout=cfg.llm_timeout, reasoning_effort=cfg.openai_reasoning_effort or None)


def make_engine(ws: Any, provider: str | None = None, model: str | None = None, base_url: str | None = None,
                api_key: str | None = None) -> LLMEngine:
    cfg = ws.config
    settings = resolve_settings(cfg, provider, model, base_url, api_key)
    engine = LLMEngine(ws, make_client(cfg, settings), max_calls=cfg.llm_max_calls, concurrency=cfg.llm_concurrency)
    engine.settings = settings  # type: ignore[attr-defined]
    return engine
