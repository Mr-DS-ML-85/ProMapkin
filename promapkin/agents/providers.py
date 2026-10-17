"""Agent provider adapters.

The core never depends on a model vendor. A provider receives a `TaskContract`
(JSON) and returns a result matching `RESULT_SCHEMA`; provider-specific
conversation formats stay inside the adapter.

* ``none``: deterministic specialists only (default; nothing leaves the machine).
* ``command``: any local or remote agent wrapped as a command reading the contract
  on stdin and writing the result JSON on stdout.
* ``anthropic``: Claude Messages API (requires ``allow_remote`` and ANTHROPIC_API_KEY).
* ``mcp``: no provider call; tasks are queued for external agents that pull them via
  the MCP tools ``get_analysis_tasks`` / ``submit_analysis``.

Every remote call is logged to .promapkin/egress.log (what was sent, never secrets).
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from promapkin.security import UNTRUSTED_NOTICE


class ProviderError(RuntimeError):
    pass


class Provider:
    name = "none"
    remote = False

    def complete(self, contract: dict[str, Any]) -> dict[str, Any] | None:
        return None


class NoneProvider(Provider):
    pass


class QueueProvider(Provider):
    """Tasks are queued for external MCP agents instead of being executed here."""

    name = "mcp"


class CommandProvider(Provider):
    name = "command"

    def __init__(self, command: str, timeout: int = 300):
        if not command:
            raise ProviderError("provider_command is not configured")
        self.argv = shlex.split(command)
        self.timeout = timeout

    def complete(self, contract: dict[str, Any]) -> dict[str, Any] | None:
        proc = subprocess.run(self.argv, input=json.dumps(contract), capture_output=True, text=True,
                              timeout=self.timeout, check=False)
        if proc.returncode != 0:
            raise ProviderError(f"provider command failed ({proc.returncode}): {proc.stderr.strip()[-500:]}")
        return _parse_json(proc.stdout)


class AnthropicProvider(Provider):
    name = "anthropic"
    remote = True
    endpoint = "https://api.anthropic.com/v1/messages"

    def __init__(self, model: str, egress_log: Path):
        self.key = os.environ.get("ANTHROPIC_API_KEY")
        if not self.key:
            raise ProviderError("ANTHROPIC_API_KEY is not set")
        self.model = model or "claude-sonnet-5-5"
        self.egress_log = egress_log

    def complete(self, contract: dict[str, Any]) -> dict[str, Any] | None:
        system = (
            "You are a Promapkin specialist analysis agent. " + UNTRUSTED_NOTICE + " "
            "Respond with a single JSON object matching the contract's output_schema and nothing else. "
            + contract.get("evidence_requirements", "")
        )
        user = json.dumps({k: contract[k] for k in ("specialist", "objective", "scope", "revision", "facts",
                                                    "output_schema")}, default=str)
        body = {"model": self.model, "max_tokens": min(int(contract.get("budget_tokens", 4000)), 8000),
                "system": system, "messages": [{"role": "user", "content": user}]}
        self._log(contract, len(user))
        req = urllib.request.Request(self.endpoint, data=json.dumps(body).encode(), method="POST", headers={
            "x-api-key": self.key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            raise ProviderError(f"anthropic API error {exc.code}: {exc.read()[:300]!r}") from exc
        text = "".join(block.get("text", "") for block in data.get("content", []) if block.get("type") == "text")
        return _parse_json(text)

    def _log(self, contract: dict[str, Any], size: int) -> None:
        self.egress_log.parent.mkdir(parents=True, exist_ok=True)
        with self.egress_log.open("a") as fh:
            fh.write(json.dumps({"time": time.time(), "provider": self.name, "model": self.model,
                                 "task": contract.get("id"), "scope": contract.get("scope", [])[:50],
                                 "bytes": size}) + "\n")


def _parse_json(text: str) -> dict[str, Any]:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if m:
        text = m.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ProviderError("provider returned no JSON object")
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError as exc:
        raise ProviderError(f"provider returned invalid JSON: {exc}") from exc


def make_provider(ws: Any) -> Provider:
    cfg = ws.config
    if cfg.provider in ("", "none"):
        return NoneProvider()
    if cfg.provider == "mcp":
        return QueueProvider()
    if cfg.provider == "command":
        return CommandProvider(cfg.provider_command)
    if cfg.provider == "anthropic":
        if not cfg.allow_remote:
            raise ProviderError("remote providers send source excerpts off this machine; set allow_remote=true "
                                "in .promapkin/config.json to permit it")
        return AnthropicProvider(cfg.provider_model, ws.dir / "egress.log")
    raise ProviderError(f"unknown provider {cfg.provider!r}")
