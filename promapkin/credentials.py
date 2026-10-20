"""API credentials for LLM providers.

Resolution order (first match wins):
1. an explicit key passed for this invocation (`--api-key`, discouraged: visible in shell history);
2. the environment (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or the variable named by
   `openai_api_key_env` / `anthropic_api_key_env` in the config);
3. the user credential store `~/.config/promapkin/credentials.json` (mode 0600), managed with
   `promapkin auth set|remove|status`.

Keys are never written into the repository or its `.promapkin/` directory.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

DEFAULT_ENV = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}


def store_path() -> Path:
    base = os.environ.get("PROMAPKIN_CONFIG_HOME") or os.path.join(
        os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config")), "promapkin")
    return Path(base) / "credentials.json"


def _load() -> dict[str, dict[str, str]]:
    p = store_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return {}


def _save(data: dict[str, dict[str, str]]) -> None:
    p = store_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        json.dump(data, fh, indent=2)
    os.replace(tmp, p)
    os.chmod(p, stat.S_IRUSR | stat.S_IWUSR)


def set_key(provider: str, key: str, base_url: str | None = None) -> Path:
    data = _load()
    entry = {"api_key": key.strip()}
    if base_url:
        entry["base_url"] = base_url
    data[provider] = entry
    _save(data)
    return store_path()


def remove(provider: str) -> bool:
    data = _load()
    if provider in data:
        del data[provider]
        _save(data)
        return True
    return False


def resolve(provider: str, explicit: str | None = None, env_name: str | None = None) -> tuple[str | None, str]:
    """Return (key, source). Source is reported to the user; the key itself never is."""
    if explicit:
        return explicit, "command line"
    name = env_name or DEFAULT_ENV.get(provider, "")
    if name and os.environ.get(name):
        return os.environ[name], f"${name}"
    stored = _load().get(provider, {})
    if stored.get("api_key"):
        return stored["api_key"], str(store_path())
    return None, "none"


def stored_base_url(provider: str) -> str | None:
    return _load().get(provider, {}).get("base_url")


def status() -> dict[str, dict[str, str]]:
    out = {}
    for provider, env in DEFAULT_ENV.items():
        key, source = resolve(provider)
        out[provider] = {"configured": bool(key), "source": source,
                         "key": f"{key[:6]}…{key[-4:]}" if key and len(key) > 12 else ("set" if key else "")}
    return out
