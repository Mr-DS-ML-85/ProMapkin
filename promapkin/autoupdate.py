"""Keeping the index current without thinking about it: git hooks and a file watcher.

- `promapkin hook install` adds a marked block to `post-commit`, `post-checkout`, `post-merge` and
  `post-rewrite` that runs an incremental scan in the background, so commits, branch switches and pulls never
  wait on it. Existing hook scripts are kept; only the marked block is added or removed. `core.hooksPath` is
  respected.
- `promapkin watch` polls the working tree (no dependencies) and runs an incremental scan once changes settle.
  Neither ever calls an LLM unless `--llm` is passed explicitly.
"""

from __future__ import annotations

import shlex
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

HOOKS = ("post-commit", "post-checkout", "post-merge", "post-rewrite")
START, END = "# >>> promapkin (managed; `promapkin hook uninstall` removes it)", "# <<< promapkin"


def hooks_dir(root: Path) -> Path:
    custom = _git(root, "config", "--get", "core.hooksPath")
    if custom:
        p = Path(custom)
        return p if p.is_absolute() else root / p
    common = _git(root, "rev-parse", "--git-common-dir") or ".git"
    p = Path(common)
    return (p if p.is_absolute() else root / p) / "hooks"


def _git(root: Path, *args: str) -> str:
    try:
        return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, timeout=10,
                              check=False).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def hook_command(root: Path, blueprint: bool = False) -> str:
    exe = [sys.executable, "-m", "promapkin"]
    cmd = " ".join(shlex.quote(c) for c in exe + ["--root", str(root), "sync" if blueprint else "scan"])
    log = shlex.quote(str(root / ".promapkin" / "hook.log"))
    # Background and detached: git returns immediately; a busy index just waits its turn (db_busy_timeout).
    return f'[ "$PROMAPKIN_HOOKS" = "0" ] || ( {cmd} >> {log} 2>&1 & ) >/dev/null 2>&1'


def install_hooks(root: Path, blueprint: bool = False) -> dict[str, Any]:
    d = hooks_dir(root)
    d.mkdir(parents=True, exist_ok=True)
    block = f"{START}\n{hook_command(root, blueprint)}\n{END}\n"
    out = []
    for name in HOOKS:
        path = d / name
        text = path.read_text() if path.exists() else "#!/bin/sh\n"
        new = _strip(text).rstrip("\n") + "\n" + block
        status = "unchanged" if new == text else ("updated" if path.exists() else "created")
        if new != text:
            path.write_text(new)
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        out.append({"hook": name, "path": str(path), "status": status})
    return {"hooks_dir": str(d), "hooks": out, "runs": "sync (scan + affected blueprint sections)" if blueprint
            else "incremental scan", "log": str(root / ".promapkin" / "hook.log")}


def uninstall_hooks(root: Path) -> dict[str, Any]:
    d = hooks_dir(root)
    out = []
    for name in HOOKS:
        path = d / name
        if not path.exists() or START not in path.read_text():
            continue
        rest = _strip(path.read_text())
        if rest.strip() in ("", "#!/bin/sh"):
            path.unlink()
            out.append({"hook": name, "status": "deleted"})
        else:
            path.write_text(rest)
            out.append({"hook": name, "status": "block removed"})
    return {"hooks_dir": str(d), "hooks": out}


def hook_status(root: Path) -> dict[str, Any]:
    d = hooks_dir(root)
    return {"hooks_dir": str(d), "installed": [n for n in HOOKS if (d / n).exists() and START in (d / n).read_text()]}


def _strip(text: str) -> str:
    if START not in text:
        return text
    head, _, rest = text.partition(START)
    _, _, tail = rest.partition(END)
    return (head.rstrip("\n") + "\n" + tail.lstrip("\n")).replace("\n\n\n", "\n\n")


# ------------------------------------------------------------------ watcher
def snapshot(ws: Any) -> dict[str, tuple[int, int]]:
    """(mtime_ns, size) of every file the scanner would consider."""
    from promapkin.scanner import _candidate_files
    out = {}
    for rel in _candidate_files(ws):
        try:
            st = (ws.root / rel).stat()
        except OSError:
            continue
        out[rel] = (st.st_mtime_ns, st.st_size)
    return out


def watch(ws: Any, interval: float = 1.0, debounce: float = 1.5, blueprint: bool = False,
          on_event: Callable[[dict[str, Any]], None] | None = None, max_rounds: int | None = None,
          stop: Callable[[], bool] | None = None) -> int:
    """Poll for changes; after `debounce` seconds of quiet, run an incremental scan (or sync). Returns the number
    of rebuilds. `max_rounds`/`stop` exist for tests and embedding."""
    from promapkin.service import Service
    svc = Service(ws, agent="watch")
    say = on_event or (lambda _e: None)
    prev = snapshot(ws)
    pending: set[str] = set()
    last_change = 0.0
    rebuilds = rounds = 0
    say({"event": "watching", "files": len(prev), "interval": interval})
    while not (stop and stop()) and (max_rounds is None or rounds < max_rounds):
        rounds += 1
        time.sleep(interval)
        cur = snapshot(ws)
        changed = {p for p in cur.keys() | prev.keys() if cur.get(p) != prev.get(p)}
        prev = cur
        if changed:
            pending |= changed
            last_change = time.monotonic()
            continue
        if pending and time.monotonic() - last_change >= debounce:
            started = time.monotonic()
            try:
                r = svc.sync() if blueprint else svc.scan()
                say({"event": "rebuilt", "changed": sorted(pending)[:20], "count": len(pending),
                     "seconds": round(time.monotonic() - started, 2),
                     "summary": {k: r.get(k) for k in ("changed", "deleted", "resolved_refs") if k in r}})
                rebuilds += 1
            except Exception as exc:  # noqa: BLE001 - keep watching; report and retry on the next change
                say({"event": "error", "error": f"{type(exc).__name__}: {exc}"})
            pending.clear()
    return rebuilds


