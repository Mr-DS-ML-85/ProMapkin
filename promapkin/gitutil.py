"""Git integration. Read-only: Promapkin never mutates the repository it analyzes."""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

IGNORED_DIRS = {
    ".git", ".hg", ".svn", ".promapkin", "node_modules", "__pycache__", ".venv", "venv", "env",
    "dist", "build", "target", ".mypy_cache", ".pytest_cache", ".tox", ".idea", ".vscode", "vendor",
    ".next", "coverage", ".eggs",
}


def _git(root: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True, text=True, timeout=30, check=False,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout if out.returncode == 0 else None


def is_git_repo(root: Path) -> bool:
    return _git(root, "rev-parse", "--is-inside-work-tree") is not None


def find_root(start: Path) -> Path:
    out = _git(start, "rev-parse", "--show-toplevel")
    return Path(out.strip()) if out else start.resolve()


def head_revision(root: Path) -> str:
    out = _git(root, "rev-parse", "HEAD")
    return out.strip() if out else "worktree"


def is_dirty(root: Path) -> bool:
    out = _git(root, "status", "--porcelain", "--", ".", ":(exclude).promapkin")
    return bool(out and out.strip())


def revision_label(root: Path) -> str:
    rev = head_revision(root)
    return f"{rev}+dirty" if rev != "worktree" and is_dirty(root) else rev


def list_files(root: Path) -> list[str]:
    """Tracked and untracked-but-not-ignored files, relative to root, POSIX separators."""
    out = _git(root, "ls-files", "-co", "--exclude-standard", "-z")
    if out is not None:
        files = [f for f in out.split("\0") if f]
        return sorted(f for f in files if not set(Path(f).parts) & IGNORED_DIRS and (root / f).is_file())
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in IGNORED_DIRS]
        for name in filenames:
            found.append(Path(dirpath, name).relative_to(root).as_posix())
    return sorted(found)


def suspect_files(root: Path, revision: str) -> set[str] | None:
    """Files whose content may differ from `revision`: tracked files changed since it (staged or not) and untracked
    files, relative to root. None when git cannot tell (not a repository, unknown revision)."""
    rev = revision.split("+")[0]
    if not rev or rev == "worktree":
        return None
    diff = _git(root, "diff", "--name-only", "--relative", "-z", rev, "--")
    untracked = _git(root, "ls-files", "-o", "--exclude-standard", "-z")
    if diff is None or untracked is None:
        return None
    return {f for f in diff.split("\0") if f} | {f for f in untracked.split("\0") if f}


def recent_commits(root: Path, limit: int = 20) -> list[dict[str, str]]:
    out = _git(root, "log", f"-{limit}", "--pretty=format:%H%x1f%an%x1f%ad%x1f%s", "--date=short")
    if not out:
        return []
    commits = []
    for line in out.splitlines():
        parts = line.split("\x1f")
        if len(parts) == 4:
            commits.append({"sha": parts[0], "author": parts[1], "date": parts[2], "subject": parts[3]})
    return commits


def file_churn(root: Path, limit: int = 500) -> dict[str, int]:
    """How often each file changed in recent history (a cheap risk signal)."""
    out = _git(root, "log", f"-{limit}", "--name-only", "--pretty=format:")
    churn: dict[str, int] = {}
    for line in (out or "").splitlines():
        if line.strip():
            churn[line.strip()] = churn.get(line.strip(), 0) + 1
    return churn


def changed_since(root: Path, revision: str) -> list[str] | None:
    if not revision or revision == "worktree":
        return None
    out = _git(root, "diff", "--name-only", revision.split("+")[0])
    return [l for l in out.splitlines() if l] if out is not None else None


def sha_of(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()
