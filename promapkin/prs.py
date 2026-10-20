"""Pull-request dashboard over the graph: `promapkin prs`.

For each open pull request (GitHub, through the `gh` CLI or the REST API with a token) or each unmerged local
branch (no network needed), it reports CI state, review state, which worktree has it checked out, and what it
touches in the graph: modules, communities, changed symbols, the blast radius and the tests to run. Pairs of
PRs are flagged for merge-order risk when they change the same files (likely textual conflicts), the same
modules, or the same communities (likely semantic conflicts). `--triage` orders the review queue: a
deterministic score by default, or a model's ranking with reasons in --llm mode.

Data from GitHub is untrusted text (titles, bodies, branch names) and is never executed or followed.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from promapkin.config import Workspace

HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", re.M)


class PRError(RuntimeError):
    pass


# ====================================================================== sources
def _git(root: Path, *args: str, timeout: int = 30) -> str:
    try:
        r = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, timeout=timeout,
                           check=False)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return r.stdout if r.returncode == 0 else ""


def github_remote(root: Path) -> tuple[str, str, str] | None:
    """(host, owner, repo) of the `origin` remote when it is a GitHub-style URL."""
    url = _git(root, "remote", "get-url", "origin").strip()
    m = re.match(r"(?:https?://(?:[^@/]+@)?|git@|ssh://git@)([^/:]+)[/:]([^/]+)/([^/]+?)(?:\.git)?/?$", url)
    if not m:
        return None
    host = m.group(1)
    if host != "github.com" and host != os.environ.get("GH_HOST"):
        return None
    return host, m.group(2), m.group(3)


def worktrees(root: Path) -> dict[str, str]:
    """branch -> worktree path, from `git worktree list --porcelain`."""
    out, path = {}, None
    for line in _git(root, "worktree", "list", "--porcelain").splitlines():
        if line.startswith("worktree "):
            path = line.split(" ", 1)[1]
        elif line.startswith("branch ") and path:
            out[line.split(" ", 1)[1].removeprefix("refs/heads/")] = path
    return out


class GitHub:
    """Minimal GitHub REST client: the `gh` CLI when available (it handles auth and enterprise hosts),
    otherwise urllib with GITHUB_TOKEN / GH_TOKEN. `get` can be replaced for tests."""

    def __init__(self, host: str, owner: str, repo: str, get: Callable[[str], Any] | None = None):
        self.host, self.owner, self.repo = host, owner, repo
        self._get = get or self._default_get()

    def _default_get(self) -> Callable[[str], Any]:
        token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
        gh = _which("gh")
        if not token and gh:
            def via_gh(path: str) -> Any:
                r = subprocess.run([gh, "api", "--hostname", self.host, path], capture_output=True, text=True,
                                   timeout=60, check=False)
                if r.returncode != 0:
                    raise PRError(f"gh api {path}: {r.stderr.strip()[:300]}")
                return json.loads(r.stdout or "null")
            return via_gh
        api = "https://api.github.com" if self.host == "github.com" else f"https://{self.host}/api/v3"

        def via_http(path: str) -> Any:
            req = urllib.request.Request(f"{api}/{path.lstrip('/')}", headers={
                "Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "promapkin", **({"Authorization": f"Bearer {token}"} if token else {})})
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    return json.loads(resp.read().decode())
            except urllib.error.HTTPError as exc:
                hint = "" if token else " (set GITHUB_TOKEN or install the gh CLI for private repositories)"
                raise PRError(f"GitHub API {exc.code} for {path}{hint}") from exc
            except urllib.error.URLError as exc:
                raise PRError(f"cannot reach {api}: {exc.reason}") from exc
        return via_http

    def get(self, path: str) -> Any:
        return self._get(path)

    def repo_path(self, suffix: str) -> str:
        return f"repos/{self.owner}/{self.repo}/{suffix}"

    def open_prs(self, limit: int = 30) -> list[dict[str, Any]]:
        return self.get(self.repo_path(f"pulls?state=open&per_page={min(limit, 100)}")) or []

    def pr(self, number: int) -> dict[str, Any]:
        return self.get(self.repo_path(f"pulls/{number}"))

    def files(self, number: int) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for page in range(1, 6):   # up to 500 files
            batch = self.get(self.repo_path(f"pulls/{number}/files?per_page=100&page={page}")) or []
            out += batch
            if len(batch) < 100:
                break
        return out

    def reviews(self, number: int) -> list[dict[str, Any]]:
        return self.get(self.repo_path(f"pulls/{number}/reviews?per_page=100")) or []

    def checks(self, sha: str) -> dict[str, Any]:
        runs = (self.get(self.repo_path(f"commits/{sha}/check-runs?per_page=100")) or {}).get("check_runs", [])
        status = self.get(self.repo_path(f"commits/{sha}/status")) or {}
        return {"runs": runs, "statuses": status.get("statuses", [])}


def _which(name: str) -> str | None:
    from shutil import which
    return which(name)


def ci_state(checks: dict[str, Any]) -> dict[str, Any]:
    failing, pending, passing = [], [], 0
    for r in checks.get("runs", []):
        if r.get("status") != "completed":
            pending.append(r.get("name"))
        elif r.get("conclusion") in ("failure", "timed_out", "cancelled", "action_required", "startup_failure"):
            failing.append(r.get("name"))
        else:
            passing += 1
    for s in checks.get("statuses", []):
        st = s.get("state")
        if st in ("failure", "error"):
            failing.append(s.get("context"))
        elif st == "pending":
            pending.append(s.get("context"))
        else:
            passing += 1
    state = "failing" if failing else "pending" if pending else "passing" if passing else "none"
    return {"state": state, "failing": failing[:10], "pending": pending[:10], "passing": passing}


def review_state(reviews: list[dict[str, Any]], requested: list[Any]) -> dict[str, Any]:
    latest: dict[str, str] = {}
    for r in reviews:
        user = (r.get("user") or {}).get("login")
        if user and r.get("state") in ("APPROVED", "CHANGES_REQUESTED", "DISMISSED"):
            latest[user] = r["state"]
    approved = sorted(u for u, s in latest.items() if s == "APPROVED")
    changes = sorted(u for u, s in latest.items() if s == "CHANGES_REQUESTED")
    pending = sorted((r.get("login") or r.get("name") or "?") for r in requested or [])
    decision = "changes_requested" if changes else "approved" if approved else \
        "review_requested" if pending else "no_review"
    return {"decision": decision, "approved_by": approved, "changes_requested_by": changes, "requested": pending}


def changed_lines(patch: str | None) -> list[tuple[int, int]]:
    """New-file line ranges touched by a unified diff."""
    out = []
    for m in HUNK.finditer(patch or ""):
        start, count = int(m.group(1)), int(m.group(2) or 1)
        out.append((start, start + max(count, 1) - 1))
    return out


# ====================================================================== collection
def collect_github(ws: Workspace, gh: GitHub, limit: int = 30, number: int | None = None) -> list[dict[str, Any]]:
    trees = worktrees(ws.root)
    raw = [gh.pr(number)] if number else gh.open_prs(limit)
    out = []
    for p in raw[:limit]:
        n = p["number"]
        files = gh.files(n)
        detail = p if "mergeable" in p else gh.pr(n)
        head = (p.get("head") or {})
        out.append({
            "id": f"#{n}", "number": n, "title": p.get("title", ""), "author": (p.get("user") or {}).get("login"),
            "branch": head.get("ref"), "base": (p.get("base") or {}).get("ref"), "draft": bool(p.get("draft")),
            "url": p.get("html_url"), "created_at": p.get("created_at"), "updated_at": p.get("updated_at"),
            "mergeable": detail.get("mergeable"), "mergeable_state": detail.get("mergeable_state"),
            "ci": ci_state(gh.checks(head.get("sha"))) if head.get("sha") else {"state": "none"},
            "review": review_state(gh.reviews(n), (p.get("requested_reviewers") or []) +
                                   (p.get("requested_teams") or [])),
            "worktree": trees.get(head.get("ref") or ""),
            "files": [{"path": f["filename"], "status": f.get("status"), "additions": f.get("additions", 0),
                       "deletions": f.get("deletions", 0), "lines": changed_lines(f.get("patch")),
                       "previous": f.get("previous_filename")} for f in files],
        })
    return out


def collect_local(ws: Workspace, base: str | None = None, branch: str | None = None) -> list[dict[str, Any]]:
    """Unmerged local branches (and the worktrees they are checked out in), compared with the base branch."""
    root = ws.root
    base = base or _default_base(root)
    trees = worktrees(root)
    names = [branch] if branch else [b.strip() for b in _git(root, "for-each-ref", "--format=%(refname:short)",
                                                               "refs/heads").splitlines() if b.strip()]
    out = []
    for name in names:
        if name == base:
            continue
        ahead = _git(root, "rev-list", "--count", f"{base}..{name}").strip()
        if not branch and ahead in ("", "0"):
            continue
        stat = _git(root, "diff", "--numstat", f"{base}...{name}")
        diff = _git(root, "diff", "--unified=0", f"{base}...{name}")
        patches = _split_patches(diff)
        files = []
        for line in stat.splitlines():
            parts = line.split("\t")
            if len(parts) == 3:
                add, rem, path = parts
                path = path.split(" => ")[-1].strip("{}")
                files.append({"path": path, "status": "modified", "additions": int(add) if add.isdigit() else 0,
                              "deletions": int(rem) if rem.isdigit() else 0, "lines": changed_lines(patches.get(path))})
        log = _git(root, "log", "-1", "--format=%an%x09%cI%x09%s", name).strip().split("\t")
        behind = _git(root, "rev-list", "--count", f"{name}..{base}").strip()
        out.append({"id": name, "number": None, "title": log[2] if len(log) > 2 else name,
                    "author": log[0] if log else None, "branch": name, "base": base, "draft": False, "url": None,
                    "updated_at": log[1] if len(log) > 1 else None, "created_at": None,
                    "ahead": int(ahead or 0), "behind": int(behind or 0), "mergeable": None,
                    "ci": {"state": "unknown"}, "review": {"decision": "local"}, "worktree": trees.get(name),
                    "files": files})
    return out


def _default_base(root: Path) -> str:
    head = _git(root, "symbolic-ref", "--short", "refs/remotes/origin/HEAD").strip()
    if head:
        return head.split("/", 1)[-1] if _git(root, "rev-parse", "--verify", head.split("/", 1)[-1]) else head
    for cand in ("main", "master", "trunk", "develop"):
        if _git(root, "rev-parse", "--verify", "--quiet", cand).strip():
            return cand
    return "HEAD"


def _split_patches(diff: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for chunk in re.split(r"^diff --git ", diff, flags=re.M):
        m = re.search(r"^\+\+\+ b/(.+)$", chunk, re.M)
        if m:
            out[m.group(1).strip()] = chunk
    return out


# ====================================================================== graph analysis
def analyse(ws: Workspace, prs: list[dict[str, Any]], deep: bool = False) -> dict[str, Any]:
    from promapkin import insights
    from promapkin.graph import Graph
    s = ws.store
    g = Graph(s)
    member = {m: c["component"] for c in insights.communities(s) for m in c["modules"]}
    hub_ids = {h["id"] for h in insights.hubs(s, 25)}
    for pr in prs:
        modules, comms, symbols, hubs, unknown = set(), set(), [], [], []
        for f in pr["files"]:
            rows = s.conn.execute("SELECT id, kind, name, line_start, line_end FROM nodes WHERE owner_file = ? AND "
                                  "kind IN ('module','class','function','method','endpoint','test','table',"
                                  "'service','deployment','cloud_resource')", (f["path"],)).fetchall()
            if not rows:
                unknown.append(f["path"])
            for nid, kind, name, a, b in rows:
                if kind == "module":
                    modules.add(nid)
                    if nid in member:
                        comms.add(member[nid])
                    continue
                touched = not f["lines"] or any(a and lo <= (b or a) and hi >= a for lo, hi in f["lines"])
                if touched:
                    symbols.append(nid)
                    if nid in hub_ids:
                        hubs.append(nid)
        tests, affected, endpoints = set(), set(), set()
        for nid in symbols[:30 if deep else 12]:
            try:
                imp = g.impact(nid, depth=3 if deep else 2, limit=60)
            except LookupError:
                continue
            for group, items in (imp.get("affected") or {}).items():
                for it in items:
                    affected.add(it["id"])
                    if group == "endpoints":
                        endpoints.add(it["id"])
            tests |= {t["id"] if isinstance(t, dict) else t for t in imp.get("tests_to_run", [])}
        size = sum(f["additions"] + f["deletions"] for f in pr["files"])
        pr["graph"] = {"modules": sorted(modules), "communities": sorted(comms),
                       "changed_symbols": symbols[:200], "hubs_touched": hubs, "affected": len(affected),
                       "endpoints_affected": sorted(endpoints)[:20], "tests_to_run": sorted(tests)[:40],
                       "unindexed_files": unknown[:20], "lines_changed": size}
        pr["risk"] = _risk(pr)
    return {"prs": prs, "conflicts": conflicts(prs)}


def _risk(pr: dict[str, Any]) -> dict[str, Any]:
    g = pr["graph"]
    score = min(1.0, g["lines_changed"] / 800) * 0.3 + min(1.0, g["affected"] / 60) * 0.3 + \
        min(1.0, len(g["hubs_touched"]) / 3) * 0.2 + min(1.0, len(g["communities"]) / 4) * 0.2
    reasons = []
    if g["hubs_touched"]:
        reasons.append(f"changes {len(g['hubs_touched'])} hub symbol(s)")
    if len(g["communities"]) > 1:
        reasons.append(f"spans {len(g['communities'])} communities")
    if g["affected"]:
        reasons.append(f"{g['affected']} dependent symbol(s)")
    if g["changed_symbols"] and not g["tests_to_run"]:
        reasons.append("no tests reach the changed code")
        score = min(1.0, score + 0.15)
    return {"score": round(score, 2), "level": "high" if score >= 0.6 else "medium" if score >= 0.3 else "low",
            "reasons": reasons}


def conflicts(prs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for i, a in enumerate(prs):
        fa = {f["path"] for f in a["files"]}
        for b in prs[i + 1:]:
            fb = {f["path"] for f in b["files"]}
            files = sorted(fa & fb)
            mods = sorted(set(a["graph"]["modules"]) & set(b["graph"]["modules"]))
            comms = sorted(set(a["graph"]["communities"]) & set(b["graph"]["communities"]))
            syms = sorted(set(a["graph"]["changed_symbols"]) & set(b["graph"]["changed_symbols"]))
            if not (files or comms):
                continue
            level = "high" if syms or files else "medium" if mods else "low"
            out.append({"a": a["id"], "b": b["id"], "level": level, "shared_files": files[:10],
                        "shared_symbols": syms[:10], "shared_modules": mods[:10], "shared_communities": comms,
                        "why": "both change the same symbols" if syms else "both change the same files" if files
                        else "both change the same modules" if mods else "both change the same communities"})
    return sorted(out, key=lambda c: ({"high": 0, "medium": 1, "low": 2}[c["level"]], c["a"]))


def triage(ws: Workspace, report: dict[str, Any], llm: Any = None) -> list[dict[str, Any]]:
    """Order the review queue. Deterministic: ready-to-merge first, then small and low-risk, failing CI last."""
    conflict_count: dict[str, int] = defaultdict(int)
    for c in report["conflicts"]:
        if c["level"] != "low":
            conflict_count[c["a"]] += 1
            conflict_count[c["b"]] += 1
    rows = []
    for pr in report["prs"]:
        ci = pr["ci"]["state"]
        rev = pr["review"]["decision"]
        score = 0.0
        why = []
        if rev == "approved" and ci in ("passing", "none", "unknown"):
            score += 3
            why.append("approved and CI green: merge")
        if ci == "failing":
            score -= 2
            why.append(f"CI failing ({', '.join(pr['ci'].get('failing', [])[:3])}): back to author")
        if rev == "changes_requested":
            score -= 1.5
            why.append("changes requested")
        if pr.get("draft"):
            score -= 3
            why.append("draft")
        score += (1 - pr["risk"]["score"]) * 1.5
        score += max(0.0, 1 - pr["graph"]["lines_changed"] / 600)
        if conflict_count[pr["id"]]:
            why.append(f"overlaps {conflict_count[pr['id']]} other PR(s): agree merge order")
        rows.append({"id": pr["id"], "title": pr["title"], "score": round(score, 2), "why": why or
                     [f"{pr['risk']['level']} risk, {pr['graph']['lines_changed']} lines"]})
    rows.sort(key=lambda r: -r["score"])
    if llm is not None and rows:
        rows = _llm_triage(llm, report, rows)
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    return rows


def _llm_triage(llm: Any, report: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from promapkin.llm.errors import LLMError
    schema = {"type": "object", "properties": {"order": {"type": "array", "items": {"type": "object", "properties": {
        "id": {"type": "string"}, "reason": {"type": "string"}}}}}}
    summary = [{"id": p["id"], "title": p["title"][:120], "draft": p["draft"], "ci": p["ci"]["state"],
                "review": p["review"]["decision"], "risk": p["risk"], "lines": p["graph"]["lines_changed"],
                "communities": p["graph"]["communities"], "tests": len(p["graph"]["tests_to_run"]),
                "hubs": len(p["graph"]["hubs_touched"])} for p in report["prs"]]
    try:
        data = llm.call_json("triage_prs", "Order this pull-request review queue for a maintainer: what to review "
                             "or merge first and why (one short reason each). Use only the data given.",
                             json.dumps({"prs": summary, "conflicts": report["conflicts"][:20]}), schema,
                             subject="pr triage", effort="low")
    except LLMError as exc:
        rows[0].setdefault("note", f"LLM triage unavailable ({str(exc)[:120]}); deterministic order")
        return rows
    by_id = {r["id"]: r for r in rows}
    ordered = []
    for item in data.get("order", []):
        r = by_id.pop(item.get("id"), None)
        if r:
            r["llm_reason"] = item.get("reason", "")
            ordered.append(r)
    return ordered + list(by_id.values())


# ====================================================================== entry point
def dashboard(ws: Workspace, source: str = "auto", limit: int = 30, number: int | str | None = None,
              base: str | None = None, do_triage: bool = False, llm: Any = None,
              gh: GitHub | None = None) -> dict[str, Any]:
    started = time.time()
    remote = github_remote(ws.root)
    use = source
    if source == "auto":
        token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
        use = "github" if remote and (gh or token or _which("gh")) else "local"
    if use == "github":
        if gh is None:
            if not remote:
                raise PRError("origin is not a GitHub remote; use --local")
            gh = GitHub(*remote)
        prs = collect_github(ws, gh, limit, int(number) if number is not None else None)
        where = f"{gh.owner}/{gh.repo}"
    else:
        prs = collect_local(ws, base, str(number) if number is not None else None)
        where = f"local branches vs {base or _default_base(ws.root)}"
    report = analyse(ws, prs, deep=number is not None)
    report.update({"source": use, "repository": where, "seconds": round(time.time() - started, 2),
                   "indexed_revision": ws.store.get_meta("indexed_revision")})
    if do_triage:
        report["triage"] = triage(ws, report, llm)
    if number is not None and prs:
        report["detail"] = _detail(ws, prs[0], report)
    return report


def _detail(ws: Workspace, pr: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    s = ws.store
    files = [f["path"] for f in pr["files"]]
    decisions, constraints = [], []
    for f in files:
        for e in s.conn.execute("SELECT src, dst, type FROM edges WHERE type IN ('constrains','affects','records') "
                                "AND dst IN (SELECT id FROM nodes WHERE owner_file = ?)", (f,)).fetchall():
            n = s.node(e[0])
            if n and n["kind"] in ("decision", "constraint", "requirement"):
                (decisions if n["kind"] == "decision" else constraints).append(
                    {"id": n["id"], "name": n["name"], "file": n["file"]})
    others = [c for c in report["conflicts"] if pr["id"] in (c["a"], c["b"])]
    return {"files": pr["files"], "changed_symbols": pr["graph"]["changed_symbols"],
            "decisions": decisions[:20], "constraints": constraints[:20], "conflicts_with_other_prs": others}
