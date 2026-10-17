"""Benchmarks.

1. Retrieval benchmark (`retrieval_benchmark`): recall@k, MRR, context precision, token usage
   versus naive exploration baselines, latency, and an ML-vs-heuristic ablation. Uses a task
   spec, or self-supervised tasks derived from documented symbols when no spec is given.

2. Agent A/B harness (`agent_benchmark`): runs the *same* coding-agent command on identical
   tasks in isolated copies of the repository, once without and once with Promapkin context,
   and records token consumption, task completion, regressions, retrieval accuracy and setup
   overhead. The agent is any command; Promapkin never assumes a particular vendor.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from promapkin.config import Workspace
from promapkin.graph import Graph
from promapkin.retrieval import Retriever, estimate_tokens
from promapkin.store import split_identifier


# ===================================================================================== retrieval

def self_supervised_tasks(ws: Workspace, limit: int = 40) -> list[dict[str, Any]]:
    """Queries from docstrings with the symbol's own name tokens removed, so lexical name matching can't cheat."""
    tasks = []
    for n in ws.store.nodes(where="kind IN ('function','method','class') AND text IS NOT NULL"):
        if n["attrs"].get("test") or not n["file"]:
            continue
        name_tokens = set(split_identifier(n["name"])) | {n["name"].lower()}
        words = [w for w in re.findall(r"[A-Za-z]+", n["text"]) if w.lower() not in name_tokens]
        if len(words) < 3:
            continue
        tasks.append({"query": " ".join(words[:25]), "expected": [n["file"]], "target": n["id"]})
    tasks.sort(key=lambda t: t["target"])
    step = max(1, len(tasks) // limit) if tasks else 1
    return tasks[::step][:limit]


def _expected_files(ws: Workspace, expected: list[str]) -> set[str]:
    g = Graph(ws.store)
    files = set()
    for e in expected:
        if (ws.root / e).is_file():
            files.add(e)
            continue
        try:
            n = ws.store.node(g.resolve(e))
            if n and n.get("file"):
                files.add(n["file"])
        except LookupError:
            pass
    return files


def _naive_exploration_tokens(ws: Workspace, query: str, expected: set[str], file_tokens: dict[str, int]) -> int:
    """Tokens an agent spends reading files ranked by query-term hits until it has read every expected file."""
    terms = {t for t in re.findall(r"[a-z]{3,}", query.lower())} | {t for w in re.findall(r"\w+", query)
                                                                   for t in split_identifier(w) if len(t) > 2}
    scored = []
    for path, toks in file_tokens.items():
        try:
            text = (ws.root / path).read_text(errors="replace").lower()
        except OSError:
            continue
        hits = sum(text.count(t) for t in terms)
        if hits:
            scored.append((-hits, path))
    scored.sort()
    spent, found = 0, set()
    for _, path in scored:
        spent += file_tokens[path]
        if path in expected:
            found.add(path)
        if found >= expected:
            return spent
    return sum(file_tokens.values())   # not found by grep: agent ends up reading everything


def retrieval_benchmark(ws: Workspace, spec: dict[str, Any] | None = None, budget: int = 2000, limit: int = 40,
                        ablation: bool = True) -> dict[str, Any]:
    tasks = (spec or {}).get("tasks") or self_supervised_tasks(ws, limit)
    source_files = [r[0] for r in ws.store.conn.execute(
        "SELECT path FROM files WHERE language IN ('python','javascript','typescript')")]
    file_tokens = {}
    for p in source_files:
        try:
            file_tokens[p] = estimate_tokens((ws.root / p).read_text(errors="replace"))
        except OSError:
            pass
    repo_tokens = sum(file_tokens.values())
    from promapkin import ml
    rt = ml.runtime(ws)
    modes = [("configured", rt.classifier.mode)]
    if ablation and rt.classifier.mode != "heuristic":
        modes.append(("heuristic", "heuristic"))
    results: dict[str, Any] = {}
    original = rt.classifier.mode
    try:
        for label, mode in modes:
            rt.classifier.mode = mode
            ret = Retriever(ws, rt)
            rows = []
            for t in tasks:
                expected = _expected_files(ws, t["expected"])
                if not expected:
                    continue
                t0 = time.perf_counter()
                hits = ret.search(t["query"], limit=10, kinds=["class", "function", "method", "module", "endpoint"])
                latency = time.perf_counter() - t0
                files_ranked = list(dict.fromkeys(_hit_file(h) for h in hits if _hit_file(h)))
                rank = next((i + 1 for i, f in enumerate(files_ranked) if f in expected), None)
                t1 = time.perf_counter()
                pkg = ret.build_context(t["query"], budget=budget, include_source=True)
                ctx_latency = time.perf_counter() - t1
                pkg_files = {f["fact"].split(" @ ")[-1].split(":")[0] for f in pkg["facts"] if " @ " in f["fact"]} | \
                    {s["location"].split(":")[0] for s in pkg["sources"]}
                rows.append({
                    "query": t["query"][:80], "expected": sorted(expected), "rank": rank,
                    "recall@5": int(bool(rank and rank <= 5)), "recall@10": int(bool(rank and rank <= 10)),
                    "mrr": 1.0 / rank if rank else 0.0,
                    "context_hit": int(bool(expected & pkg_files)),
                    "context_precision": round(len(expected & pkg_files) / len(pkg_files), 3) if pkg_files else 0.0,
                    "context_tokens": pkg["estimated_tokens"],
                    "naive_tokens": _naive_exploration_tokens(ws, t["query"], expected, file_tokens),
                    "search_ms": round(latency * 1000, 2), "context_ms": round(ctx_latency * 1000, 2),
                })
            results[label] = _aggregate(rows, repo_tokens) | {"mode": mode, "rows": rows}
    finally:
        rt.classifier.mode = original
    jobs = [j for j in ws.store.last_jobs(20) if j["kind"] == "scan" and j["status"] == "succeeded"]
    setup = {}
    if jobs:
        detail = json.loads(jobs[-1]["detail"] or "{}")
        setup = {"last_scan_seconds": detail.get("seconds"), "files_indexed": detail.get("files"),
                 "index_bytes": os.path.getsize(ws.db_path) if ws.db_path.exists() else None}
    return {"tasks": len(tasks), "task_source": "spec" if spec else "self-supervised (docstrings, names removed)",
            "budget": budget, "repository_tokens": repo_tokens, "setup_overhead": setup, "results": results,
            "notes": ["naive_tokens: tokens read by an agent opening files ranked by query-term hits until every "
                      "expected file is read (lower bound on blind exploration).",
                      "Self-supervised queries test semantic retrieval; supply a spec with real tasks for an "
                      "end-to-end measurement (see agent_benchmark)."]}


def _hit_file(h: dict[str, Any]) -> str | None:
    loc = h.get("location")
    return loc.split(":")[0] if loc else None


def _aggregate(rows: list[dict[str, Any]], repo_tokens: int) -> dict[str, Any]:
    n = len(rows) or 1
    ctx = sum(r["context_tokens"] for r in rows)
    naive = sum(r["naive_tokens"] for r in rows)
    return {
        "evaluated": len(rows),
        "recall@5": round(sum(r["recall@5"] for r in rows) / n, 3),
        "recall@10": round(sum(r["recall@10"] for r in rows) / n, 3),
        "mrr": round(sum(r["mrr"] for r in rows) / n, 3),
        "context_hit_rate": round(sum(r["context_hit"] for r in rows) / n, 3),
        "mean_context_tokens": round(ctx / n, 1),
        "mean_naive_exploration_tokens": round(naive / n, 1),
        "token_reduction_vs_naive": round(1 - ctx / naive, 3) if naive else None,
        "token_reduction_vs_whole_repo": round(1 - (ctx / n) / repo_tokens, 3) if repo_tokens else None,
        "mean_search_ms": round(sum(r["search_ms"] for r in rows) / n, 2),
        "mean_context_ms": round(sum(r["context_ms"] for r in rows) / n, 2),
    }


# ===================================================================================== agent A/B harness

AGENT_SPEC_EXAMPLE = {
    "agent_command": "my-agent --task \"$PROMAPKIN_TASK\"",
    "setup": ["python -m pip install -e ."],
    "regression_command": "python -m pytest -q",
    "tasks": [{"id": "pagination", "task": "Add pagination to search results",
               "check_command": "python -m pytest -q tests/test_pagination.py",
               "expected": ["searchapp/services/search_service.py"]}],
}


def agent_benchmark(ws: Workspace, spec: dict[str, Any], budget: int = 4000, timeout: int = 1800,
                    keep: bool = False) -> dict[str, Any]:
    """Run each task twice in fresh copies of the repository: baseline vs Promapkin-assisted.

    The agent command receives environment variables:
      PROMAPKIN_TASK          task text
      PROMAPKIN_ARM           "baseline" or "promapkin"
      PROMAPKIN_CONTEXT_FILE  (promapkin arm) path to the Markdown context package
      PROMAPKIN_CONTEXT_JSON  (promapkin arm) path to the JSON context package
      PROMAPKIN_MCP_COMMAND   (promapkin arm) command that serves Promapkin MCP for this copy
    It should print a final JSON line with token usage, e.g. {"input_tokens": 1200, "output_tokens": 300}.
    """
    from promapkin.retrieval import render_context
    from promapkin.scanner import scan

    rows = []
    for task in spec["tasks"]:
        for arm in ("baseline", "promapkin"):
            tmp = Path(tempfile.mkdtemp(prefix=f"promapkin-bench-{task['id']}-{arm}-"))
            work = tmp / "repo"
            shutil.copytree(ws.root, work, ignore=shutil.ignore_patterns(".promapkin"))
            env = {**os.environ, "PROMAPKIN_TASK": task["task"], "PROMAPKIN_ARM": arm}
            setup_s = 0.0
            for cmd in spec.get("setup", []):
                _run(cmd, work, env, timeout)
            context_hit = None
            if arm == "promapkin":
                t0 = time.perf_counter()
                wsb = Workspace(work)
                wsb.init()
                scan(wsb)
                pkg = Retriever(wsb).build_context(task["task"], budget=budget)
                setup_s = time.perf_counter() - t0
                (tmp / "context.md").write_text(render_context(pkg))
                (tmp / "context.json").write_text(json.dumps(pkg))
                env["PROMAPKIN_CONTEXT_FILE"] = str(tmp / "context.md")
                env["PROMAPKIN_CONTEXT_JSON"] = str(tmp / "context.json")
                env["PROMAPKIN_MCP_COMMAND"] = f"promapkin --root {work} mcp serve"
                expected = set(task.get("expected", []))
                files = {s["location"].split(":")[0] for s in pkg["sources"]} | {
                    f["fact"].split(" @ ")[-1].split(":")[0] for f in pkg["facts"] if " @ " in f["fact"]}
                context_hit = (len(expected & files) / len(expected)) if expected else None
                wsb.close()
            pre_regress = _run(spec["regression_command"], work, env, timeout) if spec.get("regression_command") \
                else None
            t0 = time.perf_counter()
            agent = _run(spec["agent_command"], work, env, timeout)
            agent_s = time.perf_counter() - t0
            usage = _parse_usage(agent["stdout"])
            check = _run(task["check_command"], work, env, timeout) if task.get("check_command") else None
            post_regress = _run(spec["regression_command"], work, env, timeout) if spec.get("regression_command") \
                else None
            rows.append({
                "task": task["id"], "arm": arm, "completed": bool(check and check["code"] == 0),
                "agent_exit": agent["code"], "agent_seconds": round(agent_s, 2),
                "input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens"),
                "total_tokens": (usage.get("input_tokens") or 0) + (usage.get("output_tokens") or 0)
                if usage else None,
                "regressions": _regressions(pre_regress, post_regress),
                "retrieval_accuracy": context_hit, "setup_seconds": round(setup_s, 2),
            })
            if not keep:
                shutil.rmtree(tmp, ignore_errors=True)
    return {"rows": rows, "summary": _ab_summary(rows)}


def _run(cmd: str, cwd: Path, env: dict[str, str], timeout: int) -> dict[str, Any]:
    try:
        p = subprocess.run(cmd, shell=True, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)
        return {"code": p.returncode, "stdout": p.stdout[-20000:], "stderr": p.stderr[-5000:]}
    except subprocess.TimeoutExpired:
        return {"code": 124, "stdout": "", "stderr": "timeout"}


def _parse_usage(stdout: str) -> dict[str, int]:
    for line in reversed(stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{") and "tokens" in line:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            return {k: int(v) for k, v in d.items() if k.endswith("tokens") and isinstance(v, (int, float))}
    return {}


def _failures(run: dict[str, Any] | None) -> set[str]:
    if not run:
        return set()
    text = run["stdout"] + run["stderr"]
    return set(re.findall(r"(?:FAILED|ERROR)\s+(\S+)", text)) or ({"suite"} if run["code"] != 0 else set())


def _regressions(pre: dict[str, Any] | None, post: dict[str, Any] | None) -> int | None:
    if pre is None or post is None:
        return None
    return len(_failures(post) - _failures(pre))


def _ab_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for arm in ("baseline", "promapkin"):
        rs = [r for r in rows if r["arm"] == arm]
        if not rs:
            continue
        toks = [r["total_tokens"] for r in rs if r["total_tokens"] is not None]
        regs = [r["regressions"] for r in rs if r["regressions"] is not None]
        acc = [r["retrieval_accuracy"] for r in rs if r["retrieval_accuracy"] is not None]
        out[arm] = {"tasks": len(rs), "completion_rate": round(sum(r["completed"] for r in rs) / len(rs), 3),
                    "mean_total_tokens": round(sum(toks) / len(toks), 1) if toks else None,
                    "total_regressions": sum(regs) if regs else None,
                    "mean_retrieval_accuracy": round(sum(acc) / len(acc), 3) if acc else None,
                    "mean_setup_seconds": round(sum(r["setup_seconds"] for r in rs) / len(rs), 2),
                    "mean_agent_seconds": round(sum(r["agent_seconds"] for r in rs) / len(rs), 2)}
    b, p = out.get("baseline"), out.get("promapkin")
    if b and p and b["mean_total_tokens"] and p["mean_total_tokens"]:
        out["token_change"] = round(p["mean_total_tokens"] / b["mean_total_tokens"] - 1, 3)
    return out
