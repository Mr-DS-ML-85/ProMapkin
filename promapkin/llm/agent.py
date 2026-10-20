"""LLM-driven commands for humans at the CLI: `ask` (an agent that uses Promapkin's own tools) and
`review` (a change review grounded in the graph). Coding agents get the same tools over MCP."""

from __future__ import annotations

import json
import re
import subprocess
from typing import Any, Callable

from promapkin.llm.engine import SYSTEM_BASE, LLMEngine, redact
from promapkin.llm.errors import LLMError

ASK_SYSTEM = SYSTEM_BASE + """

You answer questions about a software repository using Promapkin's tools, which expose an evidence-aware
code graph, blueprints, impact analysis, data-flow tracing and source excerpts. Prefer the graph tools over
reading whole files. Cite symbols and file:line locations. Say which facts are observed (parsed from source)
and which are inferred. If the tools cannot establish something, say so instead of guessing."""


def ask(ws: Any, engine: LLMEngine, question: str, *, allow_writes: bool = False, max_steps: int = 12,
        on_step: Callable[[str], None] | None = None) -> dict[str, Any]:
    from promapkin.mcp_server import TOOLS, _check_args
    from promapkin.service import Service, bounded

    svc = Service(ws, agent=f"cli-ask:{engine.label}")
    tools = [t for t in TOOLS if t.read_only or allow_writes]
    by_name = {t.name: t for t in tools}
    chat = engine.client.chat(ASK_SYSTEM, [{"name": t.name, "description": t.description, "input_schema": t.schema}
                                           for t in tools])
    chat.user(question)
    trace = []
    for step in range(max_steps):
        engine._reserve()
        turn = chat.send(max_tokens=16000, effort="high")
        if not turn.tool_calls:
            return {"answer": turn.text, "steps": trace, "llm": engine.describe()}
        results = []
        for call in turn.tool_calls:
            tool = by_name.get(call.name)
            if tool is None:
                results.append((call, f"unknown tool {call.name}", True))
                continue
            problems = _check_args(call.arguments, tool.schema)
            if problems:
                results.append((call, "invalid arguments: " + "; ".join(problems), True))
                continue
            if on_step:
                on_step(f"→ {call.name} {json.dumps(call.arguments)[:160]}")
            try:
                out = bounded(tool.handler(svc, call.arguments), max_chars=20_000)
                text, err = redact(json.dumps(out, default=str))[0], False
            except Exception as exc:  # noqa: BLE001 - tool errors go back to the model
                text, err = f"{type(exc).__name__}: {exc}", True
            trace.append({"tool": call.name, "arguments": call.arguments, "error": err})
            results.append((call, text, err))
        chat.tool_results(results)
    chat.user("Step limit reached. Answer now with what you have, noting anything you could not verify.")
    turn = chat.send(max_tokens=8000, effort="medium")
    return {"answer": turn.text, "steps": trace, "llm": engine.describe(), "step_limit_reached": True}


# ------------------------------------------------------------------------------------- review

def changed_hunks(root: Any, base: str | None) -> dict[str, list[tuple[int, int]]]:
    args = ["git", "-C", str(root), "diff", "--unified=0", "--no-color"]
    if base:
        args.append(base)
    out = subprocess.run(args, capture_output=True, text=True, check=False).stdout
    hunks: dict[str, list[tuple[int, int]]] = {}
    current = None
    for line in out.splitlines():
        if line.startswith("+++ "):
            current = None if line.endswith("/dev/null") else line[6:] if line.startswith("+++ b/") else line[4:]
        elif line.startswith("@@") and current:
            m = re.search(r"\+(\d+)(?:,(\d+))?", line)
            if m:
                start, count = int(m.group(1)), int(m.group(2) or 1)
                hunks.setdefault(current, []).append((start, start + max(count, 1) - 1))
    return hunks


def review(ws: Any, engine: LLMEngine | None, base: str | None = None, max_symbols: int = 25) -> dict[str, Any]:
    """Deterministic change review (changed symbols, impact, tests, decisions) plus LLM findings in --llm mode."""
    from promapkin.graph import Graph

    store, g = ws.store, Graph(ws.store)
    hunks = changed_hunks(ws.root, base)
    changed: list[dict[str, Any]] = []
    for path, ranges in hunks.items():
        for n in store.nodes(where="file=? AND kind IN ('function','method','class','test','endpoint')",
                             params=(path,)):
            a, b = n["line_start"] or 0, n["line_end"] or n["line_start"] or 0
            if any(lo <= b and hi >= a for lo, hi in ranges):
                changed.append(n)
    # Innermost symbols first (methods before their classes).
    changed.sort(key=lambda n: ((n["line_end"] or 0) - (n["line_start"] or 0)))
    impacts, tests, decisions = [], {}, {}
    for n in changed[:max_symbols]:
        imp = g.impact(n["id"], depth=3)
        impacts.append({"symbol": n["id"], "location": f"{n['file']}:{n['line_start']}",
                        "affected": {k: [x["id"] for x in v[:10]] for k, v in imp["affected"].items()},
                        "total_affected": imp["total_affected"]})
        for t in imp["tests_to_run"]:
            tests[t["id"]] = t.get("location")
        for d in imp["affected"].get("documentation", []):
            if d["id"].startswith("decision:"):
                decisions[d["id"]] = d.get("summary") or d.get("name")
    report: dict[str, Any] = {
        "base": base or "working tree vs HEAD", "files_changed": sorted(hunks), "changed_symbols":
            [{"id": n["id"], "location": f"{n['file']}:{n['line_start']}"} for n in changed[:max_symbols]],
        "impact": impacts, "tests_to_run": [{"id": k, "location": v} for k, v in tests.items()],
        "decisions_to_respect": [{"id": k, "summary": v} for k, v in decisions.items()],
    }
    if engine is None or not hunks:
        return report
    diff = subprocess.run(["git", "-C", str(ws.root), "diff", "--no-color", *( [base] if base else [])],
                          capture_output=True, text=True, check=False).stdout[:60000]
    schema = {"type": "object", "properties": {
        "summary": {"type": "string"},
        "findings": {"type": "array", "items": {"type": "object", "properties": {
            "severity": {"type": "string", "enum": ["blocker", "major", "minor", "nit"]},
            "file": {"type": "string"}, "line": {"type": "integer"}, "issue": {"type": "string"},
            "suggestion": {"type": "string"}, "evidence": {"type": "string"}}}},
        "missing_tests": {"type": "array", "items": {"type": "string"}}}}
    try:
        report["llm_review"] = engine.call_json(
            "review", "Review this diff for correctness bugs, broken contracts with dependents, violated "
            "architecture decisions and missing tests. Use the impact analysis to reason about callers. Only "
            "report issues you can tie to specific lines of the diff.",
            json.dumps({"diff": diff, "analysis": report}, default=str), schema, subject="review", effort="high")
    except LLMError as exc:
        report["llm_error"] = str(exc)
    return report
