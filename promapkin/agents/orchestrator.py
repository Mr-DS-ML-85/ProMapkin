"""Orchestrator: plans which specialists run, gives each a bounded contract, merges results.

Specialists do not each re-read the repository. They receive extracted facts for their
scope; optional model providers add a synthesis over those facts which is validated and
merged as agent inference. A targeted change only re-runs the affected specialists.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from promapkin.agents import specialists as sp
from promapkin.agents.contract import TaskContract, merge_result
from promapkin.agents.providers import Provider, ProviderError, QueueProvider, make_provider
from promapkin.config import Workspace
from promapkin.graph import Graph

SECTIONS = ["reconnaissance", "architecture", "dataflow", "api", "patterns", "decisions", "risks", "verification",
            "roadmap"]
REQUIRES = {"decisions": ["architecture", "patterns"], "risks": ["architecture"], "roadmap": []}
OBJECTIVES = {
    "reconnaissance": "Identify project boundaries, conventions, entry points and missing context.",
    "architecture": "Synthesise subsystems, layers, boundaries, integrations and architectural style.",
    "dataflow": "Interpret execution and data-flow paths from entry points to persistence and external systems.",
    "api": "Describe the API surface, schemas, auth boundaries and dependency risks.",
    "patterns": "Assess detected design patterns; confirm, reject or refine them with evidence.",
    "decisions": "Reconstruct architectural decisions; separate documented rationale from inference.",
    "risks": "Prioritise risks and explain their likely consequences.",
    "verification": "Find contradictions between documentation, findings and code.",
    "roadmap": "Turn findings into a dependency-aware implementation plan with validation.",
}


def plan_for_changes(changed_files: list[str]) -> list[str]:
    """Only the specialists whose inputs changed."""
    if not changed_files:
        return []
    sections: set[str] = {"verification"}
    for f in changed_files:
        low = f.lower()
        if low.endswith((".md", ".rst", ".markdown")):
            sections |= {"decisions"}
        elif low.endswith(("package.json", "pyproject.toml", "requirements.txt", "go.mod", "cargo.toml")):
            sections |= {"architecture", "api", "reconnaissance"}
        else:
            sections |= {"architecture", "dataflow", "api", "patterns", "risks", "decisions"}
    return [s for s in SECTIONS if s in sections]


def _ordered(sections: list[str]) -> list[str]:
    want = set(sections)
    for s in list(want):
        want |= set(REQUIRES.get(s, []))
    return [s for s in SECTIONS if s in want]


def run(ws: Workspace, sections: list[str] | None = None, goal: str | None = None,
        provider: Provider | None = None, progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    from promapkin import ml

    say = progress or (lambda _m: None)
    rt = ml.runtime(ws)
    ctx = sp.Ctx(ws, ws.store, Graph(ws.store), rt, ws.revision())
    requested = sections or [s for s in SECTIONS if s != "roadmap" or goal]
    order = _ordered(requested)
    try:
        provider = provider or make_provider(ws)
    except ProviderError as exc:
        say(f"provider unavailable ({exc}); using deterministic specialists only")
        provider = Provider()
    out: dict[str, Any] = {}
    agent_log: list[dict[str, Any]] = []
    for name in order:
        say(f"specialist: {name}")
        with ws.store.transaction():
            section = _run_specialist(ctx, name, goal)
        ctx.cache[name] = section
        out[name] = section
        if provider.name == "none":
            continue
        contract = TaskContract(specialist=name, objective=OBJECTIVES[name] + (f" Goal: {goal}" if goal and
                                                                             name == "roadmap" else ""),
                                scope=_scope(section), revision=ctx.revision, facts=_bounded(section, 12000))
        if isinstance(provider, QueueProvider):
            ws.store.put_agent_task(contract.id, name, ctx.revision, contract.to_dict())
            section["agent_task"] = {"id": contract.id, "status": "pending",
                                     "note": "queued for an external agent (MCP: get_analysis_tasks)"}
            continue
        result, error = None, None
        for _attempt in range(contract.retry_limit + 1):
            try:
                result = provider.complete(contract.to_dict())
            except ProviderError as exc:
                error = str(exc)
                continue
            if result is None:
                break
            merged = merge_result(ws.store, ws.root, contract.to_dict(), result, agent=provider.name)
            if merged["accepted"]:
                section["synthesis"] = {"summary": result.get("summary"), "claims": merged["claims"],
                                        "questions": result.get("questions", []), "provider": provider.name}
                agent_log.append({"specialist": name, **{k: v for k, v in merged.items() if k != "claims"}})
                break
            error = "; ".join(merged["errors"][:3])
        if error and "synthesis" not in section:
            section["synthesis_error"] = error
    out["_meta"] = {"revision": ctx.revision, "sections": order, "provider": provider.name, "ml": rt.describe(),
                    "agent_results": agent_log}
    return out


def _run_specialist(ctx: sp.Ctx, name: str, goal: str | None) -> dict[str, Any]:
    if name == "reconnaissance":
        return sp.reconnaissance(ctx)
    if name == "architecture":
        return sp.architecture(ctx)
    if name == "dataflow":
        return sp.dataflow(ctx)
    if name == "api":
        return sp.api_atlas(ctx)
    if name == "patterns":
        return sp.patterns(ctx)
    if name == "decisions":
        return sp.decisions(ctx, ctx.cache.get("patterns") or sp.patterns(ctx))
    if name == "risks":
        return sp.risks(ctx)
    if name == "verification":
        return sp.verification(ctx)
    if name == "roadmap":
        plan = sp.roadmap(ctx, goal)
        llm = getattr(ctx.ws, "llm", None)
        if llm is not None and plan.get("steps") and goal:
            from promapkin.llm.errors import LLMError
            try:
                refined = llm.refine_roadmap(plan, {"constraints": plan.get("constraints", [])})
                valid = {s for st in plan["steps"] for s in st.get("components", [])}
                for st in refined.get("steps", []):   # components must exist in the deterministic plan or graph
                    st["components"] = [c for c in st.get("components", []) if c in valid or ctx.store.node(c)]
                plan["llm_plan"] = {**refined, "model": llm.label,
                                    "note": "LLM refinement of the deterministic plan above (inferred)"}
            except LLMError as exc:
                plan["llm_error"] = str(exc)
        return plan
    raise ValueError(f"unknown section {name}")


def _scope(section: dict[str, Any]) -> list[str]:
    text = json.dumps(section, default=str)
    import re
    ids = re.findall(r'"((?:symbol|module|api|resource|decision|pattern|file):[^"]{1,200})"', text)
    return list(dict.fromkeys(ids))[:200]


def _bounded(section: dict[str, Any], max_chars: int) -> dict[str, Any]:
    """Trim lists until the serialized facts fit the contract budget."""
    data = json.loads(json.dumps(section, default=str))
    while len(json.dumps(data)) > max_chars:
        longest = max((k for k, v in data.items() if isinstance(v, list) and len(v) > 1),
                      key=lambda k: len(json.dumps(data[k])), default=None)
        if longest is None:
            break
        data[longest] = data[longest][: max(1, len(data[longest]) // 2)]
        data.setdefault("_truncated", []).append(longest)
    return data
