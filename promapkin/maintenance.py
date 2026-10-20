"""Continuous intelligence: sync (incremental re-analysis), verify (stale-claim detection) and status."""

from __future__ import annotations

import json
from typing import Any, Callable

from promapkin import blueprint, evidence as ev, knowledge
from promapkin.agents import orchestrator
from promapkin.config import Workspace
from promapkin.handoff import Handoffs
from promapkin.scanner import scan, stale_files


def sync(ws: Workspace, progress: Callable[[str], None] | None = None, regenerate: bool = True,
         interactive: bool = False, ask: Any = None) -> dict[str, Any]:
    """Incremental scan, then re-run only the specialists whose inputs changed."""
    result = scan(ws, progress=progress, interactive=interactive, ask=ask)
    changed = result["changed_files"] + result["deleted_files"]
    sections = orchestrator.plan_for_changes(changed)
    out: dict[str, Any] = {"scan": dict(result), "rerun_sections": sections, "written": []}
    if regenerate and sections and blueprint.load(ws) is not None:
        res = orchestrator.run(ws, sections=sections, progress=progress)
        out["written"] = blueprint.write(ws, res)
    return out


def verify(ws: Workspace) -> dict[str, Any]:
    s = ws.store
    report: dict[str, Any] = {"revision": ws.revision(), "indexed_revision": s.get_meta("indexed_revision")}
    st = stale_files(ws)
    report["index"] = {"up_to_date": not any(st.values()), **{k: v[:50] for k, v in st.items()}}

    # Agent/handoff knowledge: re-check citations against the current tree.
    rechecked, newly_stale = 0, []
    with s.transaction():
        for n in s.nodes(where="evidence IN (?,?,?,?) AND status IN ('verified','stale')",
                         params=(ev.AGENT_INFERENCE, ev.HANDOFF, ev.RUNTIME, ev.HEURISTIC)):
            cites = n["attrs"].get("citations")
            if not cites:
                continue
            rechecked += 1
            current = [knowledge.resolve_citation(s, ws.root, c["ref"]) for c in cites]
            if any(c["valid"] for c in cites) and not any(c["valid"] for c in current):
                s.set_node_status(n["id"], "stale")
                newly_stale.append({"id": n["id"], "name": n["name"],
                                    "reason": "every cited source disappeared"})
            elif any(c["file"] in st["modified"] + st["deleted"] for c in current if c.get("file")):
                if n["status"] != "stale":
                    s.set_node_status(n["id"], "stale")
                    newly_stale.append({"id": n["id"], "name": n["name"], "reason": "cited file changed"})
    report["knowledge"] = {"rechecked": rechecked, "newly_stale": newly_stale}

    from promapkin.agents import specialists as sp
    from promapkin.graph import Graph
    from promapkin import ml
    ctx = sp.Ctx(ws, s, Graph(s), ml.runtime(ws), ws.revision())
    v = sp.verification(ctx)
    report["documentation"] = {"summary": v["summary"], "contradictions": v["contradictions"],
                               "dangling_references": v["dangling_references"]}
    llm = getattr(ws, "llm", None)
    if llm is not None and v["documentation_claims"]:
        from promapkin.llm.errors import LLMError
        try:
            verdicts = llm.judge_claims([{"claim": c["claim"], "source": c["source"], "static_evidence": c["evidence"],
                                          "static_status": c["status"]} for c in v["documentation_claims"]])
            for vd in verdicts:
                i = vd.get("index")
                if isinstance(i, int) and 0 <= i < len(v["documentation_claims"]):
                    v["documentation_claims"][i]["llm_verdict"] = {"verdict": vd["verdict"],
                                                                   "explanation": vd["explanation"],
                                                                   "model": llm.label}
            report["documentation"]["llm_judged"] = v["documentation_claims"]
        except LLMError as exc:
            report["documentation"]["llm_error"] = str(exc)

    dangling = s.conn.execute(
        "SELECT COUNT(*) FROM edges WHERE src NOT IN (SELECT id FROM nodes) OR dst NOT IN (SELECT id FROM nodes)"
    ).fetchone()[0]
    report["integrity"] = {"edges_to_missing_nodes": dangling,
                           "sqlite_integrity": s.conn.execute("PRAGMA quick_check").fetchone()[0]}
    handoffs = []
    h = Handoffs(ws)
    for item in h.list():
        if item["status"] in ("complete", "abandoned", "superseded"):
            continue
        fr = h.freshness(h.get(item["id"])["data"])
        if fr["stale"]:
            handoffs.append({"id": item["id"], "task": item["task"],
                             "needs_revalidation": [f["claim"] for f in fr["findings"]
                                                    if f["current"] == "needs_revalidation"],
                             "changed_symbols": [x["symbol"] for x in fr["symbols"] if x["changed_since_handoff"]]})
    report["handoffs"] = handoffs
    issues = (0 if report["index"]["up_to_date"] else 1) + len(newly_stale) + \
        len(v["contradictions"]) + len(v["dangling_references"]) + (1 if dangling else 0) + len(handoffs)
    report["issues"] = issues
    report["ok"] = issues == 0
    return report


def status(ws: Workspace) -> dict[str, Any]:
    if not ws.initialized:
        return {"initialized": False, "root": str(ws.root), "hint": "run `promapkin init`"}
    s = ws.store
    st = stale_files(ws) if s.get_meta("indexed_revision") else {"modified": [], "added": [], "deleted": []}
    bp_rev = s.get_meta("blueprint_revision")
    ml_meta = json.loads(s.get_meta("ml") or "{}")
    from promapkin import ml
    return {
        "initialized": True, "root": str(ws.root), "revision": ws.revision(),
        "indexed_revision": s.get_meta("indexed_revision"),
        "index_fresh": not any(st.values()),
        "pending_changes": {k: len(v) for k, v in st.items()},
        "blueprint_revision": bp_rev,
        "blueprint_fresh": bool(bp_rev) and bp_rev == s.get_meta("indexed_revision"),
        "counts": s.counts(),
        "handoffs": {k: sum(1 for h in s.list_handoffs() if h["status"] == k)
                     for k in sorted({h["status"] for h in s.list_handoffs()})},
        "agent_tasks": {k: len(s.agent_tasks(k)) for k in ("pending", "claimed", "completed", "rejected")},
        "ml": ml_meta, "models": ml.manager(ws).status(),
        "provider": ws.config.provider, "last_jobs": s.last_jobs(3),
    }
