"""A compact, factual orientation for the start of an agent session (SessionStart hooks, `promapkin brief`).

It states facts only: what the index knows, what changed since, which handoffs are open, and which
Promapkin tools exist. Hook output is read by the agent as context, so it stays short and never instructs.
"""

from __future__ import annotations

from typing import Any

from promapkin.config import Workspace

OPEN_STATUSES = ("open", "investigating", "investigation_complete", "planned", "in_progress", "blocked", "review")
TOOLS_LINE = ("Promapkin MCP tools in this session include build_task_context (budgeted context for a task), "
              "analyze_impact (blast radius and tests to run), trace_path, search_graph, get_architecture_map, "
              "create_handoff / resume_handoff, and verify_claims.")


def brief(ws: Workspace, sync: bool = False, max_chars: int = 4000) -> str:
    from promapkin.service import Service

    if not ws.initialized or not ws.store.get_meta("indexed_revision"):
        return ("Promapkin: this repository has no index yet (`promapkin scan` builds one; the MCP tool "
                "scan_repository does the same).")
    svc = Service(ws, agent="brief")
    if sync:
        try:
            svc.scan()
        except Exception as exc:  # noqa: BLE001 - a busy index or a failed scan must not break session start
            note = f"(index not refreshed: {str(exc)[:120]})"
        else:
            note = ""
    else:
        note = ""
    st = svc.status()
    ov = svc.overview()
    out: list[str] = []
    pend = st.get("pending_changes", {})
    changed = sum(pend.values()) if isinstance(pend, dict) else 0
    out.append(f"Promapkin index of {ov.get('repository')} at {str(st.get('indexed_revision'))[:12]}"
               f"{'' if st.get('index_fresh') else f'; {changed} file(s) changed since indexing'} {note}".rstrip())
    langs = sorted((ov.get("languages") or {}).items(), key=lambda kv: -kv[1].get("files", 0))
    from promapkin.extract import NON_CODE
    langs = [(k, v) for k, v in langs if k not in NON_CODE]
    if langs:
        out.append("Languages: " + ", ".join(f"{k} ({v['files']} files)" for k, v in langs[:8]))
    styles = [s["statement"] for s in (ov.get("styles") or [])[:4]]
    if styles:
        out.append("Architecture: " + "; ".join(styles))
    subs = (ov.get("subsystems") or [])[:12]
    if subs:
        out.append("Subsystems: " + ", ".join(f"{s['name']} [{s.get('layer') or '?'}]" for s in subs))
    entries = (ov.get("entry_points") or [])[:6]
    if entries:
        out.append("Entry points: " + "; ".join(_entry(e) for e in entries))
    infra = _infra(ws)
    if infra:
        out.append(infra)
    try:
        hubs = svc.hubs(6).get("hubs", [])
    except Exception:  # noqa: BLE001
        hubs = []
    if hubs:
        out.append("Most depended-on symbols: " + ", ".join(
            f"{h['id'].rsplit(':', 1)[-1]} ({h['fan_in']} inbound)" for h in hubs))
    violations = ov.get("layer_violations")   # a count in the overview, a list in the architecture section
    count = violations if isinstance(violations, int) else len(violations or [])
    if count:
        out.append(f"Layer violations recorded: {count}")
    open_handoffs = [h for h in svc.handoffs().list() if h.get("status") in OPEN_STATUSES][:3]
    for h in open_handoffs:
        out.append(f"Open handoff {h['id']}: {str(h.get('task', ''))[:140]} (next: {str(h.get('next_action', ''))[:120]})")
    constraints = [c for h in open_handoffs
                   for c in (svc.handoffs().get(h["id"]).get("data", {}).get("constraints") or [])][:5]
    if constraints:
        out.append("Recorded constraints: " + "; ".join(_text_of(c)[:120] for c in constraints))
    out.append(TOOLS_LINE)
    text = "\n".join(out)
    return text[:max_chars]


def _text_of(item: Any) -> str:
    if isinstance(item, dict):
        return str(next((item[k] for k in ("text", "constraint", "statement", "claim", "description") if item.get(k)),
                        item))
    return str(item)


def _entry(e: Any) -> str:
    if isinstance(e, dict):
        return str(e.get("statement") or e.get("node") or e)[:90]
    return str(e)[:90]


def _infra(ws: Workspace) -> str:
    rows = ws.store.conn.execute(
        "SELECT kind, COUNT(*) FROM nodes WHERE kind IN ('service','deployment','table','cloud_resource','queue',"
        "'schema_type','api_operation') GROUP BY kind").fetchall()
    if not rows:
        return ""
    return "Infrastructure and schemas: " + ", ".join(f"{n} {k.replace('_', ' ')}(s)" for k, n in rows)
