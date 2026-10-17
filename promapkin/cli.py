"""Command-line interface. Every command maps onto the same service the MCP server uses."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from promapkin import __version__, blueprint, maintenance
from promapkin.config import Config, Workspace
from promapkin.handoff import HandoffError, VersionConflict
from promapkin.knowledge import KnowledgeError
from promapkin.security import PathOutsideRepository
from promapkin.service import Service, ServiceError

SECTION_CMDS = {"architecture": "architecture", "dataflow": "dataflow", "api": "api", "decisions": "decisions",
                "patterns": "patterns", "risks": "risks", "verification": "verification",
                "recon": "reconnaissance", "roadmap": "roadmap"}


def _ask(question: str) -> bool:
    try:
        answer = input(f"{question} [y/N] ")
    except EOFError:
        return False
    return answer.strip().lower() in ("y", "yes")


def _interactive(args: argparse.Namespace) -> bool:
    return sys.stdin.isatty() and sys.stderr.isatty() and not getattr(args, "json", False)


def emit(args: argparse.Namespace, data: Any, human: Any = None) -> None:
    if args.json or human is None:
        print(json.dumps(data, indent=2, default=str))
    else:
        print(human(data) if callable(human) else human)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="promapkin", description="Agentic codebase intelligence: blueprints, an "
                                "evidence-aware code graph, task context and agent handoffs over CLI and MCP.")
    p.add_argument("--version", action="version", version=f"promapkin {__version__}")
    p.add_argument("--root", default=os.environ.get("PROMAPKIN_ROOT", "."), help="repository root (default: cwd)")
    p.add_argument("--json", action="store_true", help="machine-readable JSON output")
    p.add_argument("--agent", default=os.environ.get("PROMAPKIN_AGENT", "cli"), help="agent name for provenance")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init", help="create .promapkin/ in the repository")
    sc = sub.add_parser("scan", help="index the repository (incremental)")
    sc.add_argument("--full", action="store_true", help="re-extract every file")
    sub.add_parser("sync", help="incremental scan + re-run affected blueprint specialists")
    v = sub.add_parser("verify", help="detect stale knowledge, contradictions and integrity problems")
    v.add_argument("--strict", action="store_true", help="exit 1 when issues are found")
    sub.add_parser("status", help="index, blueprint, handoff and ML status")

    bp = sub.add_parser("blueprint", help="generate and read architectural blueprints")
    bps = bp.add_subparsers(dest="bp_cmd", required=True)
    g = bps.add_parser("generate", help="run all (or selected) specialists and write artifacts")
    g.add_argument("--sections", nargs="*")
    g.add_argument("--goal")
    for name in SECTION_CMDS:
        x = bps.add_parser(name, help=f"{SECTION_CMDS[name]} section")
        x.add_argument("--refresh", action="store_true")
        if name == "roadmap":
            x.add_argument("goal", nargs="?")

    gr = sub.add_parser("graph", help="explore the code graph")
    grs = gr.add_subparsers(dest="graph_cmd", required=True)
    q = grs.add_parser("query", help="hybrid search")
    q.add_argument("text")
    q.add_argument("--limit", type=int, default=15)
    q.add_argument("--kinds", nargs="*")
    n = grs.add_parser("node", help="node detail with relationships and evidence")
    n.add_argument("ref")
    nb = grs.add_parser("neighbors")
    nb.add_argument("ref")
    nb.add_argument("--direction", default="both", choices=["in", "out", "both"])
    nb.add_argument("--types", nargs="*")
    nb.add_argument("--depth", type=int, default=1)
    nb.add_argument("--min-trust", type=float, default=0.0)
    im = grs.add_parser("impact", help="change-impact analysis")
    im.add_argument("ref")
    im.add_argument("--depth", type=int, default=4)
    im.add_argument("--min-trust", type=float, default=0.15)
    tr = grs.add_parser("trace", help="most-trusted paths between two nodes")
    tr.add_argument("--from", dest="src", required=True)
    tr.add_argument("--to", dest="dst", required=True)
    tr.add_argument("-k", type=int, default=3)
    df = grs.add_parser("dataflow", help="data paths from a node to sinks")
    df.add_argument("ref")
    dp = grs.add_parser("deps", help="module dependencies and cycles")
    dp.add_argument("module", nargs="?")
    sm = grs.add_parser("similar", help="semantically similar nodes")
    sm.add_argument("ref", nargs="?")
    sm.add_argument("--text")
    src = grs.add_parser("source", help="show source for a node or path[:a-b]")
    src.add_argument("ref")
    dg = grs.add_parser("diagram", help="Mermaid diagram")
    dg.add_argument("kind", choices=["architecture", "dataflow", "patterns", "roadmap", "neighborhood",
                                     "dependencies", "map"])
    dg.add_argument("--focus")
    mp = grs.add_parser("map", help="create a named map from nodes")
    mp.add_argument("title")
    mp.add_argument("nodes", nargs="+")
    mp.add_argument("--description", default="")
    an = grs.add_parser("annotate", help="attach a concept to a node")
    an.add_argument("ref")
    an.add_argument("concept")
    rl = grs.add_parser("relate", help="add an agent relationship")
    rl.add_argument("source")
    rl.add_argument("type")
    rl.add_argument("target")
    rl.add_argument("--rationale", default="")
    fd = grs.add_parser("finding", help="record a finding with citations")
    fd.add_argument("claim")
    fd.add_argument("--cite", action="append", default=[])
    fd.add_argument("--confidence", type=float, default=0.6)

    cx = sub.add_parser("context", help="task-specific context")
    cxs = cx.add_subparsers(dest="ctx_cmd", required=True)
    cb = cxs.add_parser("build", help="token-budgeted context package")
    cb.add_argument("task")
    cb.add_argument("--budget", type=int)
    cb.add_argument("--constraint", action="append", default=[])
    cb.add_argument("--validation", action="append", default=[])
    cb.add_argument("--focus", action="append", default=[])
    cb.add_argument("--no-source", action="store_true")

    ho = sub.add_parser("handoff", help="durable agent handoffs")
    hos = ho.add_subparsers(dest="ho_cmd", required=True)
    hc = hos.add_parser("create", help="create from flags or --file (JSON)")
    _handoff_fields(hc)
    hc.add_argument("--parent")
    hu = hos.add_parser("update", help="append to a handoff")
    hu.add_argument("id")
    _handoff_fields(hu, task_required=False)
    hu.add_argument("--expected-version", type=int)
    hr = hos.add_parser("resume", help="reconstruct context for another agent")
    hr.add_argument("id")
    hr.add_argument("--budget", type=int)
    hr.add_argument("--claim", action="store_true")
    hl = hos.add_parser("list")
    hl.add_argument("--status")
    hs = hos.add_parser("show")
    hs.add_argument("id")
    hs.add_argument("--version", type=int)
    hb = hos.add_parser("branch")
    hb.add_argument("parent")
    hb.add_argument("task")

    ag = sub.add_parser("agent", help="external specialist agent tasks")
    ags = ag.add_subparsers(dest="agent_cmd", required=True)
    at = ags.add_parser("tasks", help="queue/claim analysis contracts")
    at.add_argument("--sections", nargs="*")
    at.add_argument("--goal")
    at.add_argument("--no-claim", action="store_true")
    asb = ags.add_parser("submit", help="submit a result JSON file for a task")
    asb.add_argument("task_id")
    asb.add_argument("file")

    mc = sub.add_parser("mcp", help="MCP server")
    mcs = mc.add_subparsers(dest="mcp_cmd", required=True)
    mcs.add_parser("serve", help="serve MCP over stdio")
    mcs.add_parser("inspect", help="list tools, prompts and resources")
    mcs.add_parser("config", help="print a client configuration snippet")

    md = sub.add_parser("models", help="ML/DL models: status, download, remove, feedback")
    mds = md.add_subparsers(dest="models_cmd", required=True)
    mds.add_parser("status")
    d = mds.add_parser("download", help="install/download a small embedding model (asks first)")
    d.add_argument("--backend", choices=["fastembed", "sentence-transformers"])
    d.add_argument("--yes", action="store_true")
    mds.add_parser("remove", help="delete downloaded models")
    fb = mds.add_parser("feedback", help="add a labelled training example")
    fb.add_argument("task")
    fb.add_argument("label")
    fb.add_argument("text")
    cl = mds.add_parser("classify", help="classify text with the two-tier classifier")
    cl.add_argument("task")
    cl.add_argument("text")

    cf = sub.add_parser("config", help="read or change .promapkin/config.json")
    cfs = cf.add_subparsers(dest="config_cmd", required=True)
    cfs.add_parser("show")
    cs = cfs.add_parser("set")
    cs.add_argument("key")
    cs.add_argument("value")

    be = sub.add_parser("bench", help="benchmarks")
    bes = be.add_subparsers(dest="bench_cmd", required=True)
    br = bes.add_parser("retrieval", help="retrieval quality and token benchmark")
    br.add_argument("--spec")
    br.add_argument("--budget", type=int, default=2000)
    br.add_argument("--limit", type=int, default=40)
    ba = bes.add_parser("agents", help="A/B: same agent with and without Promapkin")
    ba.add_argument("--spec", required=True)
    ba.add_argument("--budget", type=int, default=4000)
    ba.add_argument("--keep", action="store_true")
    bes.add_parser("example-spec", help="print an example agent benchmark spec")
    return p


def _handoff_fields(p: argparse.ArgumentParser, task_required: bool = True) -> None:
    p.add_argument("--file", help="JSON file with the handoff body")
    p.add_argument("--task", required=False)
    p.add_argument("--status")
    p.add_argument("--summary")
    p.add_argument("--finding", action="append", default=[], help="'claim' or 'claim|source|status|evidence'")
    p.add_argument("--constraint", action="append", default=[])
    p.add_argument("--symbol", action="append", default=[])
    p.add_argument("--dependency", action="append", default=[])
    p.add_argument("--decision", action="append", default=[], help="'decision|rationale'")
    p.add_argument("--completed", action="append", default=[])
    p.add_argument("--next", dest="next_action")
    p.add_argument("--validation", action="append", default=[])
    p.add_argument("--question", action="append", default=[])


def _handoff_body(a: argparse.Namespace) -> dict[str, Any]:
    body: dict[str, Any] = json.loads(Path(a.file).read_text()) if a.file else {}
    if a.task:
        body["task"] = a.task
    for k, attr in (("status", "status"), ("summary", "summary"), ("next_action", "next_action")):
        if getattr(a, attr):
            body[k] = getattr(a, attr)
    findings = []
    for f in a.finding:
        parts = f.split("|")
        findings.append({"claim": parts[0], **({"sources": [parts[1]]} if len(parts) > 1 and parts[1] else {}),
                         **({"status": parts[2]} if len(parts) > 2 else {}),
                         **({"evidence": parts[3]} if len(parts) > 3 else {})})
    if findings:
        body.setdefault("findings", []).extend(findings)
    for k, attr in (("constraints", "constraint"), ("completed", "completed"), ("validation", "validation"),
                    ("open_questions", "question")):
        if getattr(a, attr):
            body.setdefault(k, []).extend(getattr(a, attr))
    if a.decision:
        body.setdefault("decisions", []).extend(
            {"decision": d.split("|")[0], "rationale": d.split("|")[1] if "|" in d else "unknown"} for d in a.decision)
    if a.symbol or a.dependency:
        rc = body.setdefault("relevant_context", {})
        rc.setdefault("symbols", []).extend(a.symbol)
        rc.setdefault("dependencies", []).extend(a.dependency)
    return body


# ------------------------------------------------------------------------------- human formatting

def fmt_scan(r: dict[str, Any]) -> str:
    lines = [f"Indexed {r['files']} files at {r['revision'][:12]} ({r['changed']} changed, {r['deleted']} deleted) "
             f"in {r['seconds']}s",
             f"  references resolved: {r['resolved_refs']}  unresolved: {r['unresolved_refs']}  "
             f"disambiguated: {r.get('disambiguated', 0)}  similarity edges: {r.get('similar_edges', 0)}",
             f"  ML: {r['ml']['tier']} · {r['ml']['embedder']} · {r['ml']['reason']}"]
    if r.get("stale_inferences"):
        lines.append(f"  {r['stale_inferences']} agent findings marked stale (their sources changed)")
    if r.get("errors"):
        lines.append(f"  {len(r['errors'])} files with extraction errors (see --json)")
    return "\n".join(lines)


def fmt_search(r: dict[str, Any]) -> str:
    lines = []
    for h in r["results"]:
        lines.append(f"{h['score']:.3f}  {h['kind']:<12} {h['id']}  {h.get('location') or ''}")
        for rel in h.get("relationships", [])[:4]:
            lines.append(f"         · {rel}")
    return "\n".join(lines) or "no results"


def fmt_impact(r: dict[str, Any]) -> str:
    lines = [f"Impact of changing {r['target']['id']} ({r['total_affected']} affected)"]
    for group, items in r["affected"].items():
        lines.append(f"\n{group}:")
        for it in items[:15]:
            via = it.get("via") or {}
            lines.append(f"  {it['impact_score']:.2f}  {it['id']}  ← {via.get('type', '')} [{via.get('evidence', '')}]")
    if r["tests_to_run"]:
        lines.append("\ntests to run:")
        lines += [f"  {t['id']}  {t.get('location', '')}" for t in r["tests_to_run"]]
    return "\n".join(lines)


def fmt_trace(r: dict[str, Any]) -> str:
    if not r["paths"]:
        return f"no path from {r['from']['id']} to {r['to']['id']}"
    lines = []
    for p in r["paths"]:
        lines.append(f"path trust {p['trust']:.3f} ({p['basis']}):")
        for st in p["steps"]:
            lines.append(f"  {st['source']} -{st['type']}-> {st['target']}  [{st['evidence']}] {st.get('at', '')}")
    return "\n".join(lines)


def fmt_flows(r: dict[str, Any]) -> str:
    lines = [f"Flows from {r['from']['id']}:"]
    for f in r["flows"]:
        lines.append(f"  {f['operation']:<7} {f['sink']:<22} [{f['basis']}, trust {f['trust']:.2f}]  " +
                     " → ".join(n.split(":")[-1] for n in f["nodes"]))
    return "\n".join(lines)


def fmt_status(r: dict[str, Any]) -> str:
    if not r.get("initialized"):
        return f"Not initialized at {r['root']}: {r['hint']}"
    pc = r["pending_changes"]
    nodes = sum(r["counts"]["nodes"].values())
    edges = sum(r["counts"]["edges"].values())
    return "\n".join([
        f"Repository   {r['root']}",
        f"Revision     {r['revision']}",
        f"Indexed      {r['indexed_revision']}  ({'fresh' if r['index_fresh'] else 'STALE: ' + str(pc)})",
        f"Blueprint    {r['blueprint_revision'] or 'none'}  ({'fresh' if r['blueprint_fresh'] else 'needs regeneration'})",
        f"Graph        {r['counts']['files']} files, {nodes} nodes, {edges} edges",
        f"Handoffs     {r['handoffs'] or 'none'}",
        f"Agent tasks  {r['agent_tasks']}",
        f"ML           {r['ml'].get('tier', '?')} · {r['ml'].get('embedder', '?')} · mode {r['models']['mode']}",
        f"Provider     {r['provider']}",
    ])


def fmt_verify(r: dict[str, Any]) -> str:
    lines = [f"{'OK' if r['ok'] else str(r['issues']) + ' issue(s)'} at {r['revision']}"]
    if not r["index"]["up_to_date"]:
        lines.append(f"  index stale: {len(r['index']['modified'])} modified, {len(r['index']['added'])} added, "
                     f"{len(r['index']['deleted'])} deleted — run `promapkin sync`")
    for n in r["knowledge"]["newly_stale"]:
        lines.append(f"  stale knowledge: {n['id']} ({n['reason']})")
    for c in r["documentation"]["contradictions"]:
        lines.append(f"  doc claim not supported by code: {c['claim']} ({c['source']})")
    for d in r["documentation"]["dangling_references"]:
        lines.append(f"  {d['statement']}")
    for h in r["handoffs"]:
        lines.append(f"  handoff {h['id']} needs revalidation: {h['needs_revalidation'] or h['changed_symbols']}")
    if r["integrity"]["edges_to_missing_nodes"]:
        lines.append(f"  {r['integrity']['edges_to_missing_nodes']} edges point to missing nodes")
    return "\n".join(lines)


def fmt_handoff(r: dict[str, Any]) -> str:
    h = r.get("handoff") or r.get("data") or {}
    lines = [f"Handoff {r.get('id', h.get('id'))} v{r.get('version')} — {h.get('task')} [{h.get('status')}]"]
    if r.get("conflicts"):
        lines.append(f"  CONFLICTS: {r['conflicts']}")
    if h.get("next_action"):
        lines.append(f"  next: {h['next_action']}")
    for f in h.get("findings", []):
        lines.append(f"  finding [{f['status']}]: {f['claim']} {f.get('sources') or ''}")
    for c in h.get("constraints", []):
        lines.append(f"  constraint: {c}")
    if r.get("graph"):
        lines.append(f"  graph: linked {len(r['graph']['linked_symbols'])} symbols; unresolved "
                     f"{r['graph']['unresolved_symbols']}")
    fr = r.get("freshness")
    if fr:
        lines.append(f"  freshness: revision {'changed' if fr['revision_changed'] else 'unchanged'}; "
                     f"{'STALE items present' if fr['stale'] else 'no stale items'}")
        for f in fr["findings"]:
            if f["current"] != f["recorded_status"]:
                lines.append(f"    revalidate: {f['claim']}")
    if r.get("context"):
        from promapkin.retrieval import render_context
        lines += ["", render_context(r["context"])]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ws = Workspace(Path(args.root))
    try:
        return _run(args, ws)
    except VersionConflict as exc:
        print(f"conflict: {exc}", file=sys.stderr)
        return 3
    except (ServiceError, KnowledgeError, HandoffError, PathOutsideRepository, LookupError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    finally:
        ws.close()


def _run(args: argparse.Namespace, ws: Workspace) -> int:
    cmd = args.cmd
    if cmd == "mcp" and args.mcp_cmd == "serve":
        from promapkin import mcp_server
        if not ws.initialized:
            ws.init()
        mcp_server.serve(ws)
        return 0
    if cmd == "mcp" and args.mcp_cmd == "inspect":
        from promapkin import mcp_server
        info = mcp_server.inspect()
        emit(args, info, lambda d: "\n".join([f"promapkin MCP {d['server']['version']} — {len(d['tools'])} tools"] +
                                             [f"  {'R ' if t['read_only'] else 'W '}{t['name']:<32} "
                                              f"{t['description'][:90]}" for t in d["tools"]] +
                                             [f"prompts: {', '.join(d['prompts'])}",
                                              f"resources: {', '.join(d['resources'])}"]))
        return 0
    if cmd == "mcp" and args.mcp_cmd == "config":
        cfg = {"mcpServers": {"promapkin": {"command": sys.executable,
                                            "args": ["-m", "promapkin", "--root", str(ws.root), "mcp", "serve"]}}}
        print(json.dumps(cfg, indent=2))
        return 0
    if cmd == "init":
        created = ws.init()
        print(f"{'Initialized' if created else 'Already initialized'} Promapkin in {ws.dir}")
        return 0
    if cmd == "config":
        if not ws.initialized:
            ws.init()
        if args.config_cmd == "show":
            print(ws.config_path.read_text())
            return 0
        field_type = Config.__dataclass_fields__.get(args.key)
        if field_type is None:
            print(f"unknown key {args.key}; keys: {', '.join(Config.__dataclass_fields__)}", file=sys.stderr)
            return 2
        cur = getattr(ws.config, args.key)
        value: Any = args.value
        if isinstance(cur, bool):
            value = args.value.lower() in ("1", "true", "yes", "on")
        elif isinstance(cur, int):
            value = int(args.value)
        elif isinstance(cur, list):
            value = [x for x in args.value.split(",") if x]
        elif isinstance(cur, dict):
            value = json.loads(args.value)
        setattr(ws.config, args.key, value)
        ws.config.save(ws.config_path)
        print(f"{args.key} = {json.dumps(value)}")
        return 0
    if cmd == "models":
        return _models(args, ws)

    svc = Service(ws, agent=args.agent)
    if cmd == "scan":
        from promapkin.scanner import scan
        r = scan(ws, full=args.full, interactive=_interactive(args), ask=_ask,
                 progress=None if args.json else (lambda m: print(m, file=sys.stderr)))
        emit(args, dict(r), fmt_scan)
    elif cmd == "sync":
        r = maintenance.sync(ws, interactive=_interactive(args), ask=_ask)
        emit(args, r, lambda d: fmt_scan(d["scan"]) + (f"\nre-ran: {', '.join(d['rerun_sections'])}"
                                                       if d["rerun_sections"] else "\nno specialists affected") +
             "".join(f"\n  wrote {w}" for w in d["written"]))
    elif cmd == "verify":
        r = svc.verify()
        emit(args, r, fmt_verify)
        return 1 if args.strict and not r["ok"] else 0
    elif cmd == "status":
        emit(args, maintenance.status(ws), fmt_status)
    elif cmd == "blueprint":
        if args.bp_cmd == "generate":
            r = svc.generate_blueprint(args.sections, args.goal)
            emit(args, r, lambda d: "\n".join([f"Blueprint at {d['revision'][:12]} (provider {d['provider']}, "
                                               f"ML {d['ml']['tier']})"] + [f"  wrote {w}" for w in d["written"]] +
                                              ([f"  queued agent tasks: {d['queued_agent_tasks']}"]
                                               if d["queued_agent_tasks"] else [])))
        else:
            section = SECTION_CMDS[args.bp_cmd]
            if section == "roadmap":
                data = svc.plan(args.goal) if args.goal else svc.get_blueprint("roadmap", refresh=True)["data"]
                rev = ws.store.get_meta("indexed_revision") or ""
                emit(args, data, lambda d: blueprint.render("roadmap", d, rev))
            else:
                r = svc.get_blueprint(section, "json", refresh=args.refresh)
                emit(args, r, lambda d: blueprint.render(section, d["data"], d["revision"] or ""))
    elif cmd == "graph":
        return _graph(args, svc)
    elif cmd == "context":
        r = svc.context(args.task, args.budget, args.constraint or None, args.validation or None,
                        args.focus or None, not args.no_source)
        from promapkin.retrieval import render_context
        emit(args, r, render_context)
    elif cmd == "handoff":
        return _handoff(args, svc)
    elif cmd == "agent":
        if args.agent_cmd == "tasks":
            r = svc.get_analysis_tasks(args.sections, args.goal, claim=not args.no_claim)
        else:
            r = svc.submit_analysis(args.task_id, json.loads(Path(args.file).read_text()))
        emit(args, r)
    elif cmd == "bench":
        from promapkin import bench
        if args.bench_cmd == "example-spec":
            print(json.dumps(bench.AGENT_SPEC_EXAMPLE, indent=2))
        elif args.bench_cmd == "retrieval":
            spec = json.loads(Path(args.spec).read_text()) if args.spec else None
            r = svc.benchmark(spec, args.budget, args.limit)
            emit(args, r, _fmt_bench)
        else:
            spec = json.loads(Path(args.spec).read_text())
            r = bench.agent_benchmark(ws, spec, budget=args.budget, keep=args.keep)
            emit(args, r)
    return 0


def _fmt_bench(r: dict[str, Any]) -> str:
    lines = [f"Retrieval benchmark: {r['tasks']} tasks ({r['task_source']}), budget {r['budget']} tokens, "
             f"repository ≈ {r['repository_tokens']} tokens", f"setup: {r['setup_overhead']}"]
    for label, res in r["results"].items():
        lines.append(f"\n[{label} · {res['mode']}] evaluated {res['evaluated']}")
        for k in ("recall@5", "recall@10", "mrr", "context_hit_rate", "mean_context_tokens",
                  "mean_naive_exploration_tokens", "token_reduction_vs_naive", "token_reduction_vs_whole_repo",
                  "mean_search_ms", "mean_context_ms"):
            lines.append(f"  {k:<32} {res[k]}")
    return "\n".join(lines)


def _graph(args: argparse.Namespace, svc: Service) -> int:
    c = args.graph_cmd
    if c == "query":
        emit(args, svc.search(args.text, args.limit, args.kinds), fmt_search)
    elif c == "node":
        emit(args, svc.node(args.ref))
    elif c == "neighbors":
        emit(args, svc.neighbors(args.ref, args.direction, args.types, args.depth, args.min_trust))
    elif c == "impact":
        emit(args, svc.impact(args.ref, args.depth, args.min_trust), fmt_impact)
    elif c == "trace":
        emit(args, svc.trace(args.src, args.dst, k=args.k), fmt_trace)
    elif c == "dataflow":
        emit(args, svc.dataflow(args.ref), fmt_flows)
    elif c == "deps":
        emit(args, svc.dependencies(args.module))
    elif c == "similar":
        emit(args, svc.similar(args.ref, args.text))
    elif c == "source":
        emit(args, svc.source(args.ref), lambda d: f"{d['file']}:{d['start']}-{d['end']}\n{d['code']}")
    elif c == "diagram":
        emit(args, svc.diagram(args.kind, args.focus), lambda d: d["source"])
    elif c == "map":
        emit(args, svc.create_map(args.title, args.nodes, args.description), lambda d: d["mermaid"])
    elif c == "annotate":
        emit(args, svc.annotate(args.ref, args.concept))
    elif c == "relate":
        emit(args, svc.add_relationship(args.source, args.target, args.type, args.rationale))
    elif c == "finding":
        emit(args, svc.add_node("finding", args.claim, args.claim, args.cite, confidence=args.confidence))
    return 0


def _handoff(args: argparse.Namespace, svc: Service) -> int:
    c = args.ho_cmd
    if c == "create":
        emit(args, svc.create_handoff(_handoff_body(args), args.parent), fmt_handoff)
    elif c == "update":
        emit(args, svc.update_handoff(args.id, _handoff_body(args), args.expected_version), fmt_handoff)
    elif c == "resume":
        emit(args, svc.resume_handoff(args.id, args.budget, args.claim), fmt_handoff)
    elif c == "list":
        emit(args, svc.handoffs().list(args.status), lambda d: "\n".join(
            f"{h['id']}  v{h['version']}  [{h['status']}]  {h['task']}  (agent {h['agent']})" for h in d) or
            "no handoffs")
    elif c == "show":
        r = svc.handoffs().get(args.id, args.version)
        emit(args, r, lambda d: fmt_handoff(d))
    elif c == "branch":
        emit(args, svc.handoffs().branch(args.parent, args.task, agent=svc.agent), fmt_handoff)
    return 0


def _models(args: argparse.Namespace, ws: Workspace) -> int:
    from promapkin import ml
    mgr = ml.manager(ws)
    c = args.models_cmd
    if c == "status":
        emit(args, mgr.status())
    elif c == "download":
        if not args.yes and not _ask("Download a small embedding model (and install its Python package if "
                                     "missing)? Only model weights are downloaded."):
            print("cancelled; Promapkin keeps using heuristic + built-in classification")
            if ws.initialized:
                ws.config.ml_consent = "declined"
                ws.config.save(ws.config_path)
            return 0
        ok, msg = mgr.download(args.backend, install_package=True)
        if ok and ws.initialized:
            ws.config.ml_consent = "granted"
            ws.config.save(ws.config_path)
        print(msg)
        return 0 if ok else 1
    elif c == "remove":
        print("removed: " + (", ".join(mgr.remove()) or "nothing"))
    elif c == "feedback":
        emit(args, Service(ws).classification_feedback(args.task, args.text, args.label))
    elif c == "classify":
        emit(args, Service(ws).classify(args.task, args.text))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
