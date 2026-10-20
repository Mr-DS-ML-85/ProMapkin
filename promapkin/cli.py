"""Command-line interface. Every command maps onto the same service the MCP server uses."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any

from promapkin import __version__, blueprint, maintenance
from promapkin.config import Config, Workspace
from promapkin.handoff import HandoffError, VersionConflict
from promapkin.knowledge import KnowledgeError
from promapkin.security import PathOutsideRepository
from promapkin.service import Service, ServiceError
from promapkin.store import BUSY_HINT, StoreBusy, is_locked_error

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


EXPLAIN = {"graph", "context", "verify", "blueprint", "handoff", "status", "bench", "review"}


def emit(args: argparse.Namespace, data: Any, human: Any = None) -> None:
    engine = getattr(args, "_engine", None)
    explanation = None
    if engine is not None and not args.no_explain and args.cmd in EXPLAIN and not getattr(args, "_explained", False):
        from promapkin.llm.errors import LLMError
        try:
            explanation = engine.explain(_command_name(args), data)
        except LLMError as exc:
            explanation = f"(LLM explanation unavailable: {exc})"
        args._explained = True
    if args.json or human is None:
        if explanation is not None and isinstance(data, dict):
            data = {**data, "llm_explanation": explanation}
        print(json.dumps(data, indent=2, default=str))
    else:
        print(human(data) if callable(human) else human)
        if explanation:
            print(f"\n── {engine.label} ──\n{explanation.strip()}")


def _command_name(args: argparse.Namespace) -> str:
    parts = [args.cmd] + [str(getattr(args, a)) for a in ("bp_cmd", "graph_cmd", "ctx_cmd", "ho_cmd")
                          if getattr(args, a, None)]
    return "promapkin " + " ".join(parts)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="promapkin", description="Agentic codebase intelligence: blueprints, an "
                                "evidence-aware code graph, task context and agent handoffs over CLI and MCP.")
    p.add_argument("--version", action="version", version=f"promapkin {__version__}")
    p.add_argument("--root", default=os.environ.get("PROMAPKIN_ROOT", "."), help="repository root (default: cwd)")
    p.add_argument("--json", action="store_true", help="machine-readable JSON output")
    p.add_argument("--agent", default=os.environ.get("PROMAPKIN_AGENT", "cli"), help="agent name for provenance")
    llm = p.add_argument_group("LLM mode", "Use an LLM instead of regex/heuristic analysis in every command "
                                            "(sends repository excerpts to the provider; see docs/LLM.md)")
    llm.add_argument("--llm", nargs="?", const="default", default=os.environ.get("PROMAPKIN_LLM"),
                     choices=["default", "anthropic", "openai"], help="enable LLM mode (provider optional)")
    llm.add_argument("--model", help="model id (default from config: anthropic_model / openai_model)")
    llm.add_argument("--base-url", help="API base URL (OpenAI-compatible endpoints, gateways, proxies)")
    llm.add_argument("--api-key", help="API key for this run (prefer env vars or `promapkin auth set`)")
    llm.add_argument("--no-explain", action="store_true", help="in LLM mode, skip the narrative explanation")
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
    tr.add_argument("--mode", default="auto", choices=["auto", "flow", "structural", "related"])
    hb = grs.add_parser("hubs", help="most-connected symbols (god nodes)")
    hb.add_argument("--limit", type=int, default=25)
    grs.add_parser("communities", help="module communities")
    sp = grs.add_parser("surprises", help="rare observed links between communities")
    sp.add_argument("--limit", type=int, default=20)
    hx = grs.add_parser("html", help="write an interactive HTML graph explorer")
    hx.add_argument("--out")
    hx.add_argument("--max-nodes", type=int, default=1500)
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
    grs.add_parser("infra", help="deployables, routing, cloud resources, pipelines, config surface")
    ga = grs.add_parser("assets", help="PDFs, Office files, images and videos with extracted text/metadata")
    ga.add_argument("--media", choices=["pdf", "office", "image", "video"])
    grs.add_parser("schema", help="tables, keys, ORM mappings, readers/writers, API schemas and spec coverage")
    dg.add_argument("kind", choices=["infrastructure", "er", "architecture", "dataflow", "patterns", "roadmap",
                                     "neighborhood",
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

    ak = sub.add_parser("ask", help="ask a question; an LLM answers using Promapkin's tools (implies --llm)")
    ak.add_argument("question")
    ak.add_argument("--allow-writes", action="store_true", help="let the LLM record findings/maps/handoffs")
    ak.add_argument("--max-steps", type=int, default=12)
    rv = sub.add_parser("review", help="review the working tree (or a diff against --base) using the graph")
    rv.add_argument("--base", help="git ref to diff against (default: uncommitted changes)")

    au = sub.add_parser("auth", help="LLM provider credentials (stored outside the repository, mode 0600)")
    aus = au.add_subparsers(dest="auth_cmd", required=True)
    a1 = aus.add_parser("set", help="store an API key (read from stdin or prompted)")
    a1.add_argument("provider", choices=["anthropic", "openai"])
    a1.add_argument("--base-url", dest="auth_base_url", help="remember a base URL for this provider")
    a2 = aus.add_parser("remove")
    a2.add_argument("provider", choices=["anthropic", "openai"])
    aus.add_parser("status")

    ins = sub.add_parser("install", help="wire Promapkin into a coding assistant: MCP server, instructions, hooks, "
                                       "commands (merge-safe; see `install --list`)")
    ins.add_argument("--platform", "-p", action="append", default=[],
                     help="assistant to set up (repeatable): claude-code, codex, gemini, opencode, vscode, cursor, "
                          "windsurf, cline, roo, kiro, zed, continue, amazonq, junie, goose, crush, augment, trae, "
                          "agents-md")
    ins.add_argument("--detected", action="store_true", help="every assistant this repository already shows signs of")
    ins.add_argument("--list", action="store_true", help="list platforms and what is detected here")
    ins.add_argument("--dry-run", action="store_true", help="show what would change, write nothing")
    ins.add_argument("--portable", action="store_true",
                     help="write `promapkin` as the command (needs it on PATH) instead of this interpreter's path")
    uns = sub.add_parser("uninstall", help="remove what `promapkin install` added (all, or --platform ...)")
    uns.add_argument("--platform", "-p", action="append", default=[])
    bf = sub.add_parser("brief", help="compact factual orientation for an agent session (used by SessionStart hooks)")
    bf.add_argument("--sync", action="store_true", help="incrementally update the index first (if one exists)")
    bf.add_argument("--max-chars", type=int, default=4000)

    pa = sub.add_parser("parsers", help="optional real parsers: TypeScript compiler and tree-sitter grammars")
    pas = pa.add_subparsers(dest="parsers_cmd", required=True)
    pas.add_parser("status")
    pi = pas.add_parser("install", help="install parsers into Promapkin's cache (asks first)")
    pi.add_argument("which", nargs="?", default="all", choices=["all", "typescript", "tree-sitter"])
    pi.add_argument("--yes", action="store_true")

    tr = sub.add_parser("trace", help="runtime evidence: run YOUR command with tracing, then ingest it")
    trs = tr.add_subparsers(dest="trace_cmd", required=True)
    t1 = trs.add_parser("run", help="e.g. promapkin trace run -- python -m pytest -q")
    t1.add_argument("--kind", default="auto", choices=["auto", "python", "node"])
    t1.add_argument("command", nargs=argparse.REMAINDER)
    t2 = trs.add_parser("ingest", help="ingest an existing trace directory")
    t2.add_argument("directory")

    sub.add_parser("doctor", help="check the environment: git, parsers, ML models, LLM credentials, index")
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
    p.add_argument("--notes", help="free-form notes (or @file); in --llm mode the LLM structures them")


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
        lines.append(f"path trust {p['trust']:.3f} ({p['basis']}, {p.get('mode', 'flow')} mode): {p.get('summary', '')}")
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


def _attach_llm(args: argparse.Namespace, ws: Workspace) -> None:
    needs = args.cmd in ("ask",)
    if not args.llm and not needs:
        return
    from promapkin import llm
    if not ws.initialized:
        ws.init()
    engine = llm.make_engine(ws, provider=None if args.llm in (None, "default") else args.llm, model=args.model,
                             base_url=args.base_url, api_key=args.api_key)
    ws.llm = engine  # type: ignore[attr-defined]
    args._engine = engine


def _usage_line(args: argparse.Namespace) -> None:
    engine = getattr(args, "_engine", None)
    if engine is None or args.json:
        return
    d = engine.describe()
    print(f"[{engine.label}] {d['calls']} call(s), {d['cached']} cached · {d['input_tokens']:,} in / "
          f"{d['output_tokens']:,} out tokens · {d['fallbacks']} fallback(s) · {d['redactions']} secret(s) "
          f"redacted · {d['dropped_ungrounded']} ungrounded fact(s) dropped", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ws = Workspace(Path(args.root))
    from promapkin.llm.errors import LLMError, LLMUnavailable
    try:
        _attach_llm(args, ws)
        return _run(args, ws)
    except LLMUnavailable as exc:
        print(f"error: LLM mode unavailable: {exc}\n  set the API key (e.g. `promapkin auth set anthropic`) or "
              f"install the SDKs: pip install 'promapkin[llm]'", file=sys.stderr)
        return 2
    except LLMError as exc:
        print(f"error: LLM request failed: {exc}", file=sys.stderr)
        return 2
    except VersionConflict as exc:
        print(f"conflict: {exc}", file=sys.stderr)
        return 3
    except StoreBusy as exc:
        print(f"error: index busy: {exc}", file=sys.stderr)
        return 3
    except sqlite3.OperationalError as exc:
        if not is_locked_error(exc):
            raise
        print(f"error: index busy: {BUSY_HINT.format(path=ws.db_path, timeout=ws.config.db_busy_timeout)}",
              file=sys.stderr)
        return 3
    except (ServiceError, KnowledgeError, HandoffError, PathOutsideRepository, LookupError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted; the index is unchanged (scans are transactional)", file=sys.stderr)
        return 130
    finally:
        _usage_line(args)
        ws.close()


def _fmt_infra(d: dict[str, Any]) -> str:
    def short(x: str) -> str:
        return x.split(":", 1)[-1] if not x.startswith("symbol:") else x.rsplit(":", 1)[-1]
    out = []
    for i in d["ingress"]:
        out.append(f"ingress {', '.join(h for h in i['hosts'] if h) or i['id']} -> {', '.join(map(short, i['routes_to']))}")
    for x in d["deployables"]:
        parts = [f"{x['kind']:<15} {x['name']:<24}"]
        for key, label in (("routes_to", "routes to"), ("runs_code", "runs"), ("builds", "built from"),
                           ("depends_on", "depends on")):
            if x[key]:
                parts.append(f"{label} {', '.join(map(short, x[key]))}")
        if x["config"]:
            parts.append(f"config {', '.join(x['config'][:6])}")
        out.append("  ".join(parts) + f"  ({x['location']})")
    for cat, items in d["cloud_resources"].items():
        out.append(f"{cat}: " + ", ".join(f"{i['type']}.{i['id'].rsplit('.', 1)[-1]}" for i in items))
    for p in d["pipelines"]:
        out.append(f"pipeline {p['name']} [{', '.join(p['triggers'])}] jobs {', '.join(p['jobs'])}"
                   + (f" runs {', '.join(map(short, p['runs_code']))}" if p["runs_code"] else ""))
    cs = d["configuration"]
    if cs["read_but_never_declared"]:
        out.append("config read by code, declared nowhere: " + ", ".join(cs["read_but_never_declared"]))
    if cs["declared_but_never_read"]:
        out.append("config declared, never read by code: " + ", ".join(cs["declared_but_never_read"][:20]))
    return "\n".join(out) or "no infrastructure files indexed (Dockerfile, compose, Kubernetes, Terraform, CI, OpenAPI)"


def _fmt_schema(d: dict[str, Any]) -> str:
    out = []
    for t in d["tables"]:
        cols = ", ".join(c["name"] + ("*" if c["name"] in t["primary_key"] else "") for c in t["columns"][:10])
        refs = "; refs " + ", ".join(r["table"] for r in t["references"]) if t["references"] else ""
        acc = f"; read by {len(t['read_by'])}, written by {len(t['written_by'])}" if t["read_by"] or t["written_by"] \
            else ""
        orm = "; ORM " + ", ".join(c.rsplit(":", 1)[-1] for c in t["orm_classes"]) if t["orm_classes"] else ""
        out.append(f"{'view ' if t['view'] else 'table'} {t['name']} ({cols}){refs}{orm}{acc}  [{t['defined_by']}]")
    for s in d["schema_types"][:40]:
        out.append(f"type  {s['name']} [{s['format']}] {', '.join(map(str, s['fields'][:8]))}")
    if d["endpoints_by_protocol"]:
        out.append("endpoints: " + ", ".join(f"{k} {v}" for k, v in d["endpoints_by_protocol"].items()))
    spec = d["api_spec"]
    if spec["operations"]:
        out.append(f"API spec: {spec['operations']} operations; not implemented: {', '.join(spec['unimplemented']) or 'none'}"
                   f"; missing from spec: {', '.join(spec['undocumented']) or 'none'}")
    return "\n".join(out) or "no tables or schemas indexed"


def _install(args: argparse.Namespace, ws: Workspace) -> int:
    from promapkin import installer
    detected = installer.detect(ws.root)
    if args.list or not (args.platform or args.detected):
        rows = [{"platform": p.name, "title": p.title, "format_confidence": p.verified, "mcp_scope": p.scope,
                 "detected": p.name in detected} for p in installer.PLATFORMS.values()]
        emit(args, {"platforms": rows, "detected": detected},
             lambda d: "\n".join([f"{'*' if r['detected'] else ' '} {r['platform']:<12} {r['title']:<32} "
                                  f"format: {r['format_confidence']:<6} mcp: {r['mcp_scope']}" for r in d["platforms"]] +
                                 ["", "* = detected in this repository/machine",
                                  "usage: promapkin install --platform claude-code [--platform cursor ...] "
                                  "| --detected [--dry-run]"]))
        return 0
    platforms = list(dict.fromkeys(args.platform + (detected if args.detected else [])))
    if not platforms:
        print("no assistants detected; name one with --platform", file=sys.stderr)
        return 2
    if not ws.initialized:
        ws.init()
    r = installer.install(ws.root, platforms, dry_run=args.dry_run, portable=args.portable)

    def human(d: dict[str, Any]) -> str:
        lines = [f"{'Would set up' if d['dry_run'] else 'Set up'}: " +
                 ", ".join(f"{p['title']} (format confidence: {p['verified']})" for p in d["platforms"]),
                 f"MCP command: {d['command']} --root <repo> mcp serve"]
        lines += [f"  {c['status']:<8} {c['path']}" + (f"  [{c['event']} hook]" if c.get("event") else "")
                  for c in d["changes"]]
        if d["unchanged"]:
            lines.append(f"  ({d['unchanged']} already up to date)")
        lines += [f"note: {n}" for n in d["notes"]]
        if not d["dry_run"]:
            lines.append("Undo with: promapkin uninstall" + "".join(f" -p {p['platform']}" for p in d["platforms"]))
        return "\n".join(lines)
    emit(args, r, human)
    return 0


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
    if cmd == "install":
        return _install(args, ws)
    if cmd == "uninstall":
        from promapkin import installer
        r = installer.uninstall(ws.root, args.platform or None)
        emit(args, r, lambda d: "\n".join([f"{x['action']}: {x['path']}" for x in d["removed"]] +
                                          [f"note: {n}" for n in d["notes"]]) or "nothing to remove")
        return 0
    if cmd == "brief":
        from promapkin.brief import brief
        text = brief(ws, sync=args.sync, max_chars=args.max_chars)
        emit(args, {"brief": text}, lambda d: d["brief"])
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
    if cmd == "auth":
        return _auth(args)
    if cmd == "parsers":
        return _parsers(args, ws)

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
    elif cmd == "ask":
        from promapkin.llm.agent import ask
        r = ask(ws, args._engine, args.question, allow_writes=args.allow_writes, max_steps=args.max_steps,
                on_step=None if args.json else (lambda m: print(m, file=sys.stderr)))
        emit(args, r, lambda d: d["answer"])
    elif cmd == "review":
        from promapkin.llm.agent import review
        r = review(ws, getattr(args, "_engine", None), args.base)
        emit(args, r, _fmt_review)
    elif cmd == "trace":
        from promapkin import runtime
        if args.trace_cmd == "run":
            command = args.command[1:] if args.command[:1] == ["--"] else args.command
            r = runtime.run(ws, command, kind=args.kind)
        else:
            r = runtime.ingest(ws, Path(args.directory))
        emit(args, r, lambda d: (f"trace {d['trace']}: {d['runtime_edges']} runtime call edges "
                                 f"({d['dynamic_link_count']} not visible statically), {d['test_links']} "
                                 f"test→code links, {d['executed_nodes']} executed symbols") +
             "".join(f"\n  dynamic: {x['source']} -> {x['target']}" for x in d["dynamic_links"][:15]))
    elif cmd == "doctor":
        r = _doctor(ws)
        emit(args, r, lambda d: "\n".join(f"{'ok ' if c['ok'] else '!! '} {c['check']:<26} {c['detail']}"
                                           for c in d["checks"]))
        return 0 if all(c["ok"] or c.get("optional") for c in r["checks"]) else 1
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


def _fmt_review(r: dict[str, Any]) -> str:
    lines = [f"Review of {r['base']}: {len(r['files_changed'])} file(s), {len(r['changed_symbols'])} changed symbol(s)"]
    for i in r["impact"][:15]:
        lines.append(f"  {i['symbol']} ({i['location']}) → {i['total_affected']} affected")
    if r["tests_to_run"]:
        lines.append("tests to run:")
        lines += [f"  {t['id']}  {t['location'] or ''}" for t in r["tests_to_run"][:25]]
    if r["decisions_to_respect"]:
        lines.append("decisions to respect:")
        lines += [f"  {d['id']}: {d['summary']}" for d in r["decisions_to_respect"]]
    lr = r.get("llm_review")
    if lr:
        lines += ["", f"LLM review: {lr.get('summary', '')}"]
        for f in lr.get("findings", []):
            lines.append(f"  [{f['severity']}] {f['file']}:{f['line']} {f['issue']}")
            if f.get("suggestion"):
                lines.append(f"      → {f['suggestion']}")
        for t in lr.get("missing_tests", []):
            lines.append(f"  missing test: {t}")
    if r.get("llm_error"):
        lines.append(f"(LLM review unavailable: {r['llm_error']})")
    return "\n".join(lines)


def _auth(args: argparse.Namespace) -> int:
    from promapkin import credentials
    if args.auth_cmd == "status":
        for prov, st in credentials.status().items():
            print(f"{prov:<10} {'configured' if st['configured'] else 'not configured':<15} {st['key']:<14} "
                  f"(source: {st['source']})")
        print(f"store: {credentials.store_path()}")
        return 0
    if args.auth_cmd == "remove":
        print("removed" if credentials.remove(args.provider) else "nothing stored")
        return 0
    if sys.stdin.isatty():
        import getpass
        key = getpass.getpass(f"{args.provider} API key: ")
    else:
        key = sys.stdin.readline()
    if not key.strip():
        print("error: empty key", file=sys.stderr)
        return 2
    path = credentials.set_key(args.provider, key, args.auth_base_url)
    print(f"stored {args.provider} key in {path} (mode 0600)")
    return 0


def _parsers(args: argparse.Namespace, ws: Workspace) -> int:
    from promapkin import toolchain
    if args.parsers_cmd == "status":
        emit(args, toolchain.status(ws.root))
        return 0
    if not args.yes and not _ask(f"Install {args.which} parser(s) into {toolchain.cache_dir()}? "
                                 "(npm for TypeScript, pip for tree-sitter)"):
        print("cancelled")
        return 0
    ok = True
    if args.which in ("all", "typescript"):
        good, msg = toolchain.install_typescript()
        ok &= good
        print(msg)
    if args.which in ("all", "tree-sitter"):
        good, msg = toolchain.install_tree_sitter()
        ok &= good
        print(msg)
    return 0 if ok else 1


def _doctor(ws: Workspace) -> dict[str, Any]:
    import importlib.util
    import shutil as _sh
    from promapkin import credentials, toolchain, ml
    checks = []

    def add(check: str, ok: bool, detail: str, optional: bool = False) -> None:
        checks.append({"check": check, "ok": ok, "detail": detail, "optional": optional})

    add("python", sys.version_info >= (3, 10), sys.version.split()[0])
    add("git", bool(_sh.which("git")), _sh.which("git") or "not found (revision tracking disabled)")
    add("workspace", ws.initialized, str(ws.dir) if ws.initialized else "run `promapkin init`")
    if ws.initialized:
        try:
            integrity = ws.store.conn.execute("PRAGMA quick_check").fetchone()[0]
            add("index integrity", integrity == "ok", integrity)
            add("index", bool(ws.store.get_meta("indexed_revision")),
                ws.store.get_meta("indexed_revision") or "not indexed (promapkin scan)")
        except Exception as exc:  # noqa: BLE001
            add("index integrity", False, str(exc))
    st = toolchain.status(ws.root)
    add("javascript/typescript parser", st["javascript_typescript"] == "typescript-compiler",
        f"{st['javascript_typescript']} {st['typescript_version'] or ''} (promapkin parsers install typescript)",
        optional=True)
    add("other-language parser", st["tree_sitter"], f"{st['other_languages']} (promapkin parsers install tree-sitter)",
        optional=True)
    mm = ml.manager(ws).status()
    neural = any(b["installed"] and b["cached"] for b in mm["backends"])
    add("neural embedding model", neural, "cached" if neural else "built-in tier (promapkin models download)",
        optional=True)
    for prov in ("anthropic", "openai"):
        sdk = importlib.util.find_spec(prov) is not None
        st_ = credentials.status()[prov]
        add(f"{prov} LLM", sdk and st_["configured"],
            f"sdk {'installed' if sdk else 'missing (pip install promapkin[llm])'}, key {st_['source']}",
            optional=True)
    add("numpy (fast vector math)", importlib.util.find_spec("numpy") is not None, "optional speed-up",
        optional=True)
    add("PDF text (pypdf)", importlib.util.find_spec("pypdf") is not None,
        "installed" if importlib.util.find_spec("pypdf") else "built-in reader (pip install 'promapkin[assets]')",
        optional=True)
    add("YAML (PyYAML)", importlib.util.find_spec("yaml") is not None,
        "installed" if importlib.util.find_spec("yaml") else "built-in subset parser (pip install 'promapkin[yaml]')",
        optional=True)
    ocr = importlib.util.find_spec("pytesseract") is not None and bool(_sh.which("tesseract"))
    add("image OCR", ocr, "pytesseract + tesseract" if ocr else "off (pip install pytesseract pillow; install "
        "tesseract); --llm describes images instead", optional=True)
    if ws.initialized:
        from promapkin import installer
        found = installer.detect(ws.root)
        add("coding assistants", True, ", ".join(found) + " detected (promapkin install --detected)" if found
            else "none detected (promapkin install --list)", optional=True)
    return {"checks": checks}


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
        emit(args, svc.trace(args.src, args.dst, k=args.k, mode=args.mode), fmt_trace)
    elif c == "hubs":
        emit(args, svc.hubs(args.limit), lambda d: "\n".join(
            f"{h['fan_in']:>5} in {h['fan_out']:>5} out  {h['kind']:<9} {h['id']}  {h['location'] or ''}"
            for h in d["hubs"]))
    elif c == "communities":
        emit(args, svc.communities(), lambda d: "\n".join(
            f"{c['component']:<32} {len(c['modules']):>4} modules  conf {c['confidence']:.2f}  ({c['method']})"
            for c in d["communities"]))
    elif c == "surprises":
        emit(args, svc.surprises(args.limit), lambda d: "\n".join(
            [f"{d['communities']} communities; {d['policy']}"] +
            [f"  {x['source']} -{x['type']} [{x['evidence']}]-> {x['target']}  ({x['why']})"
             for x in d["connections"]]))
    elif c == "html":
        emit(args, svc.export_html(args.out, args.max_nodes),
             lambda d: f"wrote {d['path']} ({d['nodes']} nodes, {d['edges']} edges, {d['communities']} communities)")
    elif c == "dataflow":
        emit(args, svc.dataflow(args.ref), fmt_flows)
    elif c == "deps":
        emit(args, svc.dependencies(args.module))
    elif c == "assets":
        emit(args, svc.assets(args.media), lambda d: "\n".join(
            [f"{a['media']:<6} {a['file']:<48} " + " ".join(
                f"{k}={a[k]}" for k in ("pages", "width", "height", "duration_seconds", "image_kind") if k in a)
             + (" text" if a.get("has_text") else "") + (f"  ← {', '.join(a['referenced_by'][:3])}"
                                                       if a["referenced_by"] else "")
             for a in d["assets"]]) or "no assets indexed")
    elif c == "infra":
        emit(args, svc.infrastructure(), _fmt_infra)
    elif c == "schema":
        emit(args, svc.data_model(), _fmt_schema)
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
        body = _handoff_body(args)
        engine = getattr(args, "_engine", None)
        if getattr(args, "notes", None):
            notes = Path(args.notes[1:]).read_text() if args.notes.startswith("@") else args.notes
            if engine is not None:
                ctx = svc.context(body.get("task") or notes[:200], budget=1500, include_source=False)
                draft = engine.draft_handoff(notes, {"facts": ctx["facts"][:30]})
                for k, v in draft.items():   # explicit flags win over the draft
                    if v and not body.get(k):
                        body[k] = v
            else:
                body.setdefault("summary", notes[:2000])
        emit(args, svc.create_handoff(body, args.parent), fmt_handoff)
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
