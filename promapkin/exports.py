"""Getting the graph out, and several graphs together.

- `graph export`: the whole graph as JSON (nodes, edges, hyperedges, with evidence and confidence) or GraphML
  (Gephi, yEd, NetworkX).
- `graph merge`: several repositories (or JSON exports) into one cross-repository workspace. Every node is
  namespaced by repository, and links between repositories are added from facts both sides already hold.
- `wiki`: an agent-crawlable Markdown wiki, one page per subsystem, module, endpoint group, table set and
  infrastructure, all cross-linked.
- `graph callflow`: Mermaid call-flow diagrams from entry points, as `.mmd` sources and one HTML page.
"""

from __future__ import annotations

import html
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape, quoteattr

from promapkin import __version__
from promapkin import evidence as ev
from promapkin.store import Store

SCHEMA_VERSION = 1
FLOW_EDGES = ("routes_to", "calls", "instantiates", "invokes", "reads", "writes", "produces", "consumes")


# ====================================================================== export
def export_graph(ws: Any, include_hyperedges: bool = True) -> dict[str, Any]:
    from promapkin import insights
    s = ws.store
    nodes = [{"id": n["id"], "kind": n["kind"], "name": n["name"], "qualname": n["qualname"], "file": n["file"],
              "line_start": n["line_start"], "line_end": n["line_end"], "evidence": n["evidence"],
              "evidence_class": ev.classify(n["evidence"]), "confidence": n["confidence"], "status": n["status"],
              "text": (n["text"] or "")[:500] or None, "attrs": n["attrs"]} for n in s.nodes()]
    edges = [{"source": e["src"], "target": e["dst"], "type": e["type"], "evidence": e["evidence"],
              "evidence_class": ev.classify(e["evidence"]), "confidence": e["confidence"], "status": e["status"],
              "file": e["file"], "line": e["line"], "attrs": e["attrs"]} for e in s.edges()]
    hyper = insights.hyperedges(s, ws.root) if include_hyperedges else []
    return {"schema_version": SCHEMA_VERSION, "promapkin_version": __version__,
            "repository": ws.root.name, "revision": s.get_meta("indexed_revision"), "exported_at": time.time(),
            "nodes": nodes, "edges": edges, "hyperedges": hyper,
            "counts": {"nodes": len(nodes), "edges": len(edges), "hyperedges": len(hyper)}}


def write_export(ws: Any, out: Path, fmt: str = "json") -> dict[str, Any]:
    data = export_graph(ws)
    out.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "json":
        out.write_text(json.dumps(data, separators=(",", ":"), default=str))
    elif fmt == "graphml":
        out.write_text(to_graphml(data))
    else:
        raise ValueError("format must be json or graphml")
    return {"path": str(out), "format": fmt, **data["counts"], "bytes": out.stat().st_size}


def to_graphml(data: dict[str, Any]) -> str:
    keys = [("kind", "node"), ("name", "node"), ("file", "node"), ("evidence", "node"), ("confidence", "node"),
            ("type", "edge"), ("evidence", "edge"), ("confidence", "edge")]
    out = ['<?xml version="1.0" encoding="UTF-8"?>',
           '<graphml xmlns="http://graphml.graphdrawing.org/xmlns">']
    for k, dom in keys:
        typ = "double" if k == "confidence" else "string"
        out.append(f'  <key id="{dom[0]}_{k}" for="{dom}" attr.name="{k}" attr.type="{typ}"/>')
    out.append('  <graph id="promapkin" edgedefault="directed">')
    ids = set()
    for n in data["nodes"]:
        ids.add(n["id"])
        out.append(f"    <node id={quoteattr(n['id'])}>" + "".join(
            f'<data key="n_{k}">{escape(str(n.get(k) if n.get(k) is not None else ""))}</data>'
            for k in ("kind", "name", "file", "evidence", "confidence")) + "</node>")
    for i, e in enumerate(data["edges"]):
        if e["source"] in ids and e["target"] in ids:
            out.append(f"    <edge id=\"e{i}\" source={quoteattr(e['source'])} target={quoteattr(e['target'])}>" + "".join(
                f'<data key="e_{k}">{escape(str(e.get(k) if e.get(k) is not None else ""))}</data>'
                for k in ("type", "evidence", "confidence")) + "</edge>")
    out += ["  </graph>", "</graphml>"]
    return "\n".join(out) + "\n"


# ====================================================================== merge
def _load(source: Path) -> tuple[str, dict[str, Any]]:
    """(namespace, export) from a repository directory (with a Promapkin index) or a JSON export."""
    from promapkin.config import Workspace
    if source.is_file():
        data = json.loads(source.read_text())
        if "nodes" not in data or "edges" not in data:
            raise ValueError(f"{source} is not a Promapkin graph export")
        return data.get("repository") or source.stem, data
    ws = Workspace(source)
    if not ws.initialized or not ws.store.get_meta("indexed_revision"):
        raise ValueError(f"{source} has no Promapkin index (run `promapkin scan` there first)")
    return source.resolve().name, export_graph(ws)


def merge(sources: list[Path], out_dir: Path, names: dict[str, str] | None = None) -> dict[str, Any]:
    """Merge graphs into a workspace at `out_dir` (queryable with `promapkin --root out_dir ...`)."""
    from promapkin.config import Workspace
    loaded = []
    for src in sources:
        ns, data = _load(src)
        ns = (names or {}).get(str(src), ns)
        while ns in {n for n, _ in loaded}:
            ns += "_"
        loaded.append((ns, data))
    out_dir.mkdir(parents=True, exist_ok=True)
    ws = Workspace(out_dir)
    ws.init()
    s = ws.store
    with s.transaction():
        s.conn.execute("DELETE FROM edges")
        s.conn.execute("DELETE FROM nodes")
        s.conn.execute("DELETE FROM search")
        for ns, data in loaded:
            s.upsert_node(f"{ns}::repo", "repository", ns, evidence=ev.MANIFEST,
                          attrs={"revision": data.get("revision"), "repository": ns})
            for n in data["nodes"]:
                if n["kind"] == "repository":
                    continue
                s.upsert_node(f"{ns}::{n['id']}", n["kind"], n["name"], evidence=n["evidence"],
                              qualname=f"{ns}::{n['qualname']}" if n.get("qualname") else None,
                              file=f"{ns}/{n['file']}" if n.get("file") else None, line_start=n.get("line_start"),
                              line_end=n.get("line_end"), confidence=n.get("confidence") or 1.0,
                              status=n.get("status") or "verified", text=n.get("text"),
                              owner_file=f"{ns}/{n['file']}" if n.get("file") else None,
                              attrs={**(n.get("attrs") or {}), "repository": ns, "original_id": n["id"]})
            def nid(x: str, ns: str = ns) -> str:
                return f"{ns}::repo" if x == "repo:root" else f"{ns}::{x}"
            for e in data["edges"]:
                s.add_edge(nid(e["source"]), nid(e["target"]), e["type"], evidence=e["evidence"],
                           confidence=e.get("confidence") or 1.0, file=f"{ns}/{e['file']}" if e.get("file") else None,
                           line=e.get("line"), status=e.get("status") or "verified", attrs=e.get("attrs") or {})
        links = _cross_links(s, loaded)
        s.set_meta("indexed_revision", "merged:" + ",".join(f"{ns}@{(d.get('revision') or '')[:12]}"
                                                            for ns, d in loaded))
        s.set_meta("indexed_at", str(time.time()))
        s.set_meta("merged_from", json.dumps([ns for ns, _ in loaded]))
    hyper = [{**h, "id": f"{ns}::{h['id']}", "members": [f"{ns}::{m}" for m in h["members"]]}
             for ns, data in loaded for h in data.get("hyperedges", [])]
    (out_dir / ".promapkin" / "hyperedges.json").write_text(json.dumps(hyper))
    return {"workspace": str(out_dir), "repositories": [ns for ns, _ in loaded],
            "nodes": s.conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0],
            "edges": s.conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0], "cross_repo_links": links}


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", (name or "").lower())


def _cross_links(s: Store, loaded: list[tuple[str, dict[str, Any]]]) -> dict[str, int]:
    """Links between repositories from facts both sides hold. Each carries `cross_repo: true`."""
    counts: Counter[str] = Counter()
    by_ns = {ns: data for ns, data in loaded}

    def link(a: str, b: str, typ: str, evidence: str, conf: float, why: str) -> None:
        s.add_edge(a, b, typ, evidence=evidence, confidence=conf, attrs={"cross_repo": True, "why": why})
        counts[typ] += 1

    # What each repository publishes: package names from manifests, top-level modules, endpoints, schemas.
    provides: dict[str, tuple[str, str]] = {}
    for ns, data in loaded:
        for n in data["nodes"]:
            a = n.get("attrs") or {}
            if n["kind"] == "manifest" and a.get("crate"):
                provides[_norm(a["crate"])] = (ns, f"{ns}::repo")
            if n["kind"] == "module" and "." not in n["name"] and "/" not in n["name"] and not a.get("test"):
                provides.setdefault(_norm(n["name"]), (ns, f"{ns}::{n['id']}"))
        provides.setdefault(_norm(ns), (ns, f"{ns}::repo"))
    for ns, data in loaded:
        for n in data["nodes"]:
            if n["kind"] == "package":
                hit = provides.get(_norm(n["name"]))
                if hit and hit[0] != ns:
                    link(f"{ns}::{n['id']}", hit[1], "provided_by", ev.MANIFEST, 0.9,
                         f"package {n['name']} is published by {hit[0]}")
    # Spec operations and code endpoints across repositories (an API spec repo, a client, a service).
    route = re.compile(r"\{[^}]*\}|:[A-Za-z_]\w*|<[^>]*>")
    endpoints: dict[tuple[str, str], list[str]] = defaultdict(list)
    for ns, data in loaded:
        for n in data["nodes"]:
            if n["kind"] == "endpoint" and " " in n["name"]:
                m, _, p = n["name"].partition(" ")
                endpoints[(m.upper(), route.sub("{}", p).rstrip("/") or "/")].append(f"{ns}::{n['id']}")
    for ns, data in loaded:
        for n in data["nodes"]:
            if n["kind"] == "api_operation":
                a = n.get("attrs") or {}
                key = (str(a.get("method", "")).upper(), route.sub("{}", str(a.get("path", ""))).rstrip("/") or "/")
                for target in endpoints.get(key, []):
                    if not target.startswith(f"{ns}::"):
                        link(f"{ns}::{n['id']}", target, "specifies", ev.MANIFEST, 0.9, "same method and route")
    # Shared names that usually mean shared resources: tables (one database), config keys, schema types.
    for kind, typ, conf, why in (("table", "same_as", 0.6, "same table name: likely a shared database"),
                                 ("config", "same_as", 0.5, "same configuration key"),
                                 ("schema_type", "same_as", 0.6, "same schema type name")):
        owners: dict[str, list[str]] = defaultdict(list)
        for ns, data in loaded:
            for n in data["nodes"]:
                if n["kind"] == kind:
                    owners[n["name"].lower()].append(f"{ns}::{n['id']}")
        for ids in owners.values():
            repos = {i.split("::", 1)[0] for i in ids}
            if len(repos) > 1:
                for a in ids:
                    for b in ids:
                        if a < b and a.split("::", 1)[0] != b.split("::", 1)[0]:
                            link(a, b, typ, ev.NAME_MATCH, conf, why)
    # Deployments and services whose image or name names another repository.
    for ns, data in loaded:
        for n in data["nodes"]:
            if n["kind"] not in ("deployment", "service", "container_image"):
                continue
            a = n.get("attrs") or {}
            images = a.get("images") or ([a["image"]] if a.get("image") else [])
            names = {_norm(n["name"])} | {_norm(str(i).split("/")[-1].split(":")[0]) for i in images if i}
            for other in by_ns:
                if other != ns and _norm(other) in names:
                    link(f"{ns}::{n['id']}", f"{other}::repo", "deploys", ev.NAME_MATCH, 0.6,
                         f"{n['kind']} {n['name']} names repository {other}")
    return dict(counts)


# ====================================================================== wiki
def wiki(ws: Any, out: Path | None = None, max_modules: int = 2000) -> dict[str, Any]:
    from promapkin import infra_view, insights
    from promapkin.service import Service
    s = ws.store
    svc = Service(ws, agent="wiki")
    out = out or ws.dir / "wiki"
    (out / "modules").mkdir(parents=True, exist_ok=True)
    (out / "subsystems").mkdir(parents=True, exist_ok=True)
    ov = svc.overview()
    modules = [m for m in s.nodes("module")][:max_modules]
    sub_of = {m["id"]: _subsystem(m["file"] or m["name"]) for m in modules}
    page = {m["id"]: f"modules/{_slug(m['name'])}.md" for m in modules}
    pages = 0

    def link_to(nid: str, base: str = "") -> str:
        target = page.get(nid) or page.get(_module_id(s, nid) or "")
        label = nid.split(":", 1)[-1].rsplit(":", 1)[-1]
        return f"[`{label}`]({base}{target})" if target else f"`{label}`"

    # index
    subs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for m in modules:
        subs[sub_of[m["id"]]].append(m)
    lines = [f"# {ov.get('repository') or ws.root.name} — codebase wiki", "",
             f"Generated by Promapkin from the index at `{str(s.get_meta('indexed_revision'))[:12]}`. Every page is "
             "derived from the graph; facts carry their evidence.", ""]
    if ov.get("styles"):
        lines += ["## Architecture", ""] + [f"- {x['statement']}" for x in ov["styles"]] + [""]
    lines += ["## Subsystems", ""] + [f"- [{name}](subsystems/{_slug(name)}.md) — {len(ms)} module(s)"
                                      for name, ms in sorted(subs.items())] + [""]
    if ov.get("entry_points"):
        lines += ["## Entry points", ""] + [f"- {e['statement'] if isinstance(e, dict) else e}"
                                           for e in ov["entry_points"][:20]] + [""]
    hubs = insights.hubs(s, 15)
    if hubs:
        lines += ["## Most depended-on symbols", ""] + [f"- {link_to(h['id'])} — {h['fan_in']} inbound"
                                                       for h in hubs] + [""]
    extra = [("endpoints.md", "API endpoints"), ("data.md", "Data model"), ("infrastructure.md", "Infrastructure"),
             ("decisions.md", "Decisions")]
    lines += ["## More", ""] + [f"- [{t}]({p})" for p, t in extra] + [""]
    (out / "index.md").write_text("\n".join(lines))
    pages += 1
    # subsystems
    for name, ms in subs.items():
        body = [f"# Subsystem `{name}`", "", "[← index](../index.md)", "", "## Modules", ""]
        for m in sorted(ms, key=lambda x: x["name"]):
            body.append(f"- {link_to(m['id'], '../')} — {(m['text'] or '').splitlines()[0][:120] if m['text'] else ''}")
        (out / "subsystems" / f"{_slug(name)}.md").write_text("\n".join(body) + "\n")
        pages += 1
    # modules
    for m in modules:
        (out / page[m["id"]]).write_text(_module_page(s, m, sub_of[m["id"]], link_to))
        pages += 1
    # endpoints, data, infrastructure, decisions
    eps = sorted(s.nodes("endpoint"), key=lambda n: n["name"])
    body = ["# API endpoints", "", "[← index](index.md)", "", "| Endpoint | Protocol | Handler | Location |", "|---|---|---|---|"]
    for e in eps:
        handler = next((x["dst"] for x in s.edges(src=e["id"], types=["routes_to"])), "")
        body.append(f"| `{e['name']}` | {e['attrs'].get('protocol', 'http')} | "
                    f"{link_to(handler) if handler else '—'} | {e['file']}:{e['line_start']} |")
    (out / "endpoints.md").write_text("\n".join(body) + "\n")
    dm = infra_view.data_model(s)
    body = ["# Data model", "", "[← index](index.md)", ""]
    for t in dm["tables"]:
        body += [f"## {'View' if t['view'] else 'Table'} `{t['name']}`", "",
                 f"Defined by {t['defined_by']} ({t['location']})." if t["location"] else "", ""]
        if t["columns"]:
            body += ["| Column | Type | Key |", "|---|---|---|"] + [
                f"| {c['name']} | {c.get('type', '')} | {'PK' if c['name'] in t['primary_key'] else ''} |"
                for c in t["columns"]] + [""]
        for label, key in (("References", "references"), ("Mapped by", "orm_classes"), ("Read by", "read_by"),
                           ("Written by", "written_by")):
            items = t[key]
            if items:
                rendered = [r["table"] if isinstance(r, dict) else link_to(r) for r in items[:15]]
                body.append(f"- **{label}:** " + ", ".join(rendered))
        body.append("")
    (out / "data.md").write_text("\n".join(body) + "\n")
    infra = infra_view.infrastructure(s)
    body = ["# Infrastructure", "", "[← index](index.md)", ""]
    for d in infra["deployables"]:
        body.append(f"- **{d['kind'].replace('_', ' ')} `{d['name']}`** ({d['location']})"
                    + (" — runs " + ", ".join(link_to(t) for t in d["runs_code"]) if d["runs_code"] else "")
                    + (" — config " + ", ".join(f"`{c}`" for c in d["config"][:8]) if d["config"] else ""))
    for cat, items in infra["cloud_resources"].items():
        body.append(f"- **{cat}:** " + ", ".join(f"`{i['type']}`" for i in items))
    (out / "infrastructure.md").write_text("\n".join(body) + "\n")
    decs = s.nodes("decision")
    body = ["# Decisions", "", "[← index](index.md)", ""] + [
        f"- **{d['name']}** ({d['attrs'].get('status', 'unknown')}) — {(d['text'] or '')[:300]} ({d['file']})"
        for d in decs]
    (out / "decisions.md").write_text("\n".join(body) + "\n")
    pages += 4
    return {"path": str(out), "pages": pages, "modules": len(modules), "subsystems": len(subs)}


def _module_page(s: Store, m: dict[str, Any], sub: str, link_to: Any) -> str:
    mid = m["id"]
    body = [f"# Module `{m['name']}`", "", f"[← subsystem {sub}](../subsystems/{_slug(sub)}.md) · "
            f"`{m['file']}` · layer: {(m['attrs'].get('layer') or {}).get('label', '?')}", ""]
    if m["text"]:
        body += [m["text"][:800], ""]
    symbols = s.conn.execute("SELECT id, kind, name, line_start, text, attrs FROM nodes WHERE owner_file = ? AND "
                             "kind IN ('class','function','method','endpoint','test') ORDER BY line_start",
                             (m["file"],)).fetchall()
    if symbols:
        body += ["## Symbols", ""]
        for sid, kind, name, line, text, attrs in symbols[:200]:
            params = ", ".join(p.get("name", "") for p in (json.loads(attrs or "{}").get("params") or [])[:8])
            body.append(f"- **{kind}** `{name}({params})`" if kind in ("function", "method") else
                        f"- **{kind}** `{name}`")
            if text:
                body.append(f"  — {' '.join(text.split())[:200]}")
        body.append("")
    imports = [e["dst"] for e in s.edges(src=mid, types=["imports"])]
    users = [e["src"] for e in s.edges(dst=mid, types=["imports"])]
    if imports:
        body += ["## Imports", ""] + [f"- {link_to(i, '../')}" for i in sorted(set(imports))[:60]] + [""]
    if users:
        body += ["## Imported by", ""] + [f"- {link_to(u, '../')}" for u in sorted(set(users))[:60]] + [""]
    data = s.conn.execute("SELECT DISTINCT e.type, e.dst FROM edges e JOIN nodes n ON n.id = e.src WHERE "
                          "n.owner_file = ? AND e.type IN ('reads','writes','maps_to') AND (e.dst LIKE 'table:%' OR "
                          "e.dst LIKE 'config:%' OR e.dst LIKE 'resource:%')", (m["file"],)).fetchall()
    if data:
        body += ["## Data and configuration", ""] + [f"- {t} `{d.split(':', 1)[1]}`" for t, d in data[:40]] + [""]
    docs = s.conn.execute("SELECT DISTINCT e.src FROM edges e JOIN nodes n ON n.id = e.dst WHERE n.owner_file = ? "
                          "AND e.type = 'documents'", (m["file"],)).fetchall()
    if docs:
        body += ["## Documented in", ""] + [f"- `{d[0].split(':', 1)[1]}`" for d in docs[:20]] + [""]
    return "\n".join(body) + "\n"


def _module_id(s: Store, nid: str) -> str | None:
    row = s.conn.execute("SELECT owner_file FROM nodes WHERE id = ?", (nid,)).fetchone()
    if not row or not row[0]:
        return None
    m = s.conn.execute("SELECT id FROM nodes WHERE kind='module' AND owner_file = ?", (row[0],)).fetchone()
    return m[0] if m else None


def _subsystem(path: str) -> str:
    parts = path.split("/")
    return "/".join(parts[:2]) if len(parts) > 2 else parts[0] if len(parts) > 1 else "(root)"


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-")[:120] or "root"


# ====================================================================== call flow
def callflow(ws: Any, entry: str | None = None, depth: int = 4, max_nodes: int = 60) -> list[dict[str, Any]]:
    """Mermaid flowcharts of what entry points call, down to data sinks. Without `entry`: the endpoints and
    executable entry modules, most connected first."""
    from promapkin.graph import Graph
    s = ws.store
    g = Graph(s)
    if entry:
        starts = [g.resolve(entry)]
    else:
        eps = [n["id"] for n in s.nodes("endpoint")]
        mains = [n["id"] for n in s.nodes("module") if n["attrs"].get("entrypoint")]
        starts = (eps + mains)[:12]
    out = []
    for start in starts:
        nodes, edges, frontier, seen = [start], [], [start], {start}
        for _ in range(depth):
            nxt = []
            for nid in frontier:
                for e in s.edges(src=nid, types=list(FLOW_EDGES)):
                    if e["dst"].startswith(("package:",)) or len(nodes) >= max_nodes:
                        continue
                    edges.append(e)
                    if e["dst"] not in seen:
                        seen.add(e["dst"])
                        nodes.append(e["dst"])
                        nxt.append(e["dst"])
            frontier = nxt
        out.append({"entry": start, "nodes": len(nodes), "mermaid": _mermaid_flow(s, start, nodes, edges)})
    return out


def _mermaid_flow(s: Store, start: str, nodes: list[str], edges: list[dict[str, Any]]) -> str:
    ids = {n: f"n{i}" for i, n in enumerate(nodes)}
    groups: dict[str, list[str]] = defaultdict(list)
    lines = ["flowchart LR"]
    for nid in nodes:
        n = s.node(nid) or {"kind": "?", "name": nid, "owner_file": None}
        label = (n["name"] or nid).replace('"', "'")[:48]
        shape = {"table": ('[("', '")]'), "endpoint": ('(["', '"])'), "config": ('{{"', '"}}'),
                 "resource": ('[/"', '"/]'), "cloud_resource": ('[("', '")]')}.get(n["kind"], ('["', '"]'))
        decl = f'{ids[nid]}{shape[0]}{label}{shape[1]}'
        owner = n.get("owner_file") if n["kind"] in ("function", "method", "class") else None
        (groups[owner] if owner else groups[""]).append(decl)
    for owner, decls in groups.items():
        if owner:
            lines.append(f'  subgraph {re.sub(r"[^A-Za-z0-9_]", "_", owner)}["{owner}"]')
            lines += [f"    {d}" for d in decls]
            lines.append("  end")
        else:
            lines += [f"  {d}" for d in decls]
    for e in edges:
        if e["src"] in ids and e["dst"] in ids:
            arrow = "-->" if ev.classify(e["evidence"]) == "observed" else "-.->"
            lines.append(f"  {ids[e['src']]} {arrow}|{e['type']}| {ids[e['dst']]}")
    lines.append(f"  style {ids[start]} stroke-width:3px")
    return "\n".join(lines)


def callflow_html(ws: Any, out: Path, flows: list[dict[str, Any]], title: str = "Call flow") -> dict[str, Any]:
    out.parent.mkdir(parents=True, exist_ok=True)
    sections = "\n".join(f'<section><h2>{html.escape(f["entry"].split(":", 1)[-1])}</h2>'
                         f'<pre class="mermaid">{html.escape(f["mermaid"])}</pre></section>' for f in flows)
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title>
<style>:root{{--bg:#f6f5f1;--fg:#1d1c1a;--muted:#6c685f;--line:#dcd7cb}}
@media (prefers-color-scheme:dark){{:root{{--bg:#121211;--fg:#ece8df;--muted:#a19b8e;--line:#33312c}}}}
body{{margin:0;padding:16px 20px;font:14px/1.5 system-ui,sans-serif;background:var(--bg);color:var(--fg)}}
h1{{font-size:18px}}h2{{font-size:14px;color:var(--muted);font-family:ui-monospace,monospace}}
section{{border-top:1px solid var(--line);padding:12px 0;overflow-x:auto}}</style></head><body>
<h1>{html.escape(title)}</h1><p style="color:var(--muted)">Solid arrows are observed (parsed, declared or traced);
dashed arrows are inferred. Mermaid sources are next to this file.</p>
{sections}
<script type="module">import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11.4.1/dist/mermaid.esm.min.mjs";
mermaid.initialize({{startOnLoad:true,securityLevel:"strict",theme:matchMedia("(prefers-color-scheme: dark)").matches?"dark":"default"}});</script>
</body></html>"""
    out.write_text(page)
    for i, f in enumerate(flows):
        (out.parent / f"callflow-{i:02d}-{_slug(f['entry'].split(':', 1)[-1])[:60]}.mmd").write_text(f["mermaid"] + "\n")
    return {"path": str(out), "diagrams": len(flows)}
