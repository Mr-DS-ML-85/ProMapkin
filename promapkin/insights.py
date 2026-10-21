"""Graph insights: hubs ("god nodes"), communities, surprising cross-community connections,
and a self-contained interactive HTML explorer.

Surprising connections only use observed (parsed / resolved / runtime) evidence. Name matches,
ML similarity and documentation claims never count, so unrelated scripts that merely call the
same builtin are not reported as connected.
"""

from __future__ import annotations

import html
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from promapkin import evidence as ev
from promapkin.store import Store

CODE_EDGES = ["calls", "instantiates", "imports", "inherits", "implements", "depends_on", "references",
              "routes_to", "overrides", "invokes", "reads", "writes", "produces", "consumes"]
CODE_KINDS = ("module", "class", "function", "method", "endpoint", "test", "resource", "config", "event", "package")
# graph.html also shows infrastructure, data and assets and the edges that tie them to code.
HTML_EDGES = CODE_EDGES + ["runs", "builds", "configures", "maps_to", "specifies", "migrates", "documents",
                           "verifies", "uses", "scales"]
HTML_KINDS = CODE_KINDS + ("table", "service", "deployment", "container_image", "cloud_resource", "ingress",
                           "schema_type", "api_operation", "pipeline", "asset", "config_source", "iac_module")
CATEGORY = {"module": "code", "class": "code", "function": "code", "method": "code", "test": "test",
            "endpoint": "api", "api_operation": "api", "package": "dependency", "resource": "data", "table": "data",
            "schema_type": "data", "config": "config", "config_source": "config", "event": "code",
            "service": "infra", "deployment": "infra", "container_image": "infra", "cloud_resource": "infra",
            "ingress": "infra", "pipeline": "infra", "iac_module": "infra", "asset": "doc"}
TRUSTED = {ev.AST, ev.STATIC_RESOLVED, ev.LEXICAL, ev.RUNTIME, ev.MANIFEST}


def hubs(store: Store, limit: int = 25, kinds: tuple[str, ...] = ("class", "function", "method", "module")
         ) -> list[dict[str, Any]]:
    """Most-connected code nodes, counting distinct neighbours over code edges (docs excluded)."""
    marks = ",".join("?" * len(CODE_EDGES))
    fan_in: Counter[str] = Counter()
    fan_out: Counter[str] = Counter()
    for src, dst in store.conn.execute(f"SELECT DISTINCT src, dst FROM edges WHERE type IN ({marks})", CODE_EDGES):
        fan_out[src] += 1
        fan_in[dst] += 1
    rows = []
    kind_of = {r[0]: (r[1], r[2], r[3], r[4]) for r in store.conn.execute(
        f"SELECT id, kind, file, line_start, attrs FROM nodes WHERE kind IN ({','.join('?' * len(kinds))})", kinds)}
    for nid, (kind, file, line, attrs) in kind_of.items():
        if '"test": true' in (attrs or ""):
            continue
        degree = fan_in[nid] + fan_out[nid]
        if degree:
            rows.append({"id": nid, "kind": kind, "fan_in": fan_in[nid], "fan_out": fan_out[nid], "degree": degree,
                         "location": f"{file}:{line}" if file and line else file})
    rows.sort(key=lambda r: (-r["fan_in"], -r["degree"], r["id"]))
    return rows[:limit]


def communities(store: Store, rt: Any = None) -> list[dict[str, Any]]:
    from promapkin.ml import graph_ml
    model = None
    if rt is not None and rt.classifier.mode != "heuristic":
        model = graph_ml.graph_model_name(rt.embedder.name)
    return graph_ml.cluster_modules(store, model)


def _module_of(store: Store, nid: str, cache: dict[str, str | None]) -> str | None:
    if nid in cache:
        return cache[nid]
    out = None
    if nid.startswith("module:"):
        out = nid
    elif nid.startswith("symbol:") and nid.count(":") >= 2:
        out = "module:" + nid.split(":")[1]
    cache[nid] = out
    return out


def surprises(store: Store, rt: Any = None, limit: int = 20) -> dict[str, Any]:
    """Observed edges that cross community boundaries rarely (few edges between the two communities)."""
    comms = communities(store, rt)
    member = {m: i for i, c in enumerate(comms) for m in c["modules"]}
    cache: dict[str, str | None] = {}
    tests = {n["id"] for n in store.nodes("module") if n["attrs"].get("test")}
    marks = ",".join("?" * len(CODE_EDGES))
    fan_in: Counter[str] = Counter(dst for (dst,) in store.conn.execute(
        f"SELECT dst FROM edges WHERE type IN ({marks})", CODE_EDGES))
    # One pass, keeping per pair of communities only the edge count and the least expected edge (lowest target
    # fan-in, first in id order): a kernel has millions of code edges.
    between: dict[tuple[int, int], list[Any]] = {}
    for src, dst, typ, evidence, status, file, line in store.conn.execute(
            f"SELECT src, dst, type, evidence, status, file, line FROM edges WHERE type IN ({marks}) ORDER BY id",
            CODE_EDGES):
        if evidence not in TRUSTED or status != "verified":
            continue
        a, b = _module_of(store, src, cache), _module_of(store, dst, cache)
        if not a or not b or a == b or a in tests or b in tests:
            continue
        ca, cb = member.get(a), member.get(b)
        if ca is None or cb is None or ca == cb:
            continue
        key = (min(ca, cb), max(ca, cb))
        e = {"src": src, "dst": dst, "type": typ, "evidence": evidence, "file": file, "line": line}
        slot = between.get(key)
        if slot is None:
            between[key] = [1, e]
        else:
            slot[0] += 1
            if fan_in[dst] < fan_in[slot[1]["dst"]]:
                slot[1] = e
    rows = []
    for (ca, cb), (count, e) in between.items():
        # The least expected link: few edges between the communities, and a target that is not a hub
        # everybody imports (a link to `fastapi` from a tutorial is not news).
        rows.append({"source": e["src"], "type": e["type"], "target": e["dst"], "evidence": e["evidence"],
                     "at": f"{e['file']}:{e['line']}" if e.get("file") else None,
                     "communities": [comms[ca]["component"], comms[cb]["component"]],
                     "edges_between_communities": count, "target_fan_in": fan_in[e["dst"]],
                     "why": f"only {count} observed edge(s) connect these two communities"})
    rows.sort(key=lambda r: (r["edges_between_communities"] * (1 + r["target_fan_in"]), r["source"]))
    return {"communities": len(comms), "connections": rows[:limit],
            "policy": "observed evidence only (AST, resolved imports, lexical, runtime); name matches, ML "
                      "similarity and documentation claims are excluded"}


# ------------------------------------------------------------------------------------- HTML explorer

def export_html(store: Store, out: Path, rt: Any = None, max_nodes: int = 1500, title: str = "Promapkin graph",
                root: Any = None) -> dict[str, Any]:
    comms = communities(store, rt)
    member = {m: i for i, c in enumerate(comms) for m in c["modules"]}
    marks = ",".join("?" * len(HTML_EDGES))
    degree: Counter[str] = Counter()
    for src, dst in store.conn.execute(f"SELECT src, dst FROM edges WHERE type IN ({marks})", HTML_EDGES):
        degree[src] += 1
        degree[dst] += 1
    kinds = ",".join("?" * len(HTML_KINDS))
    kind_of = dict(store.conn.execute(f"SELECT id, kind FROM nodes WHERE kind IN ({kinds}) ORDER BY id",
                                      HTML_KINDS))
    # Keep the most connected nodes, but never drop infrastructure/data nodes for code hubs: they are few and are
    # what the code graph alone cannot show. Only the kept nodes and the edges among them are read in full.
    ranked = sorted((n for n in kind_of if degree[n]), key=lambda n: (CATEGORY.get(kind_of[n]) == "code",
                                                                      -degree[n]))
    keep = ranked[:max_nodes]
    keep_set = set(keep)
    nodes = {nid: store.node(nid) for nid in keep}
    edges = []
    for i in range(0, len(keep), 400):
        chunk = keep[i:i + 400]
        edges += [e for e in store.edges(where=f"type IN ({marks}) AND src IN ({','.join('?' * len(chunk))})",
                                         params=[*HTML_EDGES, *chunk]) if e["dst"] in keep_set]
    edges.sort(key=lambda e: e["id"])
    cache: dict[str, str | None] = {}
    data_nodes = []
    for nid in keep:
        n = nodes[nid]
        mod = _module_of(store, nid, cache) if CATEGORY.get(n["kind"]) in ("code", "test", "api") else None
        data_nodes.append({"id": nid, "label": n["name"], "kind": n["kind"], "cat": CATEGORY.get(n["kind"], "code"),
                           "degree": degree[nid], "group": member.get(mod, -1) if mod else -1, "ev": n["evidence"],
                           "loc": f"{n['file']}:{n['line_start']}" if n["file"] and n["line_start"] else (n["file"] or ""),
                           "doc": (n["text"] or "")[:240]})
    data_edges = []
    seen = set()
    for e in edges:
        if e["src"] in keep_set and e["dst"] in keep_set:
            key = (e["src"], e["dst"], e["type"])
            if key in seen:
                continue
            seen.add(key)
            data_edges.append({"s": e["src"], "t": e["dst"], "type": e["type"], "ev": e["evidence"],
                               "cls": ev.classify(e["evidence"]), "w": round(float(e["confidence"] or 1), 2)})
    hyper = []
    for h in hyperedges(store, root, rt, kinds=("co_change", "flow", "pattern")):
        members = [m for m in h["members"] if m in keep_set]
        if len(members) >= 2:
            hyper.append({"kind": h["kind"], "label": h["label"], "members": members, "weight": h.get("weight")})
    payload = {"nodes": data_nodes, "edges": data_edges, "hyper": hyper[:200],
               "groups": [c["component"] for c in comms], "title": title,
               "truncated": len(ranked) > len(data_nodes)}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(HTML_TEMPLATE.replace("__TITLE__", html.escape(title)).replace(
        "__DATA__", json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")))
    return {"path": str(out), "nodes": len(data_nodes), "edges": len(data_edges), "communities": len(comms),
            "truncated": payload["truncated"], "bytes": out.stat().st_size}


HTML_TEMPLATE = (Path(__file__).parent / "graph_template.html").read_text()


def semantic_surprises(store: Store, rt: Any = None, limit: int = 20) -> list[dict[str, Any]]:
    """Code that looks alike (`similar_to`) across community boundaries with no structural link between them:
    candidates for duplicated logic or a missing shared abstraction. Inferred evidence, ranked by similarity."""
    comms = communities(store, rt)
    member = {m: i for i, c in enumerate(comms) for m in c["modules"]}
    cache: dict[str, str | None] = {}
    marks = ",".join("?" * len(CODE_EDGES))
    linked_sql = (f"SELECT 1 FROM edges WHERE ((src=? AND dst=?) OR (src=? AND dst=?)) AND type IN ({marks}) "
                  f"LIMIT 1")
    rows = []
    for e in store.edges(types=["similar_to"]):
        a, b = e["src"], e["dst"]
        ma, mb = _module_of(store, a, cache), _module_of(store, b, cache)
        if not ma or not mb or ma == mb:
            continue
        ca, cb = member.get(ma), member.get(mb)
        if ca is None or cb is None or ca == cb:
            continue
        if store.conn.execute(linked_sql, (a, b, b, a, *CODE_EDGES)).fetchone():
            continue
        na, nb = store.node(a), store.node(b)
        if not na or not nb or na["attrs"].get("test") or nb["attrs"].get("test") or \
                na["kind"] == "test" or nb["kind"] == "test":
            continue
        rows.append({"a": a, "b": b, "similarity": round(float(e["confidence"] or 0), 3), "evidence": e["evidence"],
                     "communities": [comms[ca]["component"], comms[cb]["component"]],
                     "locations": [f"{na['file']}:{na['line_start']}", f"{nb['file']}:{nb['line_start']}"],
                     "why": "similar code in separate communities with no call, import or inheritance between them"})
    rows.sort(key=lambda r: -r["similarity"])
    return rows[:limit]


# ====================================================================== hyperedges
def hyperedges(store: Store, root: Any = None, rt: Any = None, kinds: tuple[str, ...] = ("co_change", "flow",
                                                                                         "pattern", "community"),
               max_commits: int = 600, min_support: int = 3) -> list[dict[str, Any]]:
    """Relations among more than two nodes, which pairwise edges flatten:
    - co_change: modules that git history shows changing together (exact sets seen in >= min_support commits);
    - flow: an entry point's whole path to a data sink;
    - pattern: a design pattern and the classes playing its roles;
    - community: modules grouped by imports and meaning."""
    out: list[dict[str, Any]] = []
    if "co_change" in kinds and root is not None:
        out += _co_change(store, root, max_commits, min_support)
    if "flow" in kinds:
        out += _flows(store)
    if "pattern" in kinds:
        by_pattern: dict[str, list[str]] = defaultdict(list)
        for e in store.edges(types=["implements_pattern"]):
            by_pattern[e["dst"]].append(e["src"])
        for pid, members in by_pattern.items():
            n = store.node(pid)
            if n and len(set(members)) >= 2:
                out.append({"id": f"hyper:pattern:{pid}", "kind": "pattern", "label": n["name"],
                            "members": sorted(set(members)), "weight": round(float(n["confidence"] or 0.5), 2),
                            "evidence": n["evidence"]})
    if "community" in kinds:
        for c in communities(store, rt):
            if len(c["modules"]) >= 2:
                out.append({"id": f"hyper:community:{c['component']}", "kind": "community", "label": c["component"],
                            "members": c["modules"], "weight": c.get("cohesion"), "evidence": c.get("method")})
    return out


def _co_change(store: Store, root: Any, max_commits: int, min_support: int) -> list[dict[str, Any]]:
    import subprocess
    try:
        log = subprocess.run(["git", "-C", str(root), "log", f"-n{max_commits}", "--no-merges", "--name-only",
                              "--format=@@%H"], capture_output=True, text=True, timeout=60, check=False).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    module_of_file = {f: m for m, f in store.conn.execute(
        "SELECT id, file FROM nodes WHERE kind='module' AND file IS NOT NULL")}
    tests = {n["id"] for n in store.nodes("module") if n["attrs"].get("test")}
    sets: Counter[frozenset[str]] = Counter()
    per_module: Counter[str] = Counter()
    for chunk in log.split("@@")[1:]:
        files = [ln.strip() for ln in chunk.splitlines()[1:] if ln.strip()]
        mods = frozenset(module_of_file[f] for f in files if f in module_of_file and module_of_file[f] not in tests)
        for m in mods:
            per_module[m] += 1
        if 2 <= len(mods) <= 6:
            sets[mods] += 1
    out = []
    for mods, n in sets.most_common(60):
        if n < min_support:
            break
        confidence = round(n / max(1, min(per_module[m] for m in mods)), 2)
        out.append({"id": "hyper:co_change:" + "+".join(sorted(m.split(":", 1)[-1] for m in mods))[:200],
                    "kind": "co_change", "label": f"changed together in {n} commits", "members": sorted(mods),
                    "weight": confidence, "support": n, "evidence": "git_history"})
    return out


def _flows(store: Store, limit: int = 40) -> list[dict[str, Any]]:
    from promapkin.graph import Graph
    g = Graph(store)
    starts = [n["id"] for n in store.nodes("endpoint")][:limit]
    out = []
    for sid in starts:
        try:
            flows = g.dataflow(sid, max_depth=6, limit=3).get("flows", [])
        except LookupError:
            continue
        for f in flows:
            members = [sid] + [s["target"] for s in f.get("steps", [])]
            members = list(dict.fromkeys(members))
            if len(members) >= 3:
                out.append({"id": f"hyper:flow:{sid}->{members[-1]}", "kind": "flow",
                            "label": f"{sid.split(':', 1)[-1]} → {members[-1].split(':', 1)[-1]}", "members": members,
                            "weight": f.get("trust"), "evidence": f.get("basis") or "static"})
    return out[:limit]
