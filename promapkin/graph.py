"""Evidence-aware graph traversal: lookup, neighbourhoods, impact analysis, tracing and data flow.

Edges are not equally trustworthy. Each traversal step is weighted by the edge's evidence
class, freshness, confidence and independent corroboration, so results answer both
"what is related?" and "why do we believe it, and how much?".
"""

from __future__ import annotations

import heapq
import math
import re
from collections import defaultdict
from typing import Any, Iterable

from promapkin import evidence as ev
from promapkin.store import Store

# Edges along which a change propagates to the *source* (src depends on dst).
DEPENDENCY_EDGES = {"calls", "instantiates", "imports", "inherits", "implements", "references", "depends_on",
                    "routes_to", "verifies", "documents", "reads", "writes", "invokes", "overrides", "raises",
                    "affects", "constrains", "implements_pattern", "exposes"}
# Forward execution / data movement.
FLOW_EDGES = {"calls", "routes_to", "instantiates", "reads", "writes", "invokes", "depends_on", "dispatches"}
STRUCTURE_EDGES = {"contains", "defines"}
HUB_PREFIXES = {"package", "resource", "config", "dir", "repo", "file", "manifest", "event"}
STRUCTURAL_TRACE_EDGES = {"calls", "routes_to", "instantiates", "reads", "writes", "invokes", "depends_on",
                          "imports", "overrides", "exposes", "contains", "defines", "references", "inherits",
                          "implements", "produces", "consumes", "raises"}
SINK_KINDS = {"resource", "config", "package", "service", "table", "cloud_resource", "config_source"}


class NotFound(LookupError):
    pass


class Graph:
    def __init__(self, store: Store):
        self.store = store

    # ---------------------------------------------------------------- lookup
    def resolve(self, ref: str, fuzzy: bool = True) -> str:
        """Accept an id, qualified name, short name, 'Class::method', 'api:search', or a file path.

        With fuzzy=False, free text is not guessed into a node through full-text search."""
        ref = ref.strip()
        if self.store.node(ref):
            return ref
        norm = ref.replace("::", ".")
        for cand in (norm, f"file:{ref}", f"module:{ref}", f"api:{ref}", f"package:{ref.lower()}",
                     f"resource:{ref}", f"config:{ref}"):
            if self.store.node(cand):
                return cand
        hits = self.store.find_nodes_by_name(norm)
        if not hits:
            rows = self.store.conn.execute(
                "SELECT * FROM nodes WHERE qualname LIKE ? OR id LIKE ? ORDER BY length(id) LIMIT 20",
                (f"%.{norm}", f"%{norm}")).fetchall()
            hits = [dict(r) for r in rows]
        if ref.startswith("api:") and not hits:
            frag = ref[4:].lower()
            rows = self.store.conn.execute("SELECT * FROM nodes WHERE kind='endpoint' AND lower(name) LIKE ?",
                                           (f"%{frag}%",)).fetchall()
            hits = [dict(r) for r in rows]
        if not hits:
            found = self.store.search(ref, limit=1) if fuzzy else []
            if found:
                return found[0][0]
            raise NotFound(f"no graph node matches {ref!r}")
        order = {"class": 0, "function": 1, "method": 2, "module": 3, "endpoint": 4}

        def rank(n: dict[str, Any]) -> tuple:
            from promapkin.extract import is_test_path
            qual = n.get("qualname") or ""
            nesting = qual.count(".") - (n.get("id", "").split(":")[1].count(".") if n.get("id", "").count(":") >= 2
                                         else 0)
            # Prefer production code, then type-like kinds, then top-level over nested definitions.
            return (is_test_path(n.get("file") or ""), order.get(n["kind"], 9), nesting, len(n["id"]))

        hits.sort(key=rank)
        return hits[0]["id"]

    def edge_trust(self, e: dict[str, Any], corroborations: int = 0) -> float:
        return round(ev.weight(e["evidence"], e["status"], corroborations) * float(e["confidence"] or 1.0), 4)

    def describe_edge(self, e: dict[str, Any], corroborations: int = 0) -> dict[str, Any]:
        out = {"source": e["src"], "type": e["type"], "target": e["dst"], "evidence": e["evidence"],
               "evidence_class": ev.classify(e["evidence"]), "trust": self.edge_trust(e, corroborations),
               "status": e["status"]}
        if e.get("file"):
            out["at"] = f"{e['file']}:{e['line']}" if e.get("line") else e["file"]
        if e["attrs"].get("claim"):
            out["claim"] = e["attrs"]["claim"]
        if e["attrs"].get("classification"):
            out["classification"] = e["attrs"]["classification"]
        return out

    def _adjacent(self, node: str, direction: str, types: Iterable[str] | None) -> list[tuple[str, dict, float]]:
        """(neighbour, edge, trust) with parallel edges merged as corroboration."""
        edges = self.store.edges(src=node, types=types) if direction == "out" else self.store.edges(dst=node,
                                                                                                    types=types)
        groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for e in edges:
            other = e["dst"] if direction == "out" else e["src"]
            groups[(other, e["type"])].append(e)
        out = []
        for (other, _), es in groups.items():
            es.sort(key=lambda e: -ev.weight(e["evidence"], e["status"]))
            independent = len({ev.classify(e["evidence"]) for e in es}) - 1
            out.append((other, es[0], self.edge_trust(es[0], independent)))
        return out

    def node_summary(self, node_id: str) -> dict[str, Any]:
        n = self.store.node(node_id)
        if not n:
            raise NotFound(node_id)
        s = {"id": n["id"], "kind": n["kind"], "name": n["name"], "evidence": n["evidence"],
             "evidence_class": ev.classify(n["evidence"]), "status": n["status"]}
        if n["file"]:
            s["location"] = f"{n['file']}:{n['line_start']}" if n["line_start"] else n["file"]
        if n["qualname"]:
            s["qualname"] = n["qualname"]
        if n["text"]:
            s["summary"] = n["text"][:240]
        return s

    def node_detail(self, node_id: str, limit: int = 40) -> dict[str, Any]:
        node_id = self.resolve(node_id)
        n = self.store.node(node_id)
        assert n is not None
        out_edges = [self.describe_edge(e, c) for e, c in self._merged(self.store.edges(src=node_id))]
        in_edges = [self.describe_edge(e, c) for e, c in self._merged(self.store.edges(dst=node_id))]
        out_edges.sort(key=lambda d: -d["trust"])
        in_edges.sort(key=lambda d: -d["trust"])
        return {"node": {**self.node_summary(node_id), "attrs": _trim_attrs(n["attrs"]), "revision": n["revision"],
                         "confidence": n["confidence"]},
                "outgoing": out_edges[:limit], "incoming": in_edges[:limit],
                "truncated": len(out_edges) > limit or len(in_edges) > limit}

    def _merged(self, edges: list[dict]) -> list[tuple[dict, int]]:
        groups: dict[tuple, list[dict]] = defaultdict(list)
        for e in edges:
            groups[(e["src"], e["dst"], e["type"])].append(e)
        out = []
        for es in groups.values():
            es.sort(key=lambda e: -ev.weight(e["evidence"], e["status"]))
            out.append((es[0], len({ev.classify(e["evidence"]) for e in es}) - 1))
        return out

    def neighbors(self, ref: str, direction: str = "both", types: Iterable[str] | None = None, depth: int = 1,
                  min_trust: float = 0.0, limit: int = 60) -> dict[str, Any]:
        start = self.resolve(ref)
        types = list(types) if types else None
        seen = {start: 1.0}
        frontier = [start]
        edges_out: list[dict] = []
        for _ in range(max(1, depth)):
            nxt = []
            for node in frontier:
                for d in (["out", "in"] if direction == "both" else [direction]):
                    for other, e, trust in self._adjacent(node, d, types):
                        score = seen[node] * trust
                        if trust < min_trust:
                            continue
                        edges_out.append(self.describe_edge(e))
                        if other not in seen or seen[other] < score:
                            seen[other] = score
                            nxt.append(other)
            frontier = nxt
        nodes = sorted(((n, s) for n, s in seen.items() if n != start), key=lambda t: -t[1])[:limit]
        return {"root": self.node_summary(start),
                "nodes": [{**self._safe_summary(n), "path_trust": round(s, 3)} for n, s in nodes],
                "edges": _dedupe(edges_out)[:limit * 2], "truncated": len(seen) - 1 > limit}

    def _safe_summary(self, node_id: str) -> dict[str, Any]:
        try:
            return self.node_summary(node_id)
        except NotFound:
            return {"id": node_id, "kind": "missing"}

    # ---------------------------------------------------------------- impact analysis
    def impact(self, ref: str, depth: int = 4, min_trust: float = 0.15, limit: int = 80) -> dict[str, Any]:
        """Everything that may be affected if `ref` changes, ranked by evidence-weighted path trust."""
        start = self.resolve(ref)
        seeds = {start: 1.0}
        node = self.store.node(start)
        # A class or module change implies changes to its members.
        if node and node["kind"] in ("class", "module", "file"):
            for e in self.store.edges(src=start, types=["contains", "defines"]):
                seeds.setdefault(e["dst"], 0.95)
        best: dict[str, float] = dict(seeds)
        via: dict[str, dict] = {}
        heap = [(-s, n, 0) for n, s in seeds.items()]
        heapq.heapify(heap)
        while heap:
            neg, cur, d = heapq.heappop(heap)
            score = -neg
            if score < best.get(cur, 0) or d >= depth:
                continue
            nbrs = [(o, e, t) for o, e, t in self._adjacent(cur, "in", DEPENDENCY_EDGES)]
            # Overriding methods share their base method's contract; callers of the base reach them too.
            nbrs += [(o, e, t) for o, e, t in self._adjacent(cur, "out", ["overrides"])]
            for other, e, trust in nbrs:
                s = score * trust * 0.9
                if s < min_trust or s <= best.get(other, 0):
                    continue
                best[other] = s
                via[other] = self.describe_edge(e)
                heapq.heappush(heap, (-s, other, d + 1))
                # Impact climbs to the enclosing symbol: a method's caller lives in a class/module.
        groups: dict[str, list] = defaultdict(list)
        for nid, s in sorted(best.items(), key=lambda t: -t[1]):
            if nid in seeds:
                continue
            summ = self._safe_summary(nid)
            kind = summ.get("kind", "")
            group = ("tests" if kind == "test" or self._is_test(nid) else
                     "endpoints" if kind == "endpoint" else
                     "documentation" if kind in ("doc_section", "document", "decision", "code_note") else
                     "modules" if kind in ("module", "file") else
                     "knowledge" if kind in ("finding", "handoff", "roadmap_step", "pattern", "concept") else
                     "symbols")
            groups[group].append({**summ, "impact_score": round(s, 3), "via": via.get(nid)})
        for g in groups:
            groups[g] = groups[g][:limit]
        tests = self._tests_for(set(best))
        return {"target": self.node_summary(start), "affected": dict(groups),
                "tests_to_run": tests, "total_affected": sum(len(v) for v in groups.values()),
                "method": "reverse traversal over dependency edges, ranked by evidence-weighted path trust"}

    def _is_test(self, nid: str) -> bool:
        n = self.store.node(nid)
        return bool(n and n["attrs"].get("test"))

    def _tests_for(self, nodes: set[str]) -> list[dict[str, Any]]:
        tests: dict[str, float] = {}
        for nid in nodes:
            for e in self.store.edges(dst=nid, types=["verifies"]):
                tests[e["src"]] = max(tests.get(e["src"], 0), self.edge_trust(e))
        return [{**self._safe_summary(t), "trust": round(s, 3)} for t, s in sorted(tests.items(), key=lambda t: -t[1])]

    # ---------------------------------------------------------------- tracing
    def trace(self, src_ref: str, dst_ref: str, max_depth: int = 8, k: int = 3,
              types: Iterable[str] | None = None, exclude_evidence: Iterable[str] = (),
              mode: str = "auto") -> dict[str, Any]:
        """Most-trusted paths between two nodes (Dijkstra over -log(trust)).

        mode "flow" follows execution/data edges forward; "structural" also follows containment,
        references, inheritance and dependency edges forward; "related" ignores direction. "auto"
        tries them in that order and reports which one produced the answer."""
        src, dst = self.resolve(src_ref), self.resolve(dst_ref)
        exclude = set(exclude_evidence)
        modes = [mode] if mode != "auto" else (["flow"] if types else ["flow", "structural", "related"])
        paths: list[tuple[float, list[dict]]] = []
        used = modes[0]
        for m in modes:
            used = m
            allowed = set(types) if types else (FLOW_EDGES | {"imports", "overrides", "exposes"} if m == "flow"
                                                   else STRUCTURAL_TRACE_EDGES)
            paths = self._search(src, dst, allowed, exclude, max_depth, k, undirected=(m == "related"))
            if paths:
                break
        reports = [self._path_report(c, p) for c, p in paths]
        for r in reports:
            r["summary"] = _path_summary(r["steps"])
            r["mode"] = used
        return {"from": self.node_summary(src), "to": self.node_summary(dst), "paths": reports,
                "found": bool(paths), "mode": used if paths else None,
                **({} if paths else {"hint": "no path in flow, structural or undirected modes within "
                                              f"{max_depth} hops"})}

    def _search(self, src: str, dst: str, allowed: set[str], exclude: set[str], max_depth: int, k: int,
                undirected: bool = False) -> list[tuple[float, list[dict]]]:
        targets = {dst}
        dn = self.store.node(dst)
        if dn and dn["kind"] in ("class", "module", "file"):
            targets |= {e["dst"] for e in self.store.edges(src=dst, types=["contains", "defines"])}
        paths: list[tuple[float, list[dict]]] = []
        tie = 0
        heap: list[tuple[float, int, str, list[dict]]] = [(0.0, tie, src, [])]
        visits: dict[str, int] = defaultdict(int)
        expanded = 0
        while heap and len(paths) < k and expanded < 20000:
            cost, _, cur, path = heapq.heappop(heap)
            if cur in targets and path:
                key = tuple(s["target"] for s in path)
                if key not in {tuple(s["target"] for s in p) for _, p in paths}:
                    paths.append((cost, path))
                continue
            visits[cur] += 1
            if visits[cur] > k or len(path) >= max_depth:
                continue
            expanded += 1
            nbrs = self._forward(cur, allowed, exclude)
            if undirected:
                for o, e, t in self._adjacent(cur, "in", allowed):
                    if e["evidence"] in exclude or (o not in targets and o.split(":", 1)[0] in HUB_PREFIXES):
                        continue
                    # Walking an edge backwards: still evidence, but a weaker statement about flow.
                    nbrs.append((o, dict(e, reversed=True), round(t * 0.8, 4)))
            for other, e, trust in nbrs:
                if any(p["target"] == other for p in path) or other == src:
                    continue
                if undirected and other not in targets and other.split(":", 1)[0] in HUB_PREFIXES:
                    continue   # shared packages/resources would connect everything to everything
                step = self.describe_edge(e)
                if e.get("reversed"):
                    step = {**step, "source": cur, "target": other, "type": f"{e['type']} (reverse)"}
                step["trust"] = trust
                tie += 1
                heapq.heappush(heap, (cost - math.log(max(trust, 1e-6)), tie, other, path + [step]))
        return paths

    def _forward(self, node: str, allowed: set[str], exclude: set[str] | None = None) -> list[tuple[str, dict, float]]:
        if exclude:
            edges = [e for e in self.store.edges(src=node, types=allowed - {"overrides"})
                     if e["evidence"] not in exclude]
            out = [(e["dst"], e, self.edge_trust(e)) for e in edges]
        else:
            out = [(o, e, t) for o, e, t in self._adjacent(node, "out", allowed - {"overrides"})]
        # Dynamic dispatch: a call to a base method may execute any override (inferred, lower trust).
        for o, e, t in self._adjacent(node, "in", ["overrides"]):
            e2 = dict(e, src=node, dst=o, type="dispatches", evidence=ev.HEURISTIC)
            out.append((o, e2, round(t * 0.7, 4)))
        return out

    def _path_report(self, cost: float, steps: list[dict]) -> dict[str, Any]:
        classes = {s["evidence_class"] for s in steps}
        basis = "static" if classes <= {"observed"} else ("documented" if "inferred" not in classes else "inferred")
        return {"trust": round(math.exp(-cost), 4), "basis": basis, "length": len(steps),
                "nodes": [steps[0]["source"]] + [s["target"] for s in steps], "steps": steps}

    def dataflow(self, ref: str, max_depth: int = 7, limit: int = 30) -> dict[str, Any]:
        """Forward flows from an entry point/endpoint/function to sinks (databases, HTTP, files, config)."""
        start = self.resolve(ref)
        results = []
        stack: list[tuple[str, list[dict], float]] = [(start, [], 1.0)]
        explored = 0
        while stack and explored < 5000:
            cur, path, trust = stack.pop()
            explored += 1
            for other, e, t in self._forward(cur, FLOW_EDGES):
                if any(p["target"] == other for p in path) or other == start:
                    continue
                step = self.describe_edge(e)
                step["trust"] = t
                np_, nt = path + [step], trust * t
                kind = (self.store.node(other) or {}).get("kind")
                if kind in SINK_KINDS:
                    results.append((nt, np_))
                elif len(np_) < max_depth:
                    stack.append((other, np_, nt))
        results.sort(key=lambda r: -r[0])
        flows = []
        seen = set()
        for t, p in results:
            key = tuple((s["type"], s["target"]) for s in p)
            if key in seen:
                continue
            seen.add(key)
            rep = self._path_report(-math.log(max(t, 1e-9)), p)
            rep["sink"] = p[-1]["target"]
            rep["operation"] = p[-1]["type"]
            flows.append(rep)
        return {"from": self.node_summary(start), "flows": flows[:limit], "truncated": len(flows) > limit}

    # ---------------------------------------------------------------- dependencies
    def module_dependencies(self) -> dict[str, set[str]]:
        deps: dict[str, set[str]] = defaultdict(set)
        for src, dst in self.store.conn.execute(
                "SELECT src, dst FROM edges WHERE type='imports' AND src LIKE 'module:%' AND dst LIKE 'module:%' "
                "ORDER BY id"):
            deps[src].add(dst)
        return deps

    def dependency_report(self, ref: str | None = None, depth: int = 3) -> dict[str, Any]:
        deps = self.module_dependencies()
        cycles = find_cycles(deps)
        report: dict[str, Any] = {"cycles": [[c.removeprefix("module:") for c in cyc] for cyc in cycles]}
        if ref:
            start = self.resolve(ref)
            if not start.startswith("module:"):
                n = self.store.node(start)
                mod = next((e["src"] for e in self.store.edges(dst=start, types=["defines"])
                            if e["src"].startswith("module:")), None)
                if mod is None and n and n.get("file"):
                    mod = next((e["dst"] for e in self.store.edges(src=f"file:{n['file']}", types=["defines"])), None)
                start = mod or start
            report["module"] = start
            report["depends_on"] = _bfs(start, deps, depth)
            rev: dict[str, set[str]] = defaultdict(set)
            for a, bs in deps.items():
                for b in bs:
                    rev[b].add(a)
            report["depended_on_by"] = _bfs(start, rev, depth)
            report["external_packages"] = sorted({e["dst"] for e in self.store.edges(src=start, types=["imports"])
                                                  if e["dst"].startswith("package:")})
        else:
            fan_in: dict[str, int] = defaultdict(int)
            for a, bs in deps.items():
                for b in bs:
                    fan_in[b] += 1
            report["most_depended_on"] = [{"module": m.removeprefix("module:"), "fan_in": c}
                                          for m, c in sorted(fan_in.items(), key=lambda t: -t[1])[:15]]
            report["edges"] = sum(len(v) for v in deps.values())
        return report


def _bfs(start: str, adj: dict[str, set[str]], depth: int) -> list[dict[str, Any]]:
    seen = {start: 0}
    q = [start]
    while q:
        cur = q.pop(0)
        if seen[cur] >= depth:
            continue
        for n in sorted(adj.get(cur, ())):
            if n not in seen:
                seen[n] = seen[cur] + 1
                q.append(n)
    return [{"module": m.removeprefix("module:"), "distance": d} for m, d in sorted(seen.items(), key=lambda t: t[1])
            if m != start]


def find_cycles(adj: dict[str, set[str]]) -> list[list[str]]:
    """Strongly connected components with more than one node (Tarjan, iterative: include graphs such as the Linux
    kernel's are far deeper than Python's recursion limit). Components come out in the recursive algorithm's
    order."""
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    stack: list[str] = []
    on: set[str] = set()
    out: list[list[str]] = []
    counter = 0
    for root in list(adj):
        if root in index:
            continue
        index[root] = low[root] = counter
        counter += 1
        stack.append(root)
        on.add(root)
        work = [(root, iter(adj.get(root, ())))]
        while work:
            v, it = work[-1]
            advanced = False
            for w in it:
                if w not in index:
                    index[w] = low[w] = counter
                    counter += 1
                    stack.append(w)
                    on.add(w)
                    work.append((w, iter(adj.get(w, ()))))
                    advanced = True
                    break
                if w in on:
                    low[v] = min(low[v], index[w])
            if advanced:
                continue
            work.pop()
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[v])
            if low[v] == index[v]:
                comp = []
                while True:
                    w = stack.pop()
                    on.discard(w)
                    comp.append(w)
                    if w == v:
                        break
                if len(comp) > 1:
                    out.append(sorted(comp))
    return out


def _dedupe(edges: list[dict]) -> list[dict]:
    seen, out = set(), []
    for e in edges:
        key = (e["source"], e["type"], e["target"])
        if key not in seen:
            seen.add(key)
            out.append(e)
    return out


def _trim_attrs(attrs: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for k, v in attrs.items():
        if k == "imports" and isinstance(v, dict) and len(v) > 20:
            v = dict(list(v.items())[:20])
        if isinstance(v, str) and len(v) > 400:
            v = v[:400] + "…"
        out[k] = v
    return out


def _path_summary(steps: list[dict[str, Any]]) -> str:
    """'FastAPI --contains--> include_router --references--> APIRouter' (evidence in brackets)."""
    if not steps:
        return ""
    label = lambda nid: nid.split(":")[-1]  # noqa: E731
    out = label(steps[0]["source"])
    for st in steps:
        out += f" --{st['type']} [{st['evidence']}]--> {label(st['target'])}"
    return out


def short(node_id: str) -> str:
    return re.sub(r"^(symbol|module):", "", node_id)
