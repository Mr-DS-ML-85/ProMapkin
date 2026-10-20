"""Deterministic specialist agents.

Each specialist receives a bounded scope over extracted facts (never the whole repository
text) and returns a structured section whose claims carry evidence classes, sources and
confidence. ML-backed steps (layer and design-pattern classification, semantic clustering)
use the two-tier classifier; with ML disabled they use rules with static confidence.
LLM providers, when configured, add a synthesis on top of these grounded facts.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any

from promapkin import evidence as ev
from promapkin import gitutil
from promapkin.config import Workspace
from promapkin.graph import Graph, find_cycles
from promapkin.store import Store

LAYER_RANK = {"interface": 0, "application": 1, "integration": 2, "persistence": 2, "infrastructure": 2,
              "domain": 3, "shared": 4}


@dataclass
class Ctx:
    ws: Workspace
    store: Store
    graph: Graph
    rt: Any
    revision: str
    cache: dict[str, Any] = field(default_factory=dict)

    @property
    def ml(self) -> bool:
        return self.rt.classifier.mode != "heuristic"


def claim(statement: str, evidence: str, sources: list[str] | None = None, confidence: float | None = None,
          method: str = "", **extra: Any) -> dict[str, Any]:
    c = {"statement": statement, "evidence": evidence, "evidence_class": ev.classify(evidence),
         "confidence": round(confidence if confidence is not None else ev.WEIGHTS.get(evidence, 0.5), 3),
         "sources": [s for s in (sources or []) if s][:8]}
    if method:
        c["method"] = method
    c.update(extra)
    return c


def loc(n: dict[str, Any] | None) -> str:
    if not n:
        return ""
    if n.get("file") and n.get("line_start"):
        return f"{n['file']}:{n['line_start']}"
    return n.get("file") or n.get("id", "")


def module_of(store: Store, node_id: str) -> str | None:
    if node_id.startswith("module:"):
        return node_id
    if node_id.startswith("symbol:"):
        return "module:" + node_id.split(":")[1]
    n = store.node(node_id)
    if n and n.get("file"):
        for e in store.edges(src=f"file:{n['file']}", types=["defines"]):
            if e["dst"].startswith("module:"):
                return e["dst"]
    return None


def subsystem_of(path: str | None) -> str:
    if not path:
        return "(external)"
    parts = PurePosixPath(path).parts[:-1]
    return "/".join(parts[:2]) or "(root)"


def layer_of(mod: dict[str, Any] | None) -> tuple[str, float, str]:
    if not mod:
        return "unknown", 0.0, ""
    lay = mod["attrs"].get("layer") or {}
    return lay.get("label", "unknown"), float(lay.get("confidence", 0.0)), lay.get("tier", "")


# ===================================================================================== reconnaissance

def reconnaissance(ctx: Ctx) -> dict[str, Any]:
    s = ctx.store
    files = s.conn.execute("SELECT language, COUNT(*), SUM(size) FROM files GROUP BY language").fetchall()
    languages = {r[0]: {"files": r[1], "bytes": r[2]} for r in files}
    modules = s.nodes("module")
    tests = [m for m in modules if m["attrs"].get("test")]
    docs = s.nodes("document")
    decisions = s.nodes("decision")
    entry = _entry_points(ctx)
    top_dirs = Counter(PurePosixPath(r[0]).parts[0] if "/" in r[0] else "(root)"
                       for r in s.conn.execute("SELECT path FROM files"))
    conventions = []
    if any(m["file"].endswith(".py") for m in tests):
        conventions.append(claim("Python tests follow the test_*.py convention (pytest/unittest style)", ev.AST,
                                 [tests[0]["file"]]))
    if s.node("manifest:pyproject.toml"):
        conventions.append(claim("Packaging is declared in pyproject.toml", ev.MANIFEST, ["pyproject.toml"]))
    if s.node("manifest:package.json"):
        conventions.append(claim("JavaScript/TypeScript packaging via package.json", ev.MANIFEST, ["package.json"]))
    missing = []
    if not docs:
        missing.append("No Markdown documentation was found.")
    if not decisions:
        missing.append("No architecture decision records (ADRs); design rationale must be inferred or asked.")
    if not tests:
        missing.append("No test files were detected; impact analysis cannot recommend tests.")
    untested = _untested_modules(ctx)
    if untested:
        missing.append(f"{len(untested)} non-test modules have no test that imports or calls them.")
    suspicious = [n for n in s.nodes(where="json_extract(attrs, '$.untrusted_instructions[0]') IS NOT NULL")]
    commits = gitutil.recent_commits(ctx.ws.root, 10)
    return {
        "title": "Repository reconnaissance",
        "revision": ctx.revision,
        "languages": languages,
        "top_level": dict(top_dirs.most_common(12)),
        "counts": s.counts(),
        "entry_points": entry,
        "conventions": conventions,
        "missing_context": missing,
        "untested_modules": untested[:25],
        "security_notices": [claim(f"Embedded instruction-like text in {n['file']} is treated as untrusted data",
                                   ev.HEURISTIC, [n["file"]], 0.6, snippets=n["attrs"]["untrusted_instructions"][:2])
                             for n in suspicious[:10]],
        "recent_commits": commits,
        "ml": ctx.rt.describe(),
    }


def _entry_points(ctx: Ctx) -> list[dict[str, Any]]:
    s = ctx.store
    out = []
    for m in s.nodes("module"):
        if m["attrs"].get("entrypoint") and not m["attrs"].get("test"):
            out.append(claim(f"`{m['name']}` is an executable entry point", ev.AST if m["evidence"] == ev.AST
                             else ev.LEXICAL, [m["file"]], kind="module", node=m["id"]))
    for e in s.nodes("entrypoint"):
        out.append(claim(f"Console/bin entry `{e['name']}` → {e['attrs'].get('target')}", ev.MANIFEST, [e["file"]],
                         kind="script", node=e["id"]))
    for f in s.nodes(where="kind IN ('function','method') AND json_extract(attrs, '$.cli_command') = 1"):
        out.append(claim(f"CLI command `{f['name']}`", ev.AST, [loc(f)], kind="cli", node=f["id"]))
    endpoints = s.nodes("endpoint")
    if endpoints:
        out.append(claim(f"{len(endpoints)} HTTP endpoints exposed", endpoints[0]["evidence"],
                         [loc(e) for e in endpoints[:5]], kind="http"))
    return out


def _untested_modules(ctx: Ctx) -> list[str]:
    s = ctx.store
    tested = {e["dst"] for e in s.edges(types=["verifies"])}
    tested_mods = {module_of(s, t) for t in tested}
    out = []
    for m in s.nodes("module"):
        if m["attrs"].get("test") or (m["attrs"].get("loc") or 0) < 5:
            continue
        if m["id"] not in tested_mods:
            out.append(m["name"])
    return out


# ===================================================================================== architecture

def architecture(ctx: Ctx) -> dict[str, Any]:
    s, g = ctx.store, ctx.graph
    from promapkin.ml import graph_ml

    modules = {m["id"]: m for m in s.nodes("module")}
    subsystems: dict[str, dict[str, Any]] = {}
    for mid, m in modules.items():
        sub = subsystem_of(m["file"])
        entry = subsystems.setdefault(sub, {"name": sub, "modules": [], "layers": Counter(), "symbols": 0,
                                            "loc": 0, "test": True})
        entry["modules"].append(m["name"])
        layer, conf, _ = layer_of(m)
        entry["layers"][layer] += 1
        entry["loc"] += m["attrs"].get("loc") or 0
        entry["symbols"] += len(s.edges(src=mid, types=["defines"]))
        entry["test"] = entry["test"] and bool(m["attrs"].get("test"))

    # Subsystem dependency edges aggregated from module imports.
    sub_deps: dict[tuple[str, str], int] = Counter()
    layer_edges: list[dict[str, Any]] = []
    violations = []
    for e in s.edges(types=["imports"]):
        a, b = modules.get(e["src"]), modules.get(e["dst"])
        if not a or not b:
            continue
        sa, sb = subsystem_of(a["file"]), subsystem_of(b["file"])
        if sa != sb:
            sub_deps[(sa, sb)] += 1
        la, ca, _ = layer_of(a)
        lb, cb, _ = layer_of(b)
        if a["attrs"].get("test") or b["attrs"].get("test"):
            continue
        layer_edges.append({"from": la, "to": lb})
        if la in LAYER_RANK and lb in LAYER_RANK and LAYER_RANK[la] > LAYER_RANK[lb] and lb != "shared":
            violations.append(claim(
                f"`{a['name']}` ({la}) imports `{b['name']}` ({lb}): dependency points against the layer order",
                ev.AST, [f"{e['file']}:{e['line']}"], round(min(ca, cb) * 0.9, 3),
                method="layer classification + import edge"))

    layers = []
    for mid, m in modules.items():
        if m["attrs"].get("test"):
            continue
        layer, conf, tier = layer_of(m)
        layers.append({"module": m["name"], "layer": layer, "confidence": conf, "tier": tier,
                       "alternatives": (m["attrs"].get("layer") or {}).get("alternatives", [])})

    integrations = []
    for p in s.nodes("package"):
        users = [e["src"] for e in s.edges(dst=p["id"], types=["imports"])]
        if users and not p["attrs"].get("stdlib"):
            integrations.append(claim(f"External package `{p['name']}` used by {len(users)} module(s)", ev.AST,
                                      [loc(modules.get(u)) for u in users[:4]], package=p["name"],
                                      used_by=[u.removeprefix("module:") for u in users[:10]]))
    for r in s.nodes("resource"):
        users = s.edges(dst=r["id"])
        if users:
            evs = Counter(u["evidence"] for u in users).most_common(1)[0][0]
            integrations.append(claim(
                f"{r['name'].title()} access from {len({u['src'] for u in users})} symbol(s)", evs,
                [f"{u['file']}:{u['line']}" for u in users[:4] if u.get("file")],
                round(sum(g.edge_trust(u) for u in users) / len(users), 3), resource=r["name"],
                operations=dict(Counter(u["type"] for u in users))))

    styles = _styles(ctx, layers, violations)
    from promapkin import infra_view
    infra_claims, data_claims = infra_view.claims(s)
    clusters = graph_ml.cluster_modules(s, _graph_model(ctx) if ctx.ml else None)
    cycles = find_cycles(g.module_dependencies())

    return {
        "title": "Architecture map",
        "subsystems": [{**v, "layers": dict(v["layers"]), "dominant_layer": v["layers"].most_common(1)[0][0]}
                       for v in sorted(subsystems.values(), key=lambda v: v["name"])],
        "subsystem_dependencies": [{"from": a, "to": b, "imports": n} for (a, b), n in sorted(sub_deps.items())],
        "layers": sorted(layers, key=lambda l: (LAYER_RANK.get(l["layer"], 9), l["module"])),
        "layer_violations": violations[:30],
        "entry_points": _entry_points(ctx),
        "external_integrations": integrations,
        "configuration": [claim(f"Configuration key `{c['name']}` read by "
                                f"{len(s.edges(dst=c['id'], types=['reads']))} symbol(s)", ev.AST,
                                [f"{e['file']}:{e['line']}" for e in s.edges(dst=c["id"])[:3] if e.get("file")])
                          for c in s.nodes("config")],
        "styles": styles,
        "infrastructure": infra_claims,
        "data_model": data_claims,
        "semantic_components": clusters,
        "dependency_cycles": [[c.removeprefix("module:") for c in cyc] for cyc in cycles],
    }


def _graph_model(ctx: Ctx) -> str:
    from promapkin.ml.graph_ml import graph_model_name
    return graph_model_name(ctx.rt.embedder.name)


def _styles(ctx: Ctx, layers: list[dict], violations: list[dict]) -> list[dict[str, Any]]:
    s = ctx.store
    out = []
    present = Counter(l["layer"] for l in layers)
    known = [l for l in present if l in LAYER_RANK]
    if len(known) >= 3:
        conf = 0.75 if not violations else max(0.35, 0.75 - 0.08 * len(violations))
        out.append(claim(f"Layered architecture ({', '.join(sorted(known, key=LAYER_RANK.get))})", ev.HEURISTIC,
                         [], conf, method="layer classification of modules + import direction",
                         note=f"{len(violations)} layer violation(s)"))
    if s.nodes("endpoint"):
        frameworks = {p["name"] for p in s.nodes("package")} & {"fastapi", "flask", "django", "express", "koa",
                                                                 "starlette", "aiohttp", "fastify", "@nestjs/core"}
        out.append(claim(f"HTTP service{(' using ' + ', '.join(sorted(frameworks))) if frameworks else ''}",
                         ev.AST, [loc(e) for e in s.nodes("endpoint")[:3]]))
    if any(e["attrs"].get("cli_command") for e in s.nodes("function")) or s.nodes("entrypoint"):
        out.append(claim("Command-line application", ev.MANIFEST, [n.get("file") for n in s.nodes("entrypoint")][:3]))
    injected = s.edges(types=["depends_on"], where="json_extract(attrs, '$.injected') = 1")
    if len(injected) >= 2:
        out.append(claim("Constructor dependency injection is used to compose components", ev.AST,
                         [f"{e['file']}:{e['line']}" for e in injected[:4]], 0.8))
    services = s.nodes("service")
    if len(services) >= 2:
        out.append(claim(f"Multi-service deployment ({len(services)} compose services)", ev.MANIFEST,
                         [services[0]["file"]]))
    return out


# ===================================================================================== data flow

def dataflow(ctx: Ctx, limit: int = 25) -> dict[str, Any]:
    s, g = ctx.store, ctx.graph
    starts = [e["id"] for e in s.nodes("endpoint")]
    starts += [e["node"] for e in _entry_points(ctx) if e.get("node") and e.get("kind") in ("module", "cli")]
    if not starts:   # libraries: start from public functions in the interface/application layers
        for m in s.nodes("module"):
            if layer_of(m)[0] in ("interface", "application"):
                starts += [e["dst"] for e in s.edges(src=m["id"], types=["defines"])][:3]
    flows = []
    for st in starts[:limit]:
        try:
            report = g.dataflow(st, max_depth=7, limit=12)
        except LookupError:
            continue
        for f in report["flows"]:
            flows.append({"entry": st, "sink": f["sink"], "operation": f["operation"], "basis": f["basis"],
                          "trust": f["trust"], "path": f["nodes"],
                          "evidence": [{"edge": f"{x['source']} -{x['type']}-> {x['target']}",
                                        "evidence": x["evidence"], "at": x.get("at")} for x in f["steps"]]})
    resources = []
    for r in s.nodes("resource"):
        ops = s.edges(dst=r["id"])
        resources.append({"resource": r["name"], "readers": sorted({o["src"] for o in ops if o["type"] == "reads"}),
                          "writers": sorted({o["src"] for o in ops if o["type"] == "writes"}),
                          "invokers": sorted({o["src"] for o in ops if o["type"] == "invokes"}),
                          "classified_by": dict(Counter(o["evidence"] for o in ops))})
    return {"title": "Data-flow and execution map", "flows": flows,
            "resources": resources,
            "legend": {"static": "every step observed in source (AST/lexical/resolved imports)",
                       "documented": "includes steps asserted by documentation",
                       "inferred": "includes heuristic/ML/name-match steps (e.g. resource classification, dispatch)",
                       "runtime": "not available: Promapkin never executes repository code"}}


# ===================================================================================== API atlas

def api_atlas(ctx: Ctx) -> dict[str, Any]:
    s, g = ctx.store, ctx.graph
    endpoints = []
    for e in s.nodes("endpoint"):
        handlers = [x["dst"] for x in s.edges(src=e["id"], types=["routes_to"])]
        h = s.node(handlers[0]) if handlers else None
        auth = e["attrs"].get("auth") or (h["attrs"].get("auth") if h else []) or []
        try:
            imp = g.impact(e["id"], depth=2)
            dependents = imp["total_affected"]
        except LookupError:
            dependents = 0
        downstream = g.dataflow(e["id"], max_depth=6, limit=10)["flows"] if h else []
        endpoints.append({
            "endpoint": e["name"], "id": e["id"], "handler": handlers[0] if handlers else None,
            "location": loc(h) or loc(e), "params": (h or {}).get("attrs", {}).get("params", []),
            "returns": (h or {}).get("attrs", {}).get("returns"),
            "auth": auth, "auth_boundary": "protected" if auth else "no auth detected",
            "touches": sorted({f["sink"] for f in downstream}), "dependents": dependents,
            "evidence": e["evidence"], "evidence_class": ev.classify(e["evidence"]),
        })
    public = []
    for m in s.nodes("module"):
        if m["attrs"].get("test"):
            continue
        layer = layer_of(m)[0]
        exports = m["attrs"].get("exports")
        for d in s.edges(src=m["id"], types=["defines"]):
            n = s.node(d["dst"])
            if not n or n["kind"] not in ("class", "function") or not n["attrs"].get("public", True) \
                    or n["name"].startswith("_"):
                continue
            if exports is not None and n["name"] not in exports:
                continue
            fan_in = len({x["src"] for x in s.edges(dst=n["id"], types=["calls", "instantiates", "depends_on",
                                                                          "inherits"])})
            if fan_in or layer in ("interface", "application") or exports:
                public.append({"symbol": n["qualname"], "kind": n["kind"], "location": loc(n), "fan_in": fan_in,
                               "signature": _signature(n), "layer": layer})
    public.sort(key=lambda p: -p["fan_in"])
    packages = []
    for p in s.nodes("package"):
        if p["attrs"].get("stdlib"):
            continue
        decl = s.edges(dst=p["id"], types=["depends_on"])
        used = s.edges(dst=p["id"], types=["imports"])
        packages.append({"package": p["name"], "declared": [f"{d['src'].removeprefix('manifest:')} "
                                                             f"{d['attrs'].get('version', '')}".strip() for d in decl],
                         "imported_by": len({u["src"] for u in used}),
                         "status": "declared+used" if decl and used else ("declared, unused in scanned code" if decl
                                                                          else "imported but not declared")})
    return {"title": "API and dependency atlas", "endpoints": endpoints, "public_interfaces": public[:60],
            "packages": sorted(packages, key=lambda p: (-p["imported_by"], p["package"])),
            "change_impact_hotspots": [{"symbol": p["symbol"], "fan_in": p["fan_in"]} for p in public[:10]
                                       if p["fan_in"] > 1]}


def _signature(n: dict[str, Any]) -> str:
    params = n["attrs"].get("params") or []
    args = ", ".join(p["name"] + (f": {p['type']}" if p.get("type") else "") for p in params)
    ret = n["attrs"].get("returns")
    return f"{n['name']}({args})" + (f" -> {ret}" if ret else "")


# ===================================================================================== patterns

def describe_class(ctx: Ctx, n: dict[str, Any]) -> str:
    """Structural feature description consumed by the design-pattern classifier (rules and ML)."""
    s = ctx.store
    subclasses = len(s.edges(dst=n["id"], types=["inherits", "implements"]))
    injects = [e["dst"].rsplit(":", 1)[-1] for e in s.edges(src=f"{n['id']}.__init__", types=["depends_on"])]
    members = [f"{n['id']}.{m}" for m in n["attrs"].get("methods", [])]
    touches = sorted({e["dst"].removeprefix("resource:") for m in members + [n["id"]]
                      for e in s.edges(src=m, types=["reads", "writes", "invokes"]) if e["dst"].startswith("resource:")})
    instantiates = len({e["dst"] for m in members + [n["id"]] for e in s.edges(src=m, types=["instantiates"])})
    methods = n["attrs"].get("methods", []) + n["attrs"].get("class_attrs", [])
    return (f"class: {n['name']}; methods: {' '.join(methods) or 'none'}; subclasses: {subclasses}; "
            f"abstract: {'yes' if n['attrs'].get('abstract') else 'no'}; injects: {' '.join(injects) or 'none'}; "
            f"touches: {' '.join(touches) or 'none'}; instantiates: {instantiates}; "
            f"bases: {' '.join(n['attrs'].get('bases', [])) or 'none'}; doc: {(n['text'] or '')[:120]}")


def pattern_candidates(ctx: Ctx) -> tuple[list[dict[str, Any]], list[str]]:
    """Non-test classes and their structural descriptions (read-only)."""
    s = ctx.store
    candidates = []
    for n in s.nodes("class"):
        mod = s.node(module_of(s, n["id"]) or "") or {}
        if n["attrs"].get("test") or mod.get("attrs", {}).get("test"):
            continue
        candidates.append(n)
    return candidates, [describe_class(ctx, n) for n in candidates]


def warm_patterns(ctx: Ctx) -> None:
    """--llm: classify the candidates before the write transaction, so the specialist itself only reads
    cached answers while it holds the lock."""
    llm = getattr(ctx.ws, "llm", None)
    if llm is not None:
        ctx.rt.classifier.classify_many("design_pattern", pattern_candidates(ctx)[1], llm)


def patterns(ctx: Ctx) -> dict[str, Any]:
    s = ctx.store
    clf = ctx.rt.classifier
    s.delete_nodes("kind='pattern' AND evidence IN (?, ?, ?)", (ev.HEURISTIC, ev.ML_CLASSIFIER, ev.LLM_INFERENCE))
    found = []
    candidates, descs = pattern_candidates(ctx)
    llm = getattr(ctx.ws, "llm", None)
    for n, desc, c in zip(candidates, descs, clf.classify_many("design_pattern", descs, llm)):
        if c.label in ("none", "unknown") or c.confidence < 0.45:
            continue
        roles = _pattern_roles(ctx, c.label, n)
        evidence = (ev.LLM_INFERENCE if c.tier == "llm" else
                    ev.HEURISTIC if c.tier == "heuristic" or c.tier.startswith("heuristic>") else ev.ML_CLASSIFIER)
        pid = f"pattern:{c.label}:{n['qualname']}"
        s.upsert_node(pid, "pattern", f"{c.label.replace('_', ' ').title()} — {n['name']}", evidence=evidence,
                      confidence=c.confidence, revision=ctx.revision, file=n["file"], line_start=n["line_start"],
                      text=f"{c.label} pattern anchored at {n['qualname']}. Intent is not established by code alone.",
                      attrs={"pattern": c.label, "anchor": n["id"], "classification": c.as_dict(), "roles": roles,
                             "features": desc})
        for role, members in roles.items():
            for m in members:
                if s.node(m):
                    s.add_edge(m, pid, "implements_pattern", evidence=evidence, confidence=c.confidence,
                               revision=ctx.revision, attrs={"role": role})
        found.append({
            "pattern": c.label, "anchor": n["qualname"], "location": loc(n), "roles": {
                k: [x.split(":")[-1] for x in v] for k, v in roles.items()},
            **claim(f"{n['name']} likely implements the {c.label.replace('_', ' ')} pattern", evidence, [loc(n)],
                    c.confidence, method=f"design-pattern classifier ({c.tier})",
                    alternatives=c.alternatives, agreement=c.agreement),
            "intent": "unknown — code demonstrates structure, not why it was chosen",
        })
    composition_roots = []
    for f in s.nodes("function"):
        inst = {e["dst"] for e in s.edges(src=f["id"], types=["instantiates"])}
        if len(inst) >= 2 and not f["attrs"].get("test"):
            composition_roots.append(claim(
                f"`{f['qualname']}` wires {len(inst)} components together (composition root / factory function)",
                ev.AST, [loc(f)], 0.7, instantiates=[i.split(":")[-1] for i in sorted(inst)]))
    found.sort(key=lambda p: -p["confidence"])
    return {"title": "Design patterns", "patterns": found, "composition_roots": composition_roots,
            "classifier": {"mode": "llm" if llm else clf.mode, "model": llm.label if llm else clf.model_name},
            "note": "Patterns are recognised from structure. Why an author chose them is unknown unless a document "
                    "says so (see Decisions)."}


def _pattern_roles(ctx: Ctx, label: str, n: dict[str, Any]) -> dict[str, list[str]]:
    roles = _raw_pattern_roles(ctx, label, n)
    # Tests exercise patterns but are not participants; keep roles readable.
    return {k: [m for m in v if not _is_test_node(ctx, m)][:8] for k, v in roles.items()}


def _is_test_node(ctx: Ctx, nid: str) -> bool:
    mod = module_of(ctx.store, nid)
    node = ctx.store.node(mod) if mod else None
    return bool(node and node["attrs"].get("test"))


def _raw_pattern_roles(ctx: Ctx, label: str, n: dict[str, Any]) -> dict[str, list[str]]:
    s = ctx.store
    roles: dict[str, list[str]] = {"anchor": [n["id"]]}
    if label == "strategy":
        roles["concrete_strategies"] = [e["src"] for e in s.edges(dst=n["id"], types=["inherits", "implements"])]
        roles["context"] = sorted({e["src"].rsplit(".", 1)[0] for e in s.edges(dst=n["id"], types=["depends_on"])})
    elif label in ("adapter", "repository"):
        members = [f"{n['id']}.{m}" for m in n["attrs"].get("methods", [])]
        roles["adaptee"] = sorted({e["dst"] for m in members for e in s.edges(src=m, types=["reads", "writes",
                                                                                         "invokes"])})
        roles["clients"] = sorted({e["src"].rsplit(".", 1)[0] for m in members + [n["id"]]
                                   for e in s.edges(dst=m, types=["calls", "depends_on", "instantiates"])})[:10]
    elif label in ("dependency_injection", "facade"):
        roles["injected"] = [e["dst"] for e in s.edges(src=f"{n['id']}.__init__", types=["depends_on"])]
        roles["composers"] = [e["src"] for e in s.edges(dst=n["id"], types=["instantiates"])]
    return roles


# ===================================================================================== decisions

def decisions(ctx: Ctx, pattern_section: dict[str, Any] | None = None) -> dict[str, Any]:
    s = ctx.store
    records = []
    for d in s.nodes("decision"):
        if d["evidence"] != ev.DOCUMENTATION:
            continue
        refs = [e["dst"] for e in s.edges(src=d["id"], types=["documents"])]
        records.append({"decision": d["name"], "status": d["attrs"].get("status"), "origin": "documented (ADR)",
                        "context": d["attrs"].get("context"), "choice": d["attrs"].get("decision"),
                        "consequences": d["attrs"].get("consequences"), "alternatives": d["attrs"].get("alternatives"),
                        "affects": refs, **claim(d["name"], ev.DOCUMENTATION, [d["file"]], 0.85),
                        "verified_references": [r for r in refs if s.node(r)]})
    rationale_notes = []
    for n in s.nodes("code_note"):
        if n["attrs"].get("category") == "rationale":
            rationale_notes.append({**claim(n["text"][:300], ev.DOCUMENTATION, [loc(n)], 0.7,
                                            method="code comment", classification=n["attrs"].get("classification")),
                                    "origin": "documented in code"})
    inferred = []
    for p in (pattern_section or {}).get("patterns", [])[:20]:
        documented = [r for r in records if any(p["anchor"].endswith(a.split(":")[-1]) for a in r["affects"])]
        inferred.append({
            **claim(f"{p['anchor'].split('.')[-1]} is structured as a {p['pattern'].replace('_', ' ')}", p["evidence"],
                    p["sources"], round(p["confidence"] * 0.8, 3), method="inferred from design-pattern detection"),
            "origin": "inferred",
            "rationale": (f"see {documented[0]['decision']}" if documented else
                          "unknown — no document explains why; treat as an observation, not a historical decision"),
        })
    for st in architecture_styles_cached(ctx):
        inferred.append({**st, "origin": "inferred", "rationale": "unknown"})
    return {"title": "Design decisions", "documented": records, "code_rationale": rationale_notes[:40],
            "inferred": inferred,
            "policy": "Documented decisions cite their source. Inferred decisions describe structure only; their "
                      "rationale is reported as unknown unless a document or comment states it."}


def architecture_styles_cached(ctx: Ctx) -> list[dict[str, Any]]:
    arch = ctx.cache.get("architecture")
    return arch["styles"] if arch else []


# ===================================================================================== risks

def risks(ctx: Ctx) -> dict[str, Any]:
    """Risk register. Scores are a static, explainable weighting of structural signals."""
    s = ctx.store
    churn = gitutil.file_churn(ctx.ws.root)
    tested = {e["dst"] for e in s.edges(types=["verifies"])}
    items = []
    for n in s.nodes(where="kind IN ('function','method','class')"):
        if n["attrs"].get("test") or (n["file"] and ("test" in n["file"].split("/")[0])):
            continue
        fan_in = len({e["src"] for e in s.edges(dst=n["id"], types=["calls", "instantiates", "depends_on",
                                                                      "inherits", "routes_to"])})
        complexity = n["attrs"].get("complexity") or 1
        writes = bool(s.edges(src=n["id"], types=["writes"]))
        untested = n["id"] not in tested and not any(t.startswith(n["id"] + ".") for t in tested)
        score = 0.25 * min(fan_in / 5, 1) + 0.2 * min(complexity / 15, 1) + 0.25 * untested + 0.15 * writes + \
            0.15 * min(churn.get(n["file"] or "", 0) / 10, 1)
        if score >= 0.4:
            reasons = [r for r, on in (("high fan-in", fan_in >= 3), ("complex", complexity >= 10),
                                       ("no test reaches it", untested), ("writes external state", writes),
                                       ("frequently changed", churn.get(n["file"] or "", 0) >= 5)) if on]
            items.append({"symbol": n["qualname"], "location": loc(n), "score": round(score, 3), "reasons": reasons,
                          "fan_in": fan_in, "complexity": complexity, "evidence": ev.HEURISTIC,
                          "evidence_class": "inferred", "method": "static weighted risk score"})
    items.sort(key=lambda i: -i["score"])
    arch = ctx.cache.get("architecture") or {}
    structural = []
    for cyc in arch.get("dependency_cycles", []):
        structural.append(claim(f"Import cycle: {' ↔ '.join(cyc)}", ev.AST, [], 1.0))
    for v in arch.get("layer_violations", [])[:10]:
        structural.append(v)
    for n in s.nodes("endpoint"):
        if not n["attrs"].get("auth") and n["attrs"].get("method") in ("POST", "PUT", "PATCH", "DELETE"):
            structural.append(claim(f"Mutating endpoint {n['name']} has no detected auth guard", ev.HEURISTIC,
                                    [loc(n)], 0.5, method="decorator/dependency name heuristics"))
    todos = [{"note": n["text"][:200], "location": loc(n)} for n in s.nodes("code_note")
             if n["attrs"].get("category") in ("todo", "warning")]
    return {"title": "Risk register", "symbols": items[:30], "structural": structural, "open_notes": todos[:30]}


# ===================================================================================== verification

def verification(ctx: Ctx) -> dict[str, Any]:
    """Cross-check documentation claims, ADR references and agent findings against extracted facts."""
    s = ctx.store
    checks = []
    for e in s.edges(where="evidence=? AND json_extract(attrs, '$.claim') IS NOT NULL", params=(ev.DOCUMENTATION,)):
        corroborating = _corroborate(ctx, e["src"], e["dst"], e["type"])
        status = "corroborated" if corroborating else "contradicted_or_unverified"
        checks.append({"claim": e["attrs"]["claim"], "source": f"{e['file']}:{e['line']}", "status": status,
                       "evidence": corroborating[:3],
                       "explanation": ("static analysis found a matching path" if corroborating else
                                       "no static path supports this documented relationship; the documentation may "
                                       "be stale or describe runtime behaviour")})
    dangling = []
    for d in s.nodes("decision"):
        if d["evidence"] != ev.DOCUMENTATION:
            continue
        mentioned = re.findall(r"`([A-Za-z_][\w.]+)`", (d["attrs"].get("decision") or "") +
                               (d["attrs"].get("consequences") or ""))
        for name in mentioned:
            if not s.find_nodes_by_name(name) and not s.search(name, limit=1):
                dangling.append(claim(f"ADR '{d['name']}' references `{name}`, which no longer exists",
                                      ev.DOCUMENTATION, [d["file"]], 0.7))
    stale = [{"id": n["id"], "name": n["name"], "status": n["status"], "evidence": n["evidence"]}
             for n in s.nodes(where="status IN ('stale','unsupported')")]
    contradictions = [c for c in checks if c["status"] != "corroborated"]
    return {"title": "Verification", "documentation_claims": checks, "contradictions": contradictions,
            "dangling_references": dangling, "stale_or_unsupported_knowledge": stale[:50],
            "summary": {"claims": len(checks), "corroborated": len(checks) - len(contradictions),
                        "contradicted_or_unverified": len(contradictions), "stale": len(stale)}}


NON_CORROBORATING = (ev.DOCUMENTATION, ev.AGENT_INFERENCE, ev.HANDOFF)


def _corroborate(ctx: Ctx, src: str, dst: str, typ: str) -> list[str]:
    s, g = ctx.store, ctx.graph
    static = [e for e in s.edges(src=src, dst=dst) if ev.classify(e["evidence"]) == "observed"]
    if static:
        return [f"{e['type']} ({e['evidence']}) at {e['file']}:{e['line']}" for e in static]
    try:
        t = g.trace(src, dst, max_depth=5, k=1, exclude_evidence=NON_CORROBORATING)
    except LookupError:
        return []
    if not t["paths"]:
        # Class-level claims: any member reaching the target class or its members.
        members = [e["dst"] for e in s.edges(src=src, types=["contains"])]
        for m in members:
            try:
                t = g.trace(m, dst, max_depth=5, k=1, exclude_evidence=NON_CORROBORATING)
            except LookupError:
                continue
            if t["paths"]:
                break
    out = []
    for p in t["paths"]:
        if typ in ("writes",) and not any(st["type"] == "writes" for st in p["steps"]):
            continue
        out.append(f"path ({p['basis']}, trust {p['trust']}): " + " -> ".join(n.split(":")[-1] for n in p["nodes"]))
    return out


# ===================================================================================== roadmap

def roadmap(ctx: Ctx, goal: str | None = None, budget: int = 3000) -> dict[str, Any]:
    """Dependency-aware implementation plan with prerequisites, tests, risks and validation."""
    s, g = ctx.store, ctx.graph
    if not goal:
        return _improvement_roadmap(ctx)
    from promapkin.retrieval import Retriever

    ret = Retriever(ctx.ws, ctx.rt)
    hits = ret.search(goal, limit=20)
    seeds = [h for h in hits if h["kind"] in ("class", "function", "method", "endpoint")][:5] or \
        [h for h in hits if h["kind"] == "module"][:3]
    if not seeds:
        return {"title": "Implementation roadmap", "goal": goal, "steps": [],
                "warning": "No relevant code found for this goal; refine the goal or scan the repository."}
    affected_mods: dict[str, float] = {}
    tests: dict[str, dict] = {}
    endpoints: set[str] = set()
    docs: set[str] = set()
    impact_nodes: dict[str, float] = {}
    for sd in seeds:
        imp = g.impact(sd["id"], depth=3)
        mod = module_of(s, sd["id"])
        if mod:
            affected_mods[mod] = max(affected_mods.get(mod, 0), sd["score"])
        for group, items in imp["affected"].items():
            for it in items:
                impact_nodes[it["id"]] = max(impact_nodes.get(it["id"], 0), it["impact_score"])
                if group == "endpoints":
                    endpoints.add(it["id"])
                elif group == "documentation":
                    docs.add(it["id"])
                m = module_of(s, it["id"])
                if m and group in ("symbols", "modules") and it["impact_score"] >= 0.3:
                    affected_mods.setdefault(m, it["impact_score"] * 0.5)
        for t in imp["tests_to_run"]:
            tests[t["id"]] = t
    deps = g.module_dependencies()
    order = _topo([m for m in affected_mods if not (s.node(m) or {}).get("attrs", {}).get("test")], deps)
    constraints = _constraints_for(ctx, [sd["id"] for sd in seeds] + list(affected_mods))
    steps: list[dict[str, Any]] = []

    def add(title: str, detail: str, kind: str, components: list[str], prereq: list[int], validation: list[str],
            risk: str = "low") -> int:
        steps.append({"step": len(steps) + 1, "title": title, "detail": detail, "kind": kind,
                      "components": components, "prerequisites": prereq, "validation": validation, "risk": risk})
        return len(steps)

    first = add("Confirm constraints and decisions",
                "Read the decisions and constraints below; they bound the change.", "prepare",
                [c["source"] for c in constraints][:6], [], ["constraints acknowledged in the handoff"])
    baseline = add("Establish a baseline",
                   "Run the existing tests that cover the affected code before changing it.", "test",
                   [t["id"] for t in tests.values()][:10], [first],
                   ["all listed tests pass on the current revision"] if tests else
                   ["no existing tests reach this code — add characterization tests first"],
                   "medium" if not tests else "low")
    prev = [baseline]
    risk_items = {r["symbol"]: r for r in risks(ctx)["symbols"]}
    for m in order:
        node = s.node(m)
        if not node:
            continue
        members = [sd for sd in seeds if module_of(s, sd["id"]) == m]
        dependents = [x for x in affected_mods if m in deps.get(x, set())]
        lay = layer_of(node)[0]
        touches_db = any(e["dst"] == "resource:database" for sym in s.edges(src=m, types=["defines"])
                         for e in s.edges(src=sym["dst"], types=["writes"]))
        risky = [r for q, r in risk_items.items() if q.startswith(node["name"] + ".")]
        level = "high" if touches_db or len(dependents) >= 3 else ("medium" if risky or dependents else "low")
        what = ", ".join(x["name"] for x in members) or "adapt to upstream changes"
        prereq = [st["step"] for st in steps if any(d in st["components"] for d in deps.get(m, set()))] or prev
        sn = add(f"Modify {node['name']} ({lay})", f"Change: {what}. Location: {loc(node)}.", "implement",
                 [m] + [x["id"] for x in members], sorted(set(prereq)),
                 [f"unit tests for {node['name']}"] + (["migration/backfill reviewed for persisted data"]
                                                       if touches_db else []), level)
        if touches_db:
            add(f"Plan data migration for {node['name']}",
                "This module writes to a database. Version the schema change, make it backward compatible, and "
                "provide a rollback.", "migration", [m, "resource:database"], [sn],
                ["migration applies and rolls back on a copy of production-like data"], "high")
    impl_steps = [st["step"] for st in steps if st["kind"] in ("implement", "migration")]
    if endpoints:
        add("Verify API contracts", "Endpoints reached by the change: " + ", ".join(
            e.removeprefix("api:") for e in sorted(endpoints)), "verify", sorted(endpoints), impl_steps,
            ["request/response schemas unchanged unless intended", "auth boundaries preserved"], "medium")
    add("Update and add tests", "Extend tests for changed behaviour; cover untested affected symbols.", "test",
        [t for t in tests][:10], impl_steps,
        ["new tests fail before and pass after the change", "full affected test set passes"])
    if docs:
        add("Update documentation", "Docs describing affected code: " + ", ".join(sorted(docs)[:6]), "docs",
            sorted(docs)[:10], impl_steps, ["docs match new behaviour; ADR added if the design changed"])
    add("Final validation", "Re-run promapkin sync + verify, then the affected tests.", "validate",
        [], [st["step"] for st in steps], ["promapkin verify shows no new contradictions",
                                           "impact analysis of the diff matches the planned components"])
    _persist_roadmap(ctx, goal, steps)
    return {"title": "Implementation roadmap", "goal": goal,
            "relevant_code": [{"id": sd["id"], "score": sd["score"], "location": sd.get("location")} for sd in seeds],
            "affected_modules": [m.removeprefix("module:") for m in order],
            "constraints": constraints, "steps": steps,
            "tests": list(tests.values())[:20],
            "risk_assessment": {"overall": max((st["risk"] for st in steps), key=["low", "medium", "high"].index),
                                "endpoints_affected": len(endpoints), "tests_available": len(tests),
                                "untested_seed_symbols": [sd["id"] for sd in seeds if sd["id"] not in
                                                          {e["dst"] for e in s.edges(types=["verifies"])}]}}


def _topo(nodes: list[str], deps: dict[str, set[str]]) -> list[str]:
    """Dependencies first: a module is scheduled after the modules it imports (within the set)."""
    nodes_set = set(nodes)
    out, seen = [], set()

    def visit(n: str, stack: set[str]) -> None:
        if n in seen or n in stack:
            return
        stack.add(n)
        for d in sorted(deps.get(n, set()) & nodes_set):
            visit(d, stack)
        stack.discard(n)
        seen.add(n)
        out.append(n)

    for n in sorted(nodes):
        visit(n, set())
    return out


def _constraints_for(ctx: Ctx, node_ids: list[str]) -> list[dict[str, Any]]:
    s = ctx.store
    out = []
    ids = set(node_ids)
    for d in s.nodes("decision"):
        refs = {e["dst"] for e in s.edges(src=d["id"])}
        if refs & ids or any(module_of(s, r) in ids for r in refs):
            out.append({"constraint": d["name"], "detail": (d["attrs"].get("decision") or d["text"] or "")[:300],
                        "source": d["file"] or d["id"], "evidence": d["evidence"], "critical": True})
    for n in s.nodes(where="kind IN ('constraint','requirement') AND status != 'superseded'"):
        refs = {e["dst"] for e in s.edges(src=n["id"])}
        if refs & ids or not refs:
            out.append({"constraint": n["name"], "detail": (n["text"] or "")[:300], "source": n["id"],
                        "evidence": n["evidence"], "critical": True})
    for e in s.edges(types=["constrains"]):
        if e["dst"] in ids:
            src = s.node(e["src"])
            if src:
                out.append({"constraint": src["name"], "detail": (src["text"] or "")[:300], "source": src["id"],
                            "evidence": e["evidence"], "critical": True})
    uniq = {c["source"] + c["constraint"]: c for c in out}
    return list(uniq.values())


def _persist_roadmap(ctx: Ctx, goal: str, steps: list[dict[str, Any]]) -> None:
    from promapkin.knowledge import slugify

    s = ctx.store
    base = f"roadmap:{slugify(goal, 40)}"
    s.delete_nodes("kind='roadmap_step' AND id LIKE ?", (base + "#%",))
    for st in steps:
        sid = f"{base}#{st['step']}"
        s.upsert_node(sid, "roadmap_step", st["title"], evidence=ev.HEURISTIC, confidence=0.6, revision=ctx.revision,
                      text=st["detail"], attrs={"goal": goal, **st})
        for p in st["prerequisites"]:
            s.add_edge(sid, f"{base}#{p}", "depends_on", evidence=ev.HEURISTIC, revision=ctx.revision)
        for c in st["components"]:
            if s.node(c):
                s.add_edge(sid, c, "affects", evidence=ev.HEURISTIC, revision=ctx.revision)


def _improvement_roadmap(ctx: Ctx) -> dict[str, Any]:
    r = risks(ctx)
    steps = []
    for item in r["structural"][:8]:
        steps.append({"step": len(steps) + 1, "title": "Resolve: " + item["statement"], "kind": "refactor",
                      "prerequisites": [], "validation": ["promapkin verify", "affected tests pass"],
                      "risk": "medium", "sources": item["sources"]})
    for item in r["symbols"][:8]:
        if "no test reaches it" in item["reasons"]:
            steps.append({"step": len(steps) + 1, "title": f"Add tests for {item['symbol']}", "kind": "test",
                          "prerequisites": [], "validation": ["new tests pass", "coverage of the symbol"],
                          "risk": "low", "sources": [item["location"]]})
    return {"title": "Improvement roadmap", "goal": None, "steps": steps,
            "note": "No goal given: steps are derived from the risk register. Pass a goal for a task plan."}
