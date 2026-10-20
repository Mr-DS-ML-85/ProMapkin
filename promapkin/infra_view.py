"""Read models over infrastructure and data nodes: what is deployed and how it reaches code, which cloud
resources exist and depend on each other, which configuration is declared versus read, and the data model
(tables, keys, ORM mappings, readers and writers, API schemas and spec coverage)."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from promapkin import evidence as ev
from promapkin.store import Store

DEPLOYABLE_KINDS = ("container_image", "service", "deployment")
CODE_KINDS = ("module", "class", "function", "method", "file", "endpoint")


def _loc(n: dict[str, Any] | None) -> str:
    if not n:
        return ""
    return f"{n['file']}:{n['line_start']}" if n.get("file") and n.get("line_start") else (n.get("file") or "")


def _short(nid: str) -> str:
    return nid.split(":", 1)[-1] if not nid.startswith("symbol:") else nid.rsplit(":", 1)[-1]


def infrastructure(store: Store) -> dict[str, Any]:
    out_edges: dict[str, list[dict[str, Any]]] = defaultdict(list)
    in_edges: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for e in store.edges(types=["runs", "builds", "routes_to", "depends_on", "configures", "uses", "scales",
                                "reads"]):
        out_edges[e["src"]].append(e)
        in_edges[e["dst"]].append(e)
    nodes = {n["id"]: n for k in DEPLOYABLE_KINDS + ("ingress", "cloud_resource", "iac_module", "pipeline",
                                                     "config_source") for n in store.nodes(k)}

    def targets(nid: str, etype: str) -> list[str]:
        return sorted({e["dst"] for e in out_edges.get(nid, []) if e["type"] == etype})

    deployables = []
    for nid, n in sorted(nodes.items()):
        if n["kind"] not in DEPLOYABLE_KINDS:
            continue
        a = n["attrs"]
        deployables.append({
            "id": nid, "kind": n["kind"], "name": n["name"], "location": _loc(n),
            "source": a.get("source") or a.get("kind") or ("Dockerfile" if n["kind"] == "container_image" else None),
            "image": a.get("image") or a.get("images") or a.get("base_images"),
            "ports": a.get("ports") or [], "replicas": a.get("replicas"),
            "runs_code": targets(nid, "runs"), "builds": targets(nid, "builds"),
            "routes_to": targets(nid, "routes_to"),
            "depends_on": [t for t in targets(nid, "depends_on")],
            "config": [_short(t) for t in targets(nid, "configures")],
            "reached_from": sorted({e["src"] for e in in_edges.get(nid, []) if e["type"] == "routes_to"}),
        })
    resources = [n for n in nodes.values() if n["kind"] == "cloud_resource"]
    by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for n in sorted(resources, key=lambda n: n["id"]):
        by_cat[n["attrs"].get("category", "other")].append({
            "id": n["id"], "type": n["attrs"].get("type"), "provider": n["attrs"].get("provider"),
            "location": _loc(n), "depends_on": targets(n["id"], "depends_on"),
            "config": [_short(t) for t in targets(n["id"], "configures")]})
    ingress = [{"id": n["id"], "hosts": n["attrs"].get("hosts", []), "routes_to": targets(n["id"], "routes_to"),
                "location": _loc(n)} for n in nodes.values() if n["kind"] == "ingress"]
    pipelines = [{"id": n["id"], "name": n["name"], "triggers": n["attrs"].get("triggers", []),
                  "jobs": n["attrs"].get("jobs", []), "runs_code": targets(n["id"], "runs"), "location": _loc(n)}
                 for n in nodes.values() if n["kind"] == "pipeline"]
    return {"deployables": deployables, "ingress": ingress,
            "cloud_resources": {k: v for k, v in sorted(by_cat.items())},
            "iac_modules": [{"id": n["id"], "source": n["attrs"].get("source"), "location": _loc(n)}
                            for n in nodes.values() if n["kind"] == "iac_module"],
            "pipelines": pipelines, "configuration": config_surface(store),
            "counts": dict(Counter(n["kind"] for n in nodes.values()))}


def config_surface(store: Store) -> dict[str, Any]:
    """Configuration keys declared by infrastructure (or .env.example) versus read by code."""
    declared: dict[str, list[str]] = defaultdict(list)
    read: dict[str, list[str]] = defaultdict(list)
    for e in store.edges(types=["configures", "reads"]):
        if not e["dst"].startswith("config:"):
            continue
        key = e["dst"].split(":", 1)[1]
        if e["type"] == "configures":
            declared[key].append(e["src"])
        elif e["src"].startswith(("symbol:", "module:")):
            read[key].append(f"{e['file']}:{e['line']}" if e.get("file") else e["src"])
    has_infra = bool(declared)
    return {
        "keys": [{"key": k, "declared_by": sorted(set(declared.get(k, [])))[:8], "read_at": read.get(k, [])[:5]}
                 for k in sorted(set(declared) | set(read))],
        "read_but_never_declared": sorted(set(read) - set(declared)) if has_infra else [],
        "declared_but_never_read": sorted(k for k in set(declared) - set(read) if k.isupper()),
        "note": None if has_infra else "no infrastructure or .env.example declares configuration in this repository",
    }


def data_model(store: Store) -> dict[str, Any]:
    tables = store.nodes("table")
    access: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for e in store.edges(types=["reads", "writes", "maps_to", "migrates", "alters", "depends_on", "triggers_on",
                                "indexes"]):
        if e["dst"].startswith("table:"):
            access[e["dst"]][e["type"]].add(e["src"])
    out_tables = []
    for t in sorted(tables, key=lambda n: n["id"]):
        a = t["attrs"]
        acc = access.get(t["id"], {})
        fks = [e for e in store.edges(src=t["id"], types=["depends_on"]) if e["dst"].startswith("table:")]
        out_tables.append({
            "id": t["id"], "name": t["name"], "view": bool(a.get("view")), "location": _loc(t),
            "defined_by": "code (no DDL found)" if a.get("inferred_from") == "code" else a.get("format", "ddl"),
            "columns": a.get("columns", []), "primary_key": a.get("primary_key", []),
            "references": [{"table": e["dst"].split(":", 1)[1], **{k: v for k, v in e["attrs"].items()
                                                                   if k in ("columns", "references")}} for e in fks],
            "referenced_by": sorted(s.split(":", 1)[1] for s in acc.get("depends_on", set())
                                    if s.startswith("table:")),
            "orm_classes": sorted(acc.get("maps_to", set())),
            "read_by": sorted(acc.get("reads", set()))[:20], "written_by": sorted(acc.get("writes", set()))[:20],
            "migrations": sorted(acc.get("migrates", set()) | acc.get("alters", set()))[:20],
        })
    schema_types = [{"id": n["id"], "name": n["name"], "format": n["attrs"].get("format"),
                     "fields": n["attrs"].get("fields") or n["attrs"].get("enum") or [], "location": _loc(n)}
                    for n in sorted(store.nodes("schema_type"), key=lambda n: n["id"])]
    endpoints = store.nodes("endpoint")
    by_protocol = Counter(n["attrs"].get("protocol", "http") for n in endpoints)
    ops = store.nodes("api_operation")
    specified = {e["src"]: e["dst"] for e in store.edges(types=["specifies"]) if e["dst"].startswith("api:")}
    http_code = {n["id"] for n in endpoints if n["id"].startswith("api:")}
    spec = {
        "operations": len(ops),
        "unimplemented": sorted(n["name"] for n in ops if n["id"] not in specified),
        "undocumented": sorted(e.split(":", 1)[1] for e in http_code - set(specified.values())) if ops else [],
    }
    return {"tables": out_tables, "schema_types": schema_types,
            "endpoints_by_protocol": dict(by_protocol), "api_spec": spec,
            "counts": {"tables": len(tables), "views": sum(1 for t in tables if t["attrs"].get("view")),
                       "inferred_from_code": sum(1 for t in tables if t["attrs"].get("inferred_from") == "code"),
                       "schema_types": len(schema_types)}}


def claims(store: Store) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(infrastructure claims, data-model claims) for the architecture blueprint."""
    from promapkin.agents.specialists import claim
    infra = infrastructure(store)
    dm = data_model(store)
    ic: list[dict[str, Any]] = []
    for d in infra["deployables"]:
        bits = []
        if d["runs_code"]:
            bits.append("runs " + ", ".join(f"`{_short(t)}`" for t in d["runs_code"]))
        if d["routes_to"]:
            bits.append("routes to " + ", ".join(f"`{_short(t)}`" for t in d["routes_to"]))
        if d["builds"]:
            bits.append("built from " + ", ".join(f"`{_short(t)}`" for t in d["builds"]))
        if d["depends_on"]:
            bits.append("depends on " + ", ".join(f"`{_short(t)}`" for t in d["depends_on"]))
        if d["config"]:
            bits.append(f"sets {len(d['config'])} config key(s)")
        ic.append(claim(f"{d['kind'].replace('_', ' ').capitalize()} `{d['name']}`"
                        + (" " + "; ".join(bits) if bits else ""), ev.MANIFEST, [d["location"]]))
    for cat, items in infra["cloud_resources"].items():
        ic.append(claim(f"{len(items)} {cat} resource(s) in Terraform: " + ", ".join(
            f"`{i['type']}`" for i in items[:6]), ev.MANIFEST, [i["location"] for i in items[:4]]))
    for p in infra["pipelines"]:
        ic.append(claim(f"CI pipeline `{p['name']}` ({', '.join(p['triggers']) or 'manual'}) runs jobs "
                        f"{', '.join(p['jobs'][:6])}", ev.MANIFEST, [p["location"]]))
    cs = infra["configuration"]
    if cs["read_but_never_declared"]:
        ic.append(claim("Config read by code but declared by no infrastructure file: "
                        + ", ".join(f"`{k}`" for k in cs["read_but_never_declared"][:10]), ev.MANIFEST, [],
                        method="config surface: code reads vs infrastructure declarations"))
    dc: list[dict[str, Any]] = []
    for t in dm["tables"]:
        bits = [f"{len(t['columns'])} column(s)"] if t["columns"] else []
        if t["references"]:
            bits.append("references " + ", ".join(f"`{r['table']}`" for r in t["references"]))
        if t["orm_classes"]:
            bits.append("mapped by " + ", ".join(f"`{_short(c)}`" for c in t["orm_classes"][:3]))
        if t["read_by"] or t["written_by"]:
            bits.append(f"read by {len(t['read_by'])}, written by {len(t['written_by'])} symbol(s)")
        evidence = ev.LEXICAL if t["defined_by"].startswith("code") else ev.MANIFEST
        dc.append(claim(f"{'View' if t['view'] else 'Table'} `{t['name']}`" + (" — " + "; ".join(bits) if bits else "")
                        + (" (no DDL in repository)" if evidence == ev.LEXICAL else ""), evidence, [t["location"]]))
    if dm["api_spec"]["operations"]:
        s = dm["api_spec"]
        dc.append(claim(f"API spec: {s['operations']} operation(s); {len(s['unimplemented'])} without a matching "
                        f"code endpoint, {len(s['undocumented'])} code endpoint(s) missing from the spec",
                        ev.MANIFEST, [], unimplemented=s["unimplemented"][:10], undocumented=s["undocumented"][:10]))
    return ic, dc


def mermaid_infrastructure(store: Store) -> str:
    infra = infrastructure(store)
    lines = ["flowchart LR"]

    def nid(x: str) -> str:
        import re
        return re.sub(r"[^A-Za-z0-9_]", "_", x)[:60]

    seen: set[str] = set()

    def node(x: str, label: str, shape: str = "[]") -> None:
        if x not in seen:
            seen.add(x)
            lines.append(f'  {nid(x)}{shape[0]}"{label.replace(chr(34), "")}"{shape[1]}')
    for i in infra["ingress"]:
        node(i["id"], f"ingress {', '.join(h for h in i['hosts'] if h) or i['id']}", "()")
        for t in i["routes_to"]:
            node(t, _short(t))
            lines.append(f"  {nid(i['id'])} --> {nid(t)}")
    for d in infra["deployables"]:
        node(d["id"], f"{d['kind'].replace('_', ' ')}: {d['name']}")
        for etype, label in (("routes_to", "routes"), ("runs_code", "runs"), ("builds", "builds"),
                             ("depends_on", "depends")):
            for t in d[etype]:
                node(t, _short(t), "[]" if not t.startswith(("module:", "file:")) else "[[]]")
                lines.append(f"  {nid(d['id'])} -->|{label}| {nid(t)}")
    for cat, items in infra["cloud_resources"].items():
        for r in items:
            node(r["id"], f"{cat}: {r['type']}", "[()]" if cat in ("database", "storage", "cache") else "[]")
            for t in r["depends_on"]:
                if t.startswith("tf:"):
                    node(t, _short(t).split(":", 1)[-1])
                    lines.append(f"  {nid(r['id'])} --> {nid(t)}")
    return "\n".join(lines) if len(lines) > 1 else "flowchart LR\n  empty[\"no infrastructure files indexed\"]"


def mermaid_er(store: Store) -> str:
    import re
    dm = data_model(store)
    if not dm["tables"]:
        return "erDiagram\n"
    lines = ["erDiagram"]

    def ident(x: str) -> str:
        return re.sub(r"[^A-Za-z0-9_]", "_", x)
    for t in dm["tables"][:60]:
        cols = t["columns"][:12]
        if cols:
            lines.append(f"  {ident(t['name'])} {{")
            for c in cols:
                key = " PK" if c["name"] in t["primary_key"] else ""
                lines.append(f"    {ident(c.get('type', 'any').split('(')[0].replace(' ', '_')) or 'any'} "
                             f"{ident(c['name'])}{key}")
            lines.append("  }")
        for r in t["references"]:
            lines.append(f"  {ident(r['table'])} ||--o{{ {ident(t['name'])} : \"{r.get('columns') or 'fk'}\"")
    return "\n".join(lines)
