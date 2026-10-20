"""JavaScript/TypeScript extraction through the TypeScript compiler (parser + type checker).

Files are processed in one batch so the type checker can resolve calls across files
(`this.repo.findAll()` resolves through the declared type of `repo`). Resolved targets are
emitted as `ID:` references so incremental re-resolution keeps them consistent.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from promapkin import evidence as ev
from promapkin import toolchain
from promapkin.extract import Facts, is_test_path
from promapkin.extract.resources import AUTH_HINT, classify_call

BRIDGE = Path(__file__).with_name("ts_bridge.js")


class BridgeError(RuntimeError):
    pass


def _sym(module: str, qual: str) -> str:
    return f"symbol:{module}:{qual}"


def _target_ref(t: dict | None) -> str | None:
    if not t:
        return None
    if t.get("module") and t.get("qual"):
        return "ID:" + _sym(t["module"], t["qual"])
    return None


def extract_batch(root: Path, paths: list[str], timeout: int = 600) -> dict[str, Facts]:
    node = toolchain.node_binary()
    ts_mod = toolchain.typescript_module(root)
    if not node or not ts_mod:
        raise BridgeError("TypeScript toolchain not available")
    if not paths:
        return {}
    payload = json.dumps({"root": str(root), "files": paths, "typescript": ts_mod})
    proc = subprocess.run([node, "--max-old-space-size=4096", str(BRIDGE)], input=payload, capture_output=True,
                          text=True, timeout=timeout, check=False)
    if proc.returncode != 0:
        raise BridgeError(proc.stderr.strip()[-800:] or f"bridge exited {proc.returncode}")
    data = json.loads(proc.stdout)
    return {path: _to_facts(path, rec) for path, rec in data["files"].items()}


def _to_facts(path: str, rec: dict) -> Facts:
    language = "typescript" if path.endswith((".ts", ".tsx", ".mts", ".cts")) else "javascript"
    f = Facts(path, language)
    mod = rec["module"]
    mod_id = f"module:{mod}"
    is_test = is_test_path(path)
    stem = Path(path).stem
    f.node(mod_id, "module", mod, ev.AST, qualname=mod, file=path, line_start=1, line_end=rec["lines"],
           text=rec.get("doc"), attrs={"test": is_test, "loc": rec["lines"], "parser": "typescript",
                                       "entrypoint": stem in ("index", "main", "server", "app", "cli"),
                                       "imports": {}})
    f.edge(f"file:{path}", mod_id, "defines", ev.AST)

    for d in rec["decls"]:
        sid = _sym(mod, d["qual"])
        kind = d["kind"]
        if is_test and kind in ("function", "method"):
            kind = "test"
        attrs = {"symbol_type": d["kind"], "public": d.get("public", True) and not d["name"].startswith("_"),
                 "exported": d.get("exported", False)}
        for k in ("params", "returns", "methods", "abstract", "interface", "async"):
            if d.get(k) is not None:
                attrs[k] = d[k]
        if "." in d["qual"]:
            attrs["class"] = d["qual"].rsplit(".", 1)[0]
        f.node(sid, kind, d["name"], ev.AST, qualname=f"{mod}.{d['qual']}", file=path, line_start=d["line_start"],
               line_end=d["line_end"], text=d.get("doc"), attrs=attrs)
        parent = _sym(mod, d["qual"].rsplit(".", 1)[0]) if "." in d["qual"] else mod_id
        f.edge(parent, sid, "contains" if "." in d["qual"] else "defines", ev.AST, line=d["line_start"])
        for inj in d.get("injected") or []:
            ref = _target_ref(inj.get("target"))
            if ref:
                f.ref(sid, ref, "references", d["line_start"], module=mod, injected=True)

    for imp in rec["imports"]:
        target = ("./" + imp["module"]) if imp.get("module") else imp["package"]
        f.ref(mod_id, target, "imports", imp["line"], module=mod)

    for c in rec["calls"]:
        owner = _sym(mod, c["from"]) if c.get("from") else mod_id
        t = c.get("target") or {}
        ref = _target_ref(t)
        if ref:
            f.ref(owner, ref, "calls", c["line"], module=mod, typed=t.get("via"))
            continue
        text = re.sub(r"\s+", "", c["text"])
        if t.get("package"):
            call = f"{t['package']}.{t.get('name') or text.rsplit('.', 1)[-1]}"
            res = classify_call(text, t["package"])
            if res:
                resource, verb = res
                f.node(f"resource:{resource}", "resource", resource, ev.HEURISTIC, attrs={"category": resource})
                f.edge(owner, f"resource:{resource}", verb, ev.HEURISTIC, line=c["line"], confidence=0.75,
                       attrs={"call": text, "classifier": "heuristic", "package": t["package"]})
            else:
                f.ref(owner, call, "external_call", c["line"], module=mod, library=t["package"])
        elif text in ("fetch", "globalThis.fetch", "window.fetch"):
            f.node("resource:http", "resource", "http", ev.HEURISTIC, attrs={"category": "http"})
            f.edge(owner, "resource:http", "reads", ev.HEURISTIC, line=c["line"], confidence=0.75,
                   attrs={"call": text, "classifier": "heuristic"})

    for h in rec["inherits"]:
        ref = _target_ref(h.get("target"))
        if ref:
            f.ref(_sym(mod, h["from"]), ref, h["type"], h["line"], module=mod)

    for r in rec["routes"]:
        api_id = f"api:{r['method']} {r['route']}"
        auth = [h["text"] for h in r["handlers"] if AUTH_HINT.search(h["text"])]
        f.node(api_id, "endpoint", f"{r['method']} {r['route']}", ev.AST, file=path, line_start=r["line"],
               attrs={"method": r["method"], "route": r["route"], "router": r["router"], "auth": auth})
        f.edge(mod_id, api_id, "exposes", ev.AST, line=r["line"])
        for h in r["handlers"]:
            if h["text"] in auth:
                continue
            ref = _target_ref(h.get("target"))
            if ref:
                f.ref(api_id, ref, "routes_to", r["line"], module=mod)
            elif h.get("inline") and r.get("owner"):
                f.ref(api_id, "ID:" + _sym(mod, r["owner"]), "routes_to", r["line"], module=mod)

    for e in rec["env"]:
        owner = _sym(mod, e["owner"]) if e.get("owner") else mod_id
        f.node(f"config:{e['key']}", "config", e["key"], ev.AST, attrs={"source": "environment"})
        f.edge(owner, f"config:{e['key']}", "reads", ev.AST, line=e["line"])

    for e in rec["events"]:
        eid = f"event:{e['event']}"
        f.node(eid, "event", e["event"], ev.AST, attrs={"source": "event wiring"})
        owner = _sym(mod, e["owner"]) if e.get("owner") else mod_id
        if e["role"] == "emit":
            f.edge(owner, eid, "produces", ev.AST, line=e["line"])
        else:
            f.edge(owner, eid, "consumes", ev.AST, line=e["line"])
            ref = _target_ref(e.get("handler"))
            if ref:
                f.ref(eid, ref, "invokes", e["line"], module=mod, wiring="event handler")
    return f
