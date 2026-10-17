"""Repository reconnaissance, incremental extraction and cross-file resolution.

A scan:
1. lists repository files (respecting .gitignore) and hashes them;
2. re-extracts only files whose content changed and forgets deleted files;
3. re-resolves references into typed edges with evidence classes;
4. runs the two-tier classification pass (layers, resource calls, doc and note intent);
5. refreshes embeddings for changed nodes and marks dependent inferences stale.

Everything happens in one transaction: a failure leaves the previous index intact.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import re
import time
from pathlib import PurePosixPath
from typing import Any, Callable

from promapkin import evidence as ev
from promapkin import gitutil
from promapkin.config import Workspace
from promapkin.extract import Facts, extractor_for, is_test_path, language_of
from promapkin.security import MAX_FILE_BYTES, is_probably_binary
from promapkin.store import Store

SHARED_KINDS = {"package", "resource", "config", "service"}
CALL_STOPLIST = {
    "get", "set", "add", "append", "extend", "update", "pop", "items", "keys", "values", "join", "split", "strip",
    "format", "replace", "lower", "upper", "startswith", "endswith", "len", "str", "int", "float", "bool", "list",
    "dict", "tuple", "print", "isinstance", "super", "range", "enumerate", "zip", "map", "filter", "sorted", "open",
    "copy", "close", "read", "write", "run", "main", "__init__", "log", "info", "debug", "error", "warning", "push",
    "then", "catch", "forEach", "toString", "length", "slice", "includes", "indexOf", "find", "remove", "clear",
    "sum", "min", "max", "abs", "any", "all", "type", "id", "hash", "next", "iter", "setdefault", "encode", "decode",
}


class ScanResult(dict):
    pass


def scan(ws: Workspace, *, full: bool = False, progress: Callable[[str], None] | None = None,
         interactive: bool = False, ask: Callable[[str], bool] | None = None) -> ScanResult:
    from promapkin import ml

    store = ws.store
    revision = ws.revision()
    started = time.time()
    job = store.start_job("scan", revision)
    say = progress or (lambda _m: None)
    try:
        rt = ml.runtime(ws, interactive=interactive, ask=ask)
        files = _candidate_files(ws)
        old = store.file_index()
        contents: dict[str, tuple[str, str, int]] = {}
        for rel in files:
            lang = language_of(rel)
            if not lang:
                continue
            path = ws.root / rel
            try:
                if path.stat().st_size > MAX_FILE_BYTES:
                    continue
                data = path.read_bytes()
            except OSError:
                continue
            if is_probably_binary(data):
                continue
            contents[rel] = (gitutil.sha_of(data), lang, len(data))
        changed = [p for p, (sha, _, _) in contents.items() if full or old.get(p) != sha]
        deleted = [p for p in old if p not in contents]
        say(f"{len(contents)} files indexed, {len(changed)} changed, {len(deleted)} deleted")

        errors: list[str] = []
        with store.transaction():
            for p in deleted:
                store.forget_file(p)
            for p in changed:
                store.forget_file(p)
                sha, lang, size = contents[p]
                text = (ws.root / p).read_text(encoding="utf-8", errors="replace")
                extractor = extractor_for(lang)
                store.upsert_file(p, sha, lang, size, revision)
                store.upsert_node(f"file:{p}", "file", PurePosixPath(p).name, evidence=ev.AST, file=p,
                                  owner_file=p, revision=revision,
                                  attrs={"language": lang, "size": size, "sha": sha, "test": is_test_path(p)})
                if extractor is None:
                    continue
                try:
                    facts = extractor(p, text)
                except Exception as exc:  # noqa: BLE001 - one bad file never aborts a scan
                    errors.append(f"{p}: {exc}")
                    continue
                errors += [f"{p}: {e}" for e in facts.errors]
                _persist(store, facts, revision)
            _structure(store, ws, list(contents), revision)
            ambiguous: list[dict[str, Any]] = []
            stats = resolve(store, revision, ambiguous)
            if changed or deleted or full:
                stats["classified"] = classify_pass(store, rt, revision)
            stats["stale_inferences"] = _mark_stale(store, set(changed) | set(deleted)) if not full else 0
            _gc_shared(store)
            stats.update(graph_ml_pass(store, rt, ambiguous, revision))
            store.conn.execute("DELETE FROM meta WHERE key LIKE 'ctx:%'")   # context packages are revision-bound
            store.set_meta("indexed_revision", revision)
            store.set_meta("indexed_at", str(time.time()))
            store.set_meta("ml", json.dumps(rt.describe()))
        result = ScanResult(
            revision=revision, files=len(contents), changed=len(changed), deleted=len(deleted),
            changed_files=sorted(changed)[:500], deleted_files=sorted(deleted)[:500],
            errors=errors[:50], seconds=round(time.time() - started, 3), ml=rt.describe(), **stats,
        )
        store.finish_job(job, "succeeded", {k: v for k, v in result.items()
                                            if k not in ("errors", "changed_files", "deleted_files")})
        return result
    except BaseException as exc:
        store.finish_job(job, "failed", {"error": repr(exc)})
        raise


def _candidate_files(ws: Workspace) -> list[str]:
    files = gitutil.list_files(ws.root)
    inc, exc = ws.config.include, ws.config.exclude
    if inc:
        files = [f for f in files if any(fnmatch.fnmatch(f, pat) for pat in inc)]
    if exc:
        files = [f for f in files if not any(fnmatch.fnmatch(f, pat) for pat in exc)]
    return files


def _persist(store: Store, facts: Facts, revision: str) -> None:
    for n in facts.nodes:
        n = dict(n)
        owner = None if n["kind"] in SHARED_KINDS else n.pop("owner_file", facts.path) or facts.path
        n.pop("owner_file", None)
        nid, kind, name, evidence = n.pop("id"), n.pop("kind"), n.pop("name"), n.pop("evidence")
        store.upsert_node(nid, kind, name, evidence=evidence, owner_file=owner, revision=revision, **n)
    for e in facts.edges:
        e = dict(e)
        src, dst, typ, evidence = e.pop("src"), e.pop("dst"), e.pop("type"), e.pop("evidence")
        e.setdefault("file", facts.path)
        store.add_edge(src, dst, typ, evidence=evidence, owner_file=facts.path, revision=revision, **e)
    for r in facts.refs:
        store.add_ref(facts.path, r["src"], r["target"], r["type"], r["line"], r["context"])


def _structure(store: Store, ws: Workspace, files: list[str], revision: str) -> None:
    """Repository and directory nodes (rebuilt every scan; cheap)."""
    store.delete_nodes("kind IN ('directory', 'repository')")
    repo_id = "repo:root"
    store.upsert_node(repo_id, "repository", ws.root.name, evidence=ev.AST, revision=revision,
                      attrs={"revision": revision, "root": str(ws.root)})
    dirs: set[str] = set()
    for f in files:
        parts = PurePosixPath(f).parts[:-1]
        for i in range(1, len(parts) + 1):
            dirs.add("/".join(parts[:i]))
    for d in sorted(dirs):
        parent = str(PurePosixPath(d).parent)
        store.upsert_node(f"dir:{d}", "directory", PurePosixPath(d).name, evidence=ev.AST, file=d, revision=revision)
        store.add_edge(repo_id if parent == "." else f"dir:{parent}", f"dir:{d}", "contains", evidence=ev.AST,
                       revision=revision)
    for f in files:
        if store.node(f"file:{f}"):
            parent = str(PurePosixPath(f).parent)
            store.add_edge(repo_id if parent == "." else f"dir:{parent}", f"file:{f}", "contains",
                           evidence=ev.AST, revision=revision)
    for m in store.nodes("manifest"):
        store.add_edge(repo_id, m["id"], "configures", evidence=ev.MANIFEST, revision=revision)


# ------------------------------------------------------------------------------------- resolution

class Resolver:
    def __init__(self, store: Store):
        self.store = store
        self.symbols: dict[str, dict] = {}
        self.by_qual: dict[str, str] = {}
        self.by_name: dict[str, list[str]] = {}
        self.modules: dict[str, dict] = {}
        self.files: set[str] = set()
        self.last_ambiguous: list[str] = []
        for n in store.nodes(where="kind IN ('class','function','method','test','module','endpoint','file',"
                                   "'decision','document','doc_section')"):
            if n["kind"] == "module":
                self.modules[n["name"]] = n
                self.by_qual[n["name"]] = n["id"]
            elif n["kind"] == "file":
                self.files.add(n["file"])
            elif n["kind"] in ("class", "function", "method", "test"):
                self.symbols[n["id"]] = n
                if n["qualname"]:
                    self.by_qual[n["qualname"]] = n["id"]
                self.by_name.setdefault(n["name"], []).append(n["id"])

    def module_for(self, target: str) -> str | None:
        t = target[2:] if target.startswith("./") else target
        for cand in (t, t + "/index", t.removesuffix("/index")):
            if cand in self.modules:
                return self.modules[cand]["id"]
        return None

    def resolve_module_import(self, target: str, name: str | None) -> tuple[str, str]:
        if name and (mid := self.module_for(f"{target}.{name}")):
            return mid, ev.AST
        if mid := self.module_for(target):
            return mid, ev.AST
        parts = target.split(".")
        for i in range(len(parts) - 1, 0, -1):
            if mid := self.module_for(".".join(parts[:i])):
                return mid, ev.AST
        if target.startswith("./"):
            return "", ""
        target = target.removeprefix("node:")
        top = target.split("/")[0] if not target.startswith("@") else "/".join(target.split("/")[:2])
        top = top.split(".")[0]
        return f"package:{re.sub(r'[-_.]+', '-', top).lower()}", ev.AST

    def resolve_symbol(self, expr: str, ctx: dict[str, Any]) -> tuple[str, str, float] | None:
        if expr.startswith(("TYPE:", "RET:")):
            return self._resolve_typed(expr, ctx)
        expr = expr.replace("()", "")
        if not expr:
            return None
        module = ctx.get("module") or ""
        cls = ctx.get("cls")
        mod_node = self.modules.get(module)
        imports: dict[str, str] = (mod_node or {}).get("attrs", {}).get("imports", {}) if mod_node else {}
        parts = expr.split(".")
        js = "/" in module

        if parts[0] in ("self", "cls", "this") and len(parts) >= 2:
            if cls:
                hit = self._method(module, cls, parts[1])
                if hit:
                    return hit, ev.AST, 1.0
            return self._name_match(parts[-1], methods=True)

        head, rest = parts[0], parts[1:]
        if head in imports:
            full = imports[head]
            if js and full.startswith("./"):
                full = full[2:]
            qual = ".".join([full] + rest)
            if qual in self.by_qual:
                return self.by_qual[qual], ev.STATIC_RESOLVED, 0.95
            if js:
                # import X from './mod' -> default export; try mod.<name>
                base = full.split(".")[0] if "/" in full else full
                cand = ".".join([base] + ([rest[-1]] if rest else [head]))
                if cand in self.by_qual:
                    return self.by_qual[cand], ev.STATIC_RESOLVED, 0.85
            return None   # resolved to an external or unknown target; the import edge records the package

        local = f"{module}.{expr}"
        if local in self.by_qual:
            return self.by_qual[local], ev.AST, 1.0
        if cls and f"{module}.{cls}.{expr}" in self.by_qual:
            return self.by_qual[f"{module}.{cls}.{expr}"], ev.AST, 1.0
        if expr in self.by_qual:
            return self.by_qual[expr], ev.STATIC_RESOLVED, 0.9
        if len(parts) >= 2 and f"{module}.{parts[0]}" in self.by_qual:   # LocalClass.method
            hit = self._method(module, parts[0], parts[-1])
            if hit:
                return hit, ev.AST, 1.0
        return self._name_match(parts[-1], methods=len(parts) > 1)

    def _resolve_typed(self, expr: str, ctx: dict[str, Any]) -> tuple[str, str, float] | None:
        """'TYPE:Cls.method' or 'RET:factory.method' -> the method on the inferred receiver class."""
        kind, _, rest = expr.partition(":")
        receiver, _, member = rest.rpartition(".")
        if not receiver:
            return self.resolve_symbol(rest, ctx)
        if kind == "RET":
            fn = self.resolve_symbol(receiver, ctx)
            if not fn or fn[0] not in self.symbols:
                return self._name_match(member, methods=True)
            returns = self.symbols[fn[0]]["attrs"].get("returns")
            if not returns:
                return self._name_match(member, methods=True)
            fn_mod = self._module_of(fn[0])
            cls_hit = self.resolve_symbol(returns.strip("'\""), {"module": fn_mod})
        else:
            cls_hit = self.resolve_symbol(receiver, ctx)
        if cls_hit and cls_hit[0] in self.symbols and self.symbols[cls_hit[0]]["kind"] == "class":
            c = self.symbols[cls_hit[0]]
            hit = self._method(self._module_of(c["id"]), c["name"], member)
            if hit:
                return hit, ev.STATIC_RESOLVED, 0.9
        return self._name_match(member, methods=True)

    def _module_of(self, symbol_id: str) -> str:
        return symbol_id.split(":")[1]

    def _method(self, module: str, cls: str, name: str, depth: int = 0) -> str | None:
        qual = f"{module}.{cls}.{name}"
        if qual in self.by_qual:
            return self.by_qual[qual]
        cls_id = self.by_qual.get(f"{module}.{cls}")
        if cls_id and depth < 3:
            for base in self.symbols[cls_id]["attrs"].get("bases", []):
                r = self.resolve_symbol(base, {"module": module})
                if r and r[0] in self.symbols:
                    b = self.symbols[r[0]]
                    hit = self._method(self._module_of(b["id"]), b["name"], name, depth + 1)
                    if hit:
                        return hit
        return None

    def _name_match(self, name: str, methods: bool) -> tuple[str, str, float] | None:
        if name in CALL_STOPLIST or name.startswith("__") or len(name) < 3:
            return None
        kinds = ("method",) if methods else ("function", "class")
        cands = [c for c in self.by_name.get(name, []) if self.symbols[c]["kind"] in kinds]
        if len(cands) == 1:
            return cands[0], ev.NAME_MATCH, 0.6
        non_test = [c for c in cands if not self.symbols[c]["attrs"].get("symbol_type") == "test"
                    and not is_test_path(self.symbols[c]["file"] or "")]
        if len(non_test) == 1:
            return non_test[0], ev.NAME_MATCH, 0.5
        if 1 < len(non_test) <= 8:
            self.last_ambiguous = non_test
        return None

    def resolve_doc_mention(self, name: str) -> str | None:
        name = name.replace("::", ".").rstrip("()")
        if name in self.by_qual:
            return self.by_qual[name]
        if name in self.files:
            return f"file:{name}"
        for qual, nid in self.by_qual.items():
            if qual.endswith("." + name) and nid in self.symbols and self.symbols[nid]["kind"] in ("class", "function"):
                return nid
        cands = [c for c in self.by_name.get(name.split(".")[-1], [])
                 if self.symbols[c]["kind"] in ("class", "function", "method")]
        if "." in name:
            cands = [c for c in cands if (self.symbols[c]["qualname"] or "").endswith(name)]
        return cands[0] if len(cands) == 1 else None


CLAIM_TYPES = {"depends on": "depends_on", "uses": "depends_on", "calls": "calls", "imports": "imports",
               "reads from": "reads", "writes to": "writes", "invokes": "invokes"}


def resolve(store: Store, revision: str, ambiguous: list[dict[str, Any]] | None = None) -> dict[str, int]:
    """Turn unresolved references into typed edges. Recomputed from refs every scan.

    References that match several symbols by name are collected in `ambiguous` for the
    ML (or heuristic) disambiguation pass."""
    store.delete_edges("json_extract(attrs, '$.resolved') = 1")
    r = Resolver(store)
    resolved = unresolved = 0
    for ref in store.refs():
        typ, src, target, ctx, line, owner = ref["type"], ref["src"], ref["target"], ref["context"], ref["line"], \
            ref["owner_file"]
        attrs: dict[str, Any] = {"resolved": True}
        if typ == "imports":
            dst, evidence = r.resolve_module_import(target, ctx.get("name"))
            if not dst or dst == src:
                unresolved += 1
                continue
            if dst.startswith("package:"):
                pkg = dst.split(":", 1)[1]
                existing = store.node(dst)
                if not existing or existing["evidence"] != ev.MANIFEST:
                    stdlib = _is_stdlib(pkg, owner)
                    store.upsert_node(dst, "package", pkg, evidence=ev.AST, revision=revision,
                                      attrs={"external": True, "stdlib": stdlib})
            store.add_edge(src, dst, "imports", evidence=evidence, file=owner, line=line, owner_file=owner,
                           revision=revision, attrs=attrs)
            if (store.node(src) or {}).get("attrs", {}).get("test") and dst.startswith("module:"):
                store.add_edge(src, dst, "verifies", evidence=ev.HEURISTIC, confidence=0.6, file=owner, line=line,
                               owner_file=owner, revision=revision, attrs={**attrs, "via": "test import"})
            resolved += 1
        elif typ in ("calls", "inherits", "implements", "references", "routes_to", "raises", "invokes"):
            r.last_ambiguous = []
            hit = r.resolve_symbol(target, ctx)
            if not hit or hit[0] == src:
                if ambiguous is not None and r.last_ambiguous and typ == "calls":
                    ambiguous.append({"src": src, "candidates": [c for c in r.last_ambiguous if c != src],
                                      "type": typ, "file": owner, "line": line})
                unresolved += 1
                continue
            dst, evidence, conf = hit
            kind = r.symbols.get(dst, {}).get("kind")
            etype = typ
            if typ == "calls" and kind == "class":
                etype = "instantiates"
            if typ == "references" and ctx.get("injected"):
                etype = "depends_on"
                attrs["injected"] = True
            store.add_edge(src, dst, etype, evidence=evidence, confidence=conf, file=owner, line=line,
                           owner_file=owner, revision=revision, attrs=attrs)
            if etype == "instantiates" and f"{dst}.__init__" in r.symbols:
                store.add_edge(src, f"{dst}.__init__", "calls", evidence=evidence, confidence=conf, file=owner,
                               line=line, owner_file=owner, revision=revision, attrs={**attrs, "via": "constructor"})
            src_node = r.symbols.get(src)
            if typ == "calls" and src_node and src_node["kind"] == "test":
                store.add_edge(src, dst, "verifies", evidence=evidence, confidence=conf * 0.9, file=owner, line=line,
                               owner_file=owner, revision=revision, attrs={**attrs, "via": "test call"})
            resolved += 1
        elif typ in ("mentions", "links"):
            name = target
            if typ == "links":
                base = PurePosixPath(ctx.get("doc", "")).parent
                name = str(PurePosixPath(base, target)) if not target.startswith("/") else target.lstrip("/")
                name = _normpath(name)
            dst = r.resolve_doc_mention(name)
            if dst:
                store.add_edge(src, dst, "documents", evidence=ev.DOCUMENTATION, confidence=0.7, file=owner,
                               line=line, owner_file=owner, revision=revision, attrs=attrs)
                resolved += 1
            else:
                unresolved += 1
        elif typ == "claims":
            a = r.resolve_doc_mention(ctx.get("subject", ""))
            b = r.resolve_doc_mention(target)
            if a and b and a != b:
                store.add_edge(a, b, CLAIM_TYPES.get(ctx.get("verb", ""), "depends_on"), evidence=ev.DOCUMENTATION,
                               confidence=0.7, file=owner, line=line, owner_file=owner, revision=revision,
                               attrs={**attrs, "claim": ctx.get("sentence"), "claimed_by": src})
                resolved += 1
            else:
                unresolved += 1
    resolved += _overrides(store, r, revision)
    return {"resolved_refs": resolved, "unresolved_refs": unresolved}


def _overrides(store: Store, r: Resolver, revision: str) -> int:
    """Subclass methods that override a base-class method (static view of polymorphic dispatch)."""
    count = 0
    for e in store.edges(types=["inherits", "implements"]):
        sub, base = r.symbols.get(e["src"]), r.symbols.get(e["dst"])
        if not sub or not base:
            continue
        for m in set(sub["attrs"].get("methods", [])) & set(base["attrs"].get("methods", [])):
            if m.startswith("__"):
                continue
            s_id, b_id = f"{sub['id']}.{m}", f"{base['id']}.{m}"
            if s_id in r.symbols and b_id in r.symbols:
                store.add_edge(s_id, b_id, "overrides", evidence=e["evidence"], file=sub["file"],
                               line=r.symbols[s_id]["line_start"], owner_file=e["owner_file"], revision=revision,
                               attrs={"resolved": True})
                count += 1
    return count


NODE_BUILTINS = {"fs", "path", "http", "https", "url", "os", "crypto", "events", "stream", "util", "child_process",
                 "net", "zlib", "buffer", "assert", "readline", "querystring", "worker_threads", "cluster", "dns",
                 "tls", "process", "timers", "vm"}


def _is_stdlib(pkg: str, owner: str | None) -> bool:
    import sys

    if (owner or "").endswith((".py", ".pyi")):
        return pkg.replace("-", "_") in getattr(sys, "stdlib_module_names", ()) or pkg in ("__future__",)
    return pkg in NODE_BUILTINS


def _normpath(p: str) -> str:
    out: list[str] = []
    for part in PurePosixPath(p).parts:
        if part == "..":
            if out:
                out.pop()
        elif part != ".":
            out.append(part)
    return "/".join(out)


# ------------------------------------------------------------------------------------- ML passes

def classify_pass(store: Store, rt: Any, revision: str) -> dict[str, int]:
    """Two-tier classification of modules (layer), unclassified library calls, doc sections and notes."""
    clf = rt.classifier
    counts = {"layer": 0, "resource": 0, "doc_intent": 0, "note_intent": 0}
    for m in store.nodes("module"):
        symbols = [e["dst"].rsplit(":", 1)[-1] for e in store.edges(src=m["id"], types=["defines"])][:15]
        text = f"{m['name']} {(m['text'] or '')[:300]} {' '.join(symbols)}"
        c = clf.classify("layer", text)
        attrs = dict(m["attrs"])
        attrs["layer"] = c.as_dict()
        _update_attrs(store, m["id"], attrs)
        counts["layer"] += 1

    store.delete_edges("evidence=? AND json_extract(attrs, '$.classifier_task')='resource'", (ev.ML_CLASSIFIER,))
    seen: set[tuple[str, str]] = set()
    for ref in store.refs():
        if ref["type"] != "external_call":
            continue
        lib = ref["context"].get("library", "")
        call = ref["target"].split(".", 1)[-1]
        c = clf.classify("resource", f"{lib} {call}")
        if c.label in ("compute", "unknown") or c.confidence < 0.5:
            continue
        key = (ref["src"], c.label)
        if key in seen:
            continue
        seen.add(key)
        rid = f"resource:{c.label}"
        if not store.node(rid):
            store.upsert_node(rid, "resource", c.label, evidence=ev.ML_CLASSIFIER, attrs={"category": c.label})
        verb = "writes" if re.search(r"(write|save|insert|add|put|post|send|publish|update|delete|create|set)",
                                     call, re.I) else "reads"
        store.add_edge(ref["src"], rid, verb, evidence=ev.ML_CLASSIFIER, confidence=round(c.confidence * 0.9, 3),
                       file=ref["owner_file"], line=ref["line"], owner_file=ref["owner_file"], revision=revision,
                       attrs={"call": ref["target"], "library": lib, "classifier_task": "resource",
                              "classification": c.as_dict()})
        counts["resource"] += 1

    for s in store.nodes("doc_section"):
        c = clf.classify("doc_intent", f"{s['name']}\n{(s['text'] or '')[:400]}")
        attrs = dict(s["attrs"])
        attrs["intent"] = c.as_dict()
        _update_attrs(store, s["id"], attrs)
        counts["doc_intent"] += 1
    for n in store.nodes("code_note"):
        c = clf.classify("note_intent", n["text"] or n["name"])
        attrs = dict(n["attrs"])
        attrs["category"] = c.label
        attrs["classification"] = c.as_dict()
        _update_attrs(store, n["id"], attrs)
        counts["note_intent"] += 1
    return counts


def ml_enabled(rt: Any) -> bool:
    return rt.classifier.mode != "heuristic"


def graph_ml_pass(store: Store, rt: Any, ambiguous: list[dict[str, Any]], revision: str) -> dict[str, Any]:
    """Embeddings, graph propagation, similarity edges and disambiguation (ML), or heuristics with static
    confidence when ML is disabled."""
    from promapkin.ml import graph_ml

    if not ml_enabled(rt):
        return {"embedded": 0,
                "similar_edges": graph_ml.similarity_edges(store, None, revision=revision),
                "disambiguated": graph_ml.disambiguate(store, ambiguous, None, revision),
                "graph_ml": "heuristic (static confidence)"}
    embedded = embed_pass(store, rt)
    base = rt.embedder.name
    graph_ml.propagate(store, base)
    gmodel = graph_ml.graph_model_name(base)
    threshold = 0.75 if rt.embedder.neural else 0.6
    return {"embedded": embedded,
            "similar_edges": graph_ml.similarity_edges(store, gmodel, threshold=threshold, revision=revision),
            "disambiguated": graph_ml.disambiguate(store, ambiguous, gmodel, revision),
            "graph_ml": f"propagated {base}"}


def _update_attrs(store: Store, node_id: str, attrs: dict[str, Any]) -> None:
    store.conn.execute("UPDATE nodes SET attrs=? WHERE id=?", (json.dumps(attrs, sort_keys=True), node_id))


EMBED_KINDS = ("module", "class", "function", "method", "endpoint", "doc_section", "decision", "finding",
               "code_note", "test", "pattern", "concept", "roadmap_step", "handoff")


def embedding_text(n: dict[str, Any]) -> str:
    parts = [n["name"], n.get("qualname") or "", n.get("kind") or "", (n.get("text") or "")[:500]]
    params = n["attrs"].get("params") if isinstance(n.get("attrs"), dict) else None
    if params:
        parts.append(" ".join(p["name"] for p in params))
    return " ".join(p for p in parts if p)


def embed_pass(store: Store, rt: Any) -> int:
    from promapkin.ml.embedders import pack

    total = 0
    kinds = ",".join(f"'{k}'" for k in EMBED_KINDS)
    nodes = store.nodes(where=f"kind IN ({kinds})")
    for emb in rt.embedders():
        have = store.embedding_shas(emb.name)
        todo = []
        for n in nodes:
            text = embedding_text(n)
            sha = hashlib.sha1(text.encode()).hexdigest()
            if have.get(n["id"]) != sha:
                todo.append((n["id"], sha, text))
        for i in range(0, len(todo), 64):
            batch = todo[i:i + 64]
            vecs = emb.embed([t for _, _, t in batch])
            for (nid, sha, _), vec in zip(batch, vecs):
                store.put_embedding(nid, emb.name, sha, pack(vec))
        total += len(todo)
    store.prune_embeddings()
    return total


# ------------------------------------------------------------------------------------- freshness

def _mark_stale(store: Store, changed_files: set[str]) -> int:
    """Agent/handoff knowledge that cites changed files becomes stale until re-verified."""
    if not changed_files:
        return 0
    marked = 0
    inferred = store.nodes(where="evidence IN (?,?,?) AND status='verified'",
                           params=(ev.AGENT_INFERENCE, ev.HANDOFF, ev.RUNTIME))
    for n in inferred:
        cited = set(n["attrs"].get("files", []))
        if n.get("file"):
            cited.add(n["file"])
        for e in store.edges(src=n["id"]):
            tgt = store.node(e["dst"])
            if tgt and tgt.get("file"):
                cited.add(tgt["file"])
        if cited & changed_files:
            store.set_node_status(n["id"], "stale")
            store.conn.execute("UPDATE edges SET status='stale' WHERE src=? AND evidence IN (?,?,?)",
                               (n["id"], ev.AGENT_INFERENCE, ev.HANDOFF, ev.RUNTIME))
            marked += 1
    return marked


def _gc_shared(store: Store) -> None:
    kinds = ",".join(f"'{k}'" for k in SHARED_KINDS)
    store.delete_nodes(f"kind IN ({kinds}) AND owner_file IS NULL AND id NOT IN (SELECT dst FROM edges) "
                       f"AND id NOT IN (SELECT src FROM edges)")


def stale_files(ws: Workspace) -> dict[str, list[str]]:
    """Files whose content differs from the index (without rescanning)."""
    old = ws.store.file_index()
    modified, added = [], []
    current = set()
    for rel in _candidate_files(ws):
        if not language_of(rel):
            continue
        current.add(rel)
        try:
            data = (ws.root / rel).read_bytes()
        except OSError:
            continue
        if rel not in old:
            if len(data) <= MAX_FILE_BYTES and not is_probably_binary(data):
                added.append(rel)
        elif old[rel] != gitutil.sha_of(data):
            modified.append(rel)
    return {"modified": modified, "added": added, "deleted": [p for p in old if p not in current]}
