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
import subprocess
import json
import os
import re
import time
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from promapkin import evidence as ev
from promapkin import gitutil
from promapkin.config import Workspace
from promapkin.extract import ASSET_LANGUAGES, NON_CODE, SNIFF, Facts, data_links, extractor_for, is_generated, is_test_path, \
    language_of
from promapkin.security import MAX_FILE_BYTES, is_probably_binary
from promapkin.store import Store

SHARED_KINDS = {"package", "resource", "config", "service", "event"}
TASK_QUEUE_SUFFIXES = {"delay", "apply_async", "s", "si", "send", "kiq", "enqueue"}
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


class _Timer:
    """Per-phase wall-clock timings reported in the scan result (and streamed to progress)."""

    def __init__(self, say: Callable[[str], None]):
        self.say = say
        self.timings: dict[str, float] = {}
        self._t = time.perf_counter()

    def mark(self, phase: str) -> None:
        now = time.perf_counter()
        self.timings[phase] = round(self.timings.get(phase, 0.0) + now - self._t, 3)
        self._t = now
        self.say(f"  {phase}: {self.timings[phase]:.2f}s")


def scan(ws: Workspace, *, full: bool = False, progress: Callable[[str], None] | None = None,
         interactive: bool = False, ask: Callable[[str], bool] | None = None) -> ScanResult:
    from promapkin import ml

    store = ws.store
    revision = ws.revision()
    started = time.time()
    job = store.start_job("scan", revision)
    say = progress or (lambda _m: None)
    timer = _Timer(say if os.environ.get("PROMAPKIN_TIMINGS") else (lambda _m: None))
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
            if lang in ASSET_LANGUAGES:
                asset = _asset_fingerprint(path, lang)
                if asset:
                    contents[rel] = asset
                continue
            try:
                if path.stat().st_size > MAX_FILE_BYTES:
                    continue
                data = path.read_bytes()
            except OSError:
                continue
            if is_probably_binary(data):
                continue
            if lang in SNIFF:   # by content: .m (Objective-C/MATLAB), .v (Verilog/V), YAML/JSON infrastructure
                lang = language_of(rel, data[:4000].decode("utf-8", "replace"))
                if not lang:
                    continue
            contents[rel] = (gitutil.sha_of(data), lang, len(data))
        changed = [p for p, (sha, _, _) in contents.items() if full or old.get(p) != sha]
        deleted = [p for p in old if p not in contents]
        say(f"{len(contents)} files indexed, {len(changed)} changed, {len(deleted)} deleted")

        errors: list[str] = []
        timer.mark("discover")
        incremental = not full and bool(store.get_meta("indexed_revision"))
        touched = set(changed) | set(deleted)
        llm = getattr(ws, "llm", None)
        # Parse before taking the write lock: parsers and LLM extraction are the slow part of a scan, and
        # other promapkin processes (MCP server, a second terminal) must not wait on them.
        parsed = _batch_parse(ws, [p for p in changed if contents[p][1] in ("javascript", "typescript")], errors)
        extracted: dict[str, tuple[str, bool, Facts | None, str | None]] = {}
        todo = [p for p in changed if p not in parsed]
        workers = _scan_workers(ws, len(todo))
        if workers > 1:
            # Extraction is pure (no index access, no LLM in this mode): spread it over processes.
            extracted.update(_extract_parallel(ws, [(p, contents[p][1], contents[p][2]) for p in todo], workers,
                                               errors))
            timer.mark(f"parse ({workers} processes)")
        for p in changed:
            if p in extracted:
                continue
            if contents[p][1] in ASSET_LANGUAGES:
                extracted[p] = ("", False, *_extract_asset(ws, p, contents[p][1], contents[p][2], errors))
                continue
            text = (ws.root / p).read_text(encoding="utf-8", errors="replace")
            facts = parsed.get(p)
            parser = "typescript-compiler" if facts else None
            if facts is None:
                facts, parser = _extract_one(ws, p, contents[p][1], text, errors)
            if facts is not None and contents[p][1] not in NON_CODE:
                try:
                    data_links.annotate(facts, text, contents[p][1])
                except Exception as exc:  # noqa: BLE001 - data links are an enrichment; never lose the file
                    errors.append(f"{p}: data links skipped ({type(exc).__name__}: {str(exc)[:120]})")
            extracted[p] = (text, is_generated(p, text), facts, parser)
        timer.mark("parse")
        with store.transaction():
            # Capture what other files may depend on before the old facts disappear.
            old_names = _defined_names(store, touched) if incremental else set()
            inbound = _inbound_files(store, touched) if incremental else set()
            for p in deleted:
                store.forget_file(p)
            parsers: dict[str, int] = {}
            for p in changed:
                store.forget_file(p)
                sha, lang, size = contents[p]
                text, generated, facts, parser = extracted[p]
                store.upsert_file(p, sha, lang, size, revision)
                store.upsert_node(f"file:{p}", "file", PurePosixPath(p).name, evidence=ev.AST, file=p,
                                  owner_file=p, revision=revision,
                                  attrs={"language": lang, "size": size, "sha": sha, "test": is_test_path(p),
                                         "generated": generated})
                if facts is None:
                    continue
                parsers[parser or lang] = parsers.get(parser or lang, 0) + 1
                if generated:
                    for n in facts.nodes:
                        if n["kind"] == "module":
                            n.setdefault("attrs", {})["generated"] = True
                errors += [f"{p}: {e}" for e in facts.errors]
                _persist(store, facts, revision)
            timer.mark("extract")
            _structure(store, ws, list(contents), revision)
            timer.mark("structure")
            ambiguous: list[dict[str, Any]] = []
            scope = None
            if incremental:
                names = old_names | _defined_names(store, touched)
                scope = touched | inbound | _files_referencing(store, names)
            stats = resolve(store, revision, ambiguous, files=scope)
            timer.mark("resolve")
            warnings: list[str] = []
            if (changed or deleted or full) and llm is None:
                stats["classified"] = _optional("classification", warnings, classify_pass, store, rt, revision) or {}
            timer.mark("classify")
            stats["stale_inferences"] = _mark_stale(store, set(changed) | set(deleted)) if not full else 0
            _gc_shared(store)
            timer.mark("freshness")
            stats.update(_optional("graph ML", warnings, graph_ml_pass, store, rt, ambiguous, revision,
                                   timer) or {})
            stats["warnings"] = warnings
            stats["parsers"] = parsers
            stats["runtime_edges_stale"] = _stale_runtime_edges(store, touched) if incremental else 0
            _prune_dangling_edges(store)
            store.conn.execute("DELETE FROM meta WHERE key LIKE 'ctx:%'")   # context packages are revision-bound
            store.set_meta("indexed_revision", revision)
            store.set_meta("indexed_at", str(time.time()))
            store.set_meta("ml", json.dumps(rt.describe()))
            timer.mark("finalize")
        if (changed or deleted or full) and llm is not None:
            # --llm: ask the model with no lock held (answers land in the response cache), then apply them in
            # a short transaction that never goes to the network. The structural index above is already
            # committed; classification is an enrichment pass, as without --llm.
            _optional_unlocked("LLM classification", warnings, classify_pass, store, rt, revision, llm, True)
            with store.transaction(), llm.cache_only():
                stats["classified"] = _optional("classification", warnings, classify_pass, store, rt, revision,
                                                llm) or {}
                _gc_shared(store)
            timer.mark("classify (llm)")
        stats["timings"] = timer.timings
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


LEXICAL_PARSERS = {"javascript", "typescript"}   # languages whose fallback extractor is lexical


def _optional(name: str, warnings: list[str], fn: Callable[..., Any], *args: Any) -> Any:
    """Enrichment passes are optional: a failure is reported, never allowed to discard the structural index."""
    sp = f"pass_{abs(hash(name))}"
    store = args[0]
    store.conn.execute(f"SAVEPOINT {sp}")
    try:
        result = fn(*args)
    except Exception as exc:  # noqa: BLE001
        store.conn.execute(f"ROLLBACK TO {sp}")
        store.conn.execute(f"RELEASE {sp}")
        warnings.append(f"{name} pass skipped: {type(exc).__name__}: {str(exc)[:300]}")
        return None
    store.conn.execute(f"RELEASE {sp}")
    return result


def _optional_unlocked(name: str, warnings: list[str], fn: Callable[..., Any], *args: Any) -> Any:
    """Like _optional, for passes that only read the index (no savepoint needed)."""
    try:
        return fn(*args)
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"{name} pass skipped: {type(exc).__name__}: {str(exc)[:300]}")
        return None


def _batch_parse(ws: Workspace, js_files: list[str], errors: list[str]) -> dict[str, Facts]:
    """JS/TS through the TypeScript compiler in one batch (cross-file type resolution), when installed."""
    from promapkin import toolchain

    if not js_files or not toolchain.typescript_available(ws.root):
        return {}
    from promapkin.extract import ts_ext
    try:
        return ts_ext.extract_batch(ws.root, js_files)
    except (ts_ext.BridgeError, OSError, ValueError, subprocess.TimeoutExpired) as exc:
        errors.append(f"typescript parser unavailable, using fallback: {str(exc)[:300]}")
        return {}


def _extract_one(ws: Workspace, path: str, lang: str, text: str, errors: list[str]) -> tuple[Facts | None, str]:
    """Pick the extractor for one file: a real parser when available; otherwise the LLM in --llm mode,
    otherwise the lexical fallback."""
    from promapkin import toolchain
    from promapkin.extract import TREE_SITTER_LANGUAGES

    llm = getattr(ws, "llm", None)
    lexical = lang in LEXICAL_PARSERS or (lang in TREE_SITTER_LANGUAGES and not toolchain.tree_sitter_available())
    if lexical and llm is not None:
        try:
            return llm.extract_code(path, lang, text), "llm"
        except Exception as exc:  # noqa: BLE001 - fall back rather than lose the file
            errors.append(f"{path}: LLM extraction failed ({str(exc)[:200]}); using lexical fallback")
    extractor = extractor_for(lang)
    if extractor is None:
        return None, ""
    try:
        facts = extractor(path, text)
    except Exception as exc:  # noqa: BLE001 - one bad file never aborts a scan
        errors.append(f"{path}: {exc}")
        return None, ""
    if lang in TREE_SITTER_LANGUAGES:
        parser = "tree-sitter" if toolchain.tree_sitter_available() else "lexical"
    elif lang in LEXICAL_PARSERS:
        parser = "lexical"
    else:
        parser = {"python": "python-ast", "markdown": "markdown", "manifest": "manifest"}.get(lang, lang)
    if llm is not None and lang == "markdown":
        try:
            llm.enrich_document(path, text, facts)
            parser = "markdown+llm"
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{path}: LLM document analysis failed ({str(exc)[:200]})")
    return facts, parser


PARALLEL_MIN_FILES = 200


def _scan_workers(ws: Workspace, n_files: int) -> int:
    if getattr(ws, "llm", None) is not None or n_files < PARALLEL_MIN_FILES:
        return 1
    env = os.environ.get("PROMAPKIN_WORKERS")
    want = int(env) if env and env.isdigit() else ws.config.scan_workers
    if want <= 0:
        want = min(8, os.cpu_count() or 1)
    return max(1, min(want, n_files // max(1, PARALLEL_MIN_FILES // 4)))


def _extract_parallel(ws: Workspace, jobs: list[tuple[str, str, int]], workers: int, errors: list[str]
                      ) -> dict[str, tuple[str, bool, Facts | None, str | None]]:
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing

    out: dict[str, tuple[str, bool, Facts | None, str | None]] = {}
    chunks = [jobs[i::workers * 4] for i in range(workers * 4)]
    ctx = multiprocessing.get_context("spawn" if os.name == "nt" else "fork")
    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            for results, errs in pool.map(_extract_chunk, [(str(ws.root), c) for c in chunks if c]):
                out.update(results)
                errors.extend(errs)
    except (OSError, RuntimeError) as exc:   # no fork/spawn available: extract in this process instead
        errors.append(f"parallel extraction unavailable ({exc}); extracted sequentially")
        return {}
    return out


def _extract_chunk(arg: tuple[str, list[tuple[str, str, int]]]
                   ) -> tuple[dict[str, tuple[str, bool, Facts | None, str | None]], list[str]]:
    root, jobs = arg
    from promapkin.config import Config
    from types import SimpleNamespace
    ws = SimpleNamespace(root=Path(root), llm=None, config=Config.load(Path(root) / ".promapkin" / "config.json"))
    errors: list[str] = []
    results = {}
    for rel, lang, size in jobs:
        if lang in ASSET_LANGUAGES:
            results[rel] = ("", False, *_extract_asset(ws, rel, lang, size, errors))  # type: ignore[arg-type]
            continue
        try:
            text = (ws.root / rel).read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            errors.append(f"{rel}: {exc}")
            continue
        facts, parser = _extract_one(ws, rel, lang, text, errors)  # type: ignore[arg-type]
        if facts is not None and lang not in NON_CODE:
            try:
                data_links.annotate(facts, text, lang)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{rel}: data links skipped ({type(exc).__name__}: {str(exc)[:120]})")
        results[rel] = (text, is_generated(rel, text), facts, parser)
    return results, errors


def _asset_fingerprint(path: Path, lang: str) -> tuple[str, str, int] | None:
    """(sha, language, size) for a binary asset. Video is never read whole: its identity is size, mtime and header."""
    from promapkin.extract.assets_ext import MAX_ASSET_BYTES
    try:
        st = path.stat()
        if lang == "video":
            with path.open("rb") as fh:
                head = fh.read(65536)
            return gitutil.sha_of(f"{st.st_size}:{int(st.st_mtime)}:".encode() + head), lang, st.st_size
        if st.st_size > MAX_ASSET_BYTES[lang] or st.st_size == 0:
            return None
        return gitutil.sha_of(path.read_bytes()), lang, st.st_size
    except OSError:
        return None


def _extract_asset(ws: Workspace, rel: str, lang: str, size: int, errors: list[str]) -> tuple[Facts | None, str]:
    from promapkin.extract import assets_ext
    path = ws.root / rel
    try:
        if lang == "video":
            with path.open("rb") as fh:
                data = fh.read(4_000_000)      # enough for the MP4 `moov` box in streaming-friendly files
        else:
            data = path.read_bytes()
    except OSError as exc:
        errors.append(f"{rel}: {exc}")
        return None, ""
    llm = getattr(ws, "llm", None) if ws.config.llm_images else None
    facts = assets_ext.extract(rel, data, lang, llm=llm, size=size)
    return facts, f"asset:{lang}"


def _stale_runtime_edges(store: Store, touched: set[str]) -> int:
    """Runtime-observed edges touching changed code are kept but marked stale until re-traced."""
    if not touched:
        return 0
    n = 0
    for chunk in _chunked(sorted(touched), 400):
        marks = ",".join("?" * len(chunk))
        n += store.conn.execute(
            f"UPDATE edges SET status='stale' WHERE evidence=? AND status='verified' AND ("
            f"src IN (SELECT id FROM nodes WHERE owner_file IN ({marks})) OR "
            f"dst IN (SELECT id FROM nodes WHERE owner_file IN ({marks})))",
            [ev.RUNTIME, *chunk, *chunk]).rowcount
    store.conn.execute("DELETE FROM edges WHERE evidence=? AND (src NOT IN (SELECT id FROM nodes) OR "
                       "dst NOT IN (SELECT id FROM nodes))", (ev.RUNTIME,))
    return n


def _prune_dangling_edges(store: Store) -> int:
    """Edges whose endpoints vanished (cross-file renames, extractor gaps) would otherwise be
    reported by `verify` forever; dropped once extraction and resolution are complete."""
    return store.conn.execute(
        "DELETE FROM edges WHERE src NOT IN (SELECT id FROM nodes) OR dst NOT IN (SELECT id FROM nodes)"
    ).rowcount


def _candidate_files(ws: Workspace) -> list[str]:
    files = gitutil.list_files(ws.root)
    inc, exc = ws.config.include, ws.config.exclude
    if inc:
        files = [f for f in files if any(fnmatch.fnmatch(f, pat) for pat in inc)]
    if exc:
        files = [f for f in files if not any(fnmatch.fnmatch(f, pat) for pat in exc)]
    sources = ws.dir / "sources"          # papers, transcripts and pages added with `promapkin add`
    if sources.is_dir():
        have = set(files)
        files += sorted(p.relative_to(ws.root).as_posix() for p in sources.rglob("*")
                        if p.is_file() and p.name != "index.json" and p.relative_to(ws.root).as_posix() not in have)
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
        self.aliases: dict[str, dict[str, str]] = {}
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

    # ---------------------------------------------------------------- path-style modules (Rust, Java, Go, C#, ...)
    def _path_index(self) -> dict[str, list[str]]:
        if not hasattr(self, "_suffixes"):
            idx: dict[str, list[str]] = {}
            for name, n in self.modules.items():
                if "/" not in name:
                    continue
                parts = name.split("/")
                if parts[-1] in ("mod", "lib", "main", "index", "__init__"):
                    parts = parts[:-1]
                for k in range(1, min(len(parts), 6) + 1):
                    idx.setdefault("/".join(parts[-k:]), []).append(n["id"])
            self._suffixes = idx
            crates = {}
            for m in self.store.nodes("manifest"):
                crate = m["attrs"].get("crate")
                if crate and m.get("file"):
                    parent = str(PurePosixPath(m["file"]).parent)
                    crates[crate.replace("-", "_")] = "" if parent == "." else parent
            self.crates = crates
        return self._suffixes

    def _crate_dir(self, module: str) -> str:
        best = ""
        for d in getattr(self, "crates", {}).values():
            if (not d or module.startswith(d + "/")) and len(d) >= len(best):
                best = d
        return best

    def resolve_path_import(self, target: str, from_module: str) -> tuple[str | None, list[str]]:
        """'crate.builder.command.Command' / 'com.acme.db.Repo' / 'github.com/x/app/internal/db' ->
        (module id, remaining item segments). Candidates are ranked by directory proximity."""
        idx = self._path_index()
        segs = [s for s in re.split(r"[./]", target) if s and s != "*"]
        if not segs:
            return None, []
        scope = ""
        crate_scoped = False
        if segs[0] in ("crate", "self", "super"):
            crate_scoped = segs[0] == "crate"
            base = self._crate_dir(from_module)
            if segs[0] == "crate":
                scope = base
            else:
                scope = str(PurePosixPath(from_module).parent)
                if segs[0] == "super":
                    scope = str(PurePosixPath(scope).parent)
            segs = segs[1:]
        elif segs[0] in getattr(self, "crates", {}):
            scope = self.crates[segs[0]]
            segs = segs[1:]
            crate_scoped = True
        for k in range(len(segs), 0, -1):
            cands = idx.get("/".join(segs[:k]), [])
            if scope:
                cands = [c for c in cands if c[len("module:"):].startswith(scope.rstrip("/") + "/")]
            if cands:
                best = max(cands, key=lambda c: _common_prefix(c[len("module:"):], from_module))
                return best, segs[k:]
        if crate_scoped:   # item re-exported from the crate root (lib.rs)
            prefix = f"{scope}/" if scope else ""
            for cand in (f"{prefix}src/lib", f"{prefix}lib", f"{prefix}src/main"):
                if cand in self.modules:
                    return self.modules[cand]["id"], segs
        return None, segs

    def aliases_for(self, module: str) -> dict[str, str]:
        return getattr(self, "aliases", {}).get(module, {})

    def resolve_module_import(self, target: str, name: str | None, from_module: str = "") -> tuple[str, str]:
        if name and (mid := self.module_for(f"{target}.{name}")):
            return mid, ev.AST
        if mid := self.module_for(target):
            return mid, ev.AST
        if "/" in from_module:
            mid, rest = self.resolve_path_import(target, from_module)
            if mid:
                if rest:   # `use a::b::Item` -> local alias Item -> <module>.Item
                    mod_name = mid[len("module:"):]
                    self.aliases.setdefault(from_module, {})[rest[-1]] = f"{mod_name}.{'.'.join(rest)}"
                return mid, ev.STATIC_RESOLVED
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
        if expr.startswith("ID:"):   # already resolved by a type-checking parser
            nid = expr[3:]
            return (nid, ev.STATIC_RESOLVED, 0.95) if nid in self.symbols else None
        if expr.startswith(("TYPE:", "RET:")):
            return self._resolve_typed(expr, ctx)
        expr = expr.replace("()", "")
        if not expr:
            return None
        module = ctx.get("module") or ""
        cls = ctx.get("cls")
        mod_node = self.modules.get(module)
        imports: dict[str, str] = (mod_node or {}).get("attrs", {}).get("imports", {}) if mod_node else {}
        if "/" in module and module in getattr(self, "aliases", {}):
            imports = {**self.aliases[module], **imports}
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
            if head not in self.aliases_for(module):
                return None   # resolved to an external or unknown target; the import edge records the package
            # `use` alias whose item is re-exported elsewhere: fall back to a unique type/function name.
            return self._type_member(parts) or self._name_match(parts[-1], methods=len(parts) > 1)

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
        return self._type_member(parts) or self._name_match(parts[-1], methods=len(parts) > 1)

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

    def resolve_fixture(self, name: str) -> tuple[str, str, float] | None:
        cands = [c for c in self.by_name.get(name, [])
                 if any("fixture" in d for d in self.symbols[c]["attrs"].get("decorators", []))]
        return (cands[0], ev.HEURISTIC, 0.8) if len(cands) == 1 else None

    def _module_of(self, symbol_id: str) -> str:
        return symbol_id.split(":")[1]

    def _type_member(self, parts: list[str]) -> tuple[str, str, float] | None:
        """`Type::method` / `Type.method` where Type names exactly one class in the repo."""
        if len(parts) < 2 or not parts[-2][:1].isupper():
            return None
        classes = [c for c in self.by_name.get(parts[-2], []) if self.symbols[c]["kind"] == "class"]
        non_test = [c for c in classes if not is_test_path(self.symbols[c]["file"] or "")]
        if len(non_test) == 1:
            c = self.symbols[non_test[0]]
            hit = self._method(self._module_of(c["id"]), c["name"], parts[-1])
            if hit:
                return hit, ev.NAME_MATCH, 0.75
        return None

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

    # ---------------------------------------------------------------- infrastructure / data links
    def resolve_link(self, target: str, ctx: dict[str, Any], revision: str) -> str | None:
        """Targets of `link` references: ID:<node id>, TABLE:<name>, PRISMA:<model>, MODULE:<dotted>, FILE:<path>,
        SYMBOL:<name>, LABELS:<k=v,...> (Kubernetes selectors) or API:<METHOD path> (spec -> code endpoint)."""
        kind, _, key = target.partition(":")
        if kind == "ID":
            return key if self.store.node(key) else None
        if kind == "TABLE":
            for cand in (f"table:{key}", f"table:{key.rsplit('.', 1)[-1]}"):
                if self.store.node(cand):
                    return cand
            if ctx.get("create"):
                tid = f"table:{key}"
                self.store.upsert_node(tid, "table", key, evidence=ctx.get("evidence") or ev.LEXICAL, confidence=0.6,
                                       revision=revision, attrs={"inferred_from": "code", "columns": []})
                return tid
            return None
        if kind == "PRISMA":
            hit = self._prisma_models().get(key.lower())
            return hit
        if kind == "MODULE":
            if key in self.modules:
                return self.modules[key]["id"]
            dst, _evidence = self.resolve_module_import(key, None, "")
            return dst if dst and dst.startswith("module:") else None
        if kind == "FILE":
            path = _normpath(key)
            if path in self.files:
                return f"file:{path}"
            hits = [f for f in self.files if f.endswith("/" + path)]
            return f"file:{hits[0]}" if len(hits) == 1 else None
        if kind == "SYMBOL":
            hit = self.resolve_symbol(key, {})
            return hit[0] if hit else None
        if kind == "LABELS":
            want = dict(kv.split("=", 1) for kv in key.split(",") if "=" in kv)
            hits = [n["id"] for n in self._workloads() if want and all(
                str(n["attrs"].get("labels", {}).get(k)) == v for k, v in want.items())]
            return hits[0] if len(hits) == 1 else (hits[0] if hits else None)
        if kind == "API":
            method, _, route = key.partition(" ")
            return self._endpoints().get((method.upper(), _route_shape(route)))
        return None

    def _prisma_models(self) -> dict[str, str]:
        if not hasattr(self, "_prisma"):
            self._prisma = {}
            for n in self.store.nodes("table"):
                model = n["attrs"].get("model")
                if model:
                    self._prisma[model.lower()] = n["id"]
                    self._prisma[model[:1].lower() + model[1:]] = n["id"]
        return self._prisma

    def _workloads(self) -> list[dict[str, Any]]:
        if not hasattr(self, "_workload_nodes"):
            self._workload_nodes = self.store.nodes("deployment")
        return self._workload_nodes

    def _endpoints(self) -> dict[tuple[str, str], str]:
        if not hasattr(self, "_endpoint_index"):
            self._endpoint_index = {}
            for n in self.store.nodes("endpoint"):
                method, _, route = n["name"].partition(" ")
                if route.startswith("/"):
                    self._endpoint_index.setdefault((method.upper(), _route_shape(route)), n["id"])
        return self._endpoint_index

    def resolve_doc_mention(self, name: str) -> str | None:
        name = name.replace("::", ".").rstrip("()")
        if name in self.by_qual:
            return self.by_qual[name]
        if name in self.files:
            return f"file:{name}"
        last = name.split(".")[-1]
        # by_name is keyed by the last segment, so a suffix match is a dictionary lookup, not a scan.
        suffix = sorted(c for c in self.by_name.get(last, [])
                        if self.symbols[c]["kind"] in ("class", "function")
                        and (self.symbols[c]["qualname"] or "").endswith("." + name))
        if suffix:
            non_test = [c for c in suffix if not is_test_path(self.symbols[c]["file"] or "")]
            return (non_test or suffix)[0]
        cands = [c for c in self.by_name.get(last, [])
                 if self.symbols[c]["kind"] in ("class", "function", "method")]
        if "." in name:
            cands = [c for c in cands if (self.symbols[c]["qualname"] or "").endswith(name)]
        return cands[0] if len(cands) == 1 else None


CLAIM_TYPES = {"depends on": "depends_on", "uses": "depends_on", "calls": "calls", "imports": "imports",
               "reads from": "reads", "writes to": "writes", "invokes": "invokes"}


def resolve(store: Store, revision: str, ambiguous: list[dict[str, Any]] | None = None,
            files: set[str] | None = None) -> dict[str, Any]:
    """Turn unresolved references into typed edges.

    `files=None` re-resolves everything. Otherwise only references owned by `files` are
    re-resolved (incremental mode): the scanner passes the changed files plus every file whose
    references could resolve differently because names were added, removed or moved.
    References that match several symbols by name are collected in `ambiguous` for the
    ML (or heuristic) disambiguation pass."""
    if files is None:
        store.delete_edges("json_extract(attrs, '$.resolved') = 1")
        refs = store.refs()
    else:
        for chunk in _chunked(sorted(files), 400):
            marks = ",".join("?" * len(chunk))
            store.delete_edges(f"json_extract(attrs, '$.resolved') = 1 AND owner_file IN ({marks})", chunk)
        refs = store.refs(files)
    r = Resolver(store)
    resolved = unresolved = 0
    # Imports first, so path-style `use` aliases are known before calls in the same module resolve.
    refs.sort(key=lambda x: x["type"] != "imports")
    for ref in refs:
        typ, src, target, ctx, line, owner = ref["type"], ref["src"], ref["target"], ref["context"], ref["line"], \
            ref["owner_file"]
        attrs: dict[str, Any] = {"resolved": True}
        if typ == "imports":
            dst, evidence = r.resolve_module_import(target, ctx.get("name"), ctx.get("module") or "")
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
            if ctx.get("wiring"):
                attrs["wiring"] = ctx["wiring"]
            store.add_edge(src, dst, "imports", evidence=evidence, file=owner, line=line, owner_file=owner,
                           revision=revision, attrs=attrs)
            if (store.node(src) or {}).get("attrs", {}).get("test") and dst.startswith("module:"):
                store.add_edge(src, dst, "verifies", evidence=ev.HEURISTIC, confidence=0.6, file=owner, line=line,
                               owner_file=owner, revision=revision, attrs={**attrs, "via": "test import"})
            resolved += 1
        elif typ in ("calls", "inherits", "implements", "references", "routes_to", "raises", "invokes"):
            r.last_ambiguous = []
            if ctx.get("fixture"):
                hit = r.resolve_fixture(target)
            elif ctx.get("exact"):
                hit = (r.by_qual[target], ev.HEURISTIC, 0.6) if target in r.by_qual else None
            else:
                hit = r.resolve_symbol(target, ctx)
                if hit is None and "." in target and target.rsplit(".", 1)[1] in TASK_QUEUE_SUFFIXES:
                    hit = r.resolve_symbol(target.rsplit(".", 1)[0], ctx)
                    if hit:
                        ctx = {**ctx, "wiring": "task queue"}
            if not hit or hit[0] == src:
                if ambiguous is not None and r.last_ambiguous and typ == "calls":
                    ambiguous.append({"src": src, "candidates": [c for c in r.last_ambiguous if c != src],
                                      "type": typ, "file": owner, "line": line})
                unresolved += 1
                continue
            dst, evidence, conf = hit
            if ctx.get("wiring"):
                # Dynamic wiring is recognised syntactically but executes indirectly: weight it as inferred.
                attrs["wiring"] = ctx["wiring"]
                if evidence in (ev.AST, ev.STATIC_RESOLVED):
                    evidence = ev.HEURISTIC
                conf = round(conf * 0.85, 3)
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
        elif typ == "link":
            dst = r.resolve_link(target, ctx, revision)
            if not dst or dst == src:
                unresolved += 1
                continue
            store.add_edge(src, dst, ctx.get("edge", "references"), evidence=ctx.get("evidence") or ev.MANIFEST,
                           confidence=float(ctx.get("confidence", 1.0)), file=owner, line=line, owner_file=owner,
                           revision=revision, attrs={**attrs, **(ctx.get("attrs") or {})})
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
    resolved += _overrides(store, r, revision, files)
    return {"resolved_refs": resolved, "unresolved_refs": unresolved,
            "resolution": "full" if files is None else f"incremental ({len(files)} files)"}


def _route_shape(route: str) -> str:
    """`/items/{id}`, `/items/:id` and `/items/<int:id>` all become `/items/{}`."""
    shape = re.sub(r"\{[^}]*\}|:[A-Za-z_]\w*|<[^>]*>", "{}", route.split("?")[0])
    return shape.rstrip("/") or "/"


def _chunked(seq: list[str], n: int) -> list[list[str]]:
    return [seq[i:i + n] for i in range(0, len(seq), n)]


def _ref_tokens(target: str) -> set[str]:
    t = re.sub(r"^(TYPE|RET):", "", target).replace("()", "")
    parts = set(re.split(r"[.:/]+", t)) | {t, t.lstrip("./")}
    return {p for p in parts if p}


def _defined_names(store: Store, files: set[str]) -> set[str]:
    """Names (short, qualified and module) defined by the given files."""
    names: set[str] = set()
    for chunk in _chunked(sorted(files), 400):
        marks = ",".join("?" * len(chunk))
        for row in store.conn.execute(
                f"SELECT name, qualname FROM nodes WHERE owner_file IN ({marks}) "
                f"AND kind IN ('module','class','function','method','test','endpoint','table','service',"
                f"'deployment','cloud_resource','iac_module','iac_variable','container_image','schema_type',"
                f"'config_source','manifest')", chunk):
            names.add(row[0])
            if row[1]:
                names.add(row[1])
                names.update(row[1].split(".")[-2:])
        names.update(chunk)
    return names


def _inbound_files(store: Store, files: set[str]) -> set[str]:
    """Files owning resolved edges that point into nodes defined by `files`."""
    out: set[str] = set()
    for chunk in _chunked(sorted(files), 400):
        marks = ",".join("?" * len(chunk))
        for row in store.conn.execute(
                f"SELECT DISTINCT e.owner_file FROM edges e JOIN nodes n ON n.id = e.dst "
                f"WHERE n.owner_file IN ({marks}) AND e.owner_file IS NOT NULL", chunk):
            out.add(row[0])
    return out


def _files_referencing(store: Store, names: set[str]) -> set[str]:
    """Files with a reference whose target mentions one of `names` (over-approximation is safe)."""
    if not names:
        return set()
    out: set[str] = set()
    for row in store.conn.execute("SELECT owner_file, target, context FROM refs"):
        if row[0] in out:
            continue
        toks = _ref_tokens(row[1])
        if toks & names:
            out.add(row[0])
            continue
        if '"subject"' in row[2]:
            subject = json.loads(row[2]).get("subject", "")
            if _ref_tokens(subject) & names:
                out.add(row[0])
    return out


def _overrides(store: Store, r: Resolver, revision: str, files: set[str] | None = None) -> int:
    """Subclass methods that override a base-class method (static view of polymorphic dispatch)."""
    count = 0
    for e in store.edges(types=["inherits", "implements"]):
        if files is not None and e["owner_file"] not in files:
            continue
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


def _common_prefix(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a.split("/"), b.split("/")):
        if x != y:
            break
        n += 1
    return n


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

def classify_pass(store: Store, rt: Any, revision: str, llm: Any = None, warm_only: bool = False) -> dict[str, Any]:
    """Classify modules (layer), unclassified library calls, doc sections and notes: two-tier classifier,
    or the LLM in --llm mode. Batched so each task is one pass over its items. With warm_only, only the
    classifier runs (filling its caches) and nothing is written: --llm scans do this outside the write lock."""
    clf = rt.classifier
    counts: dict[str, Any] = {"layer": 0, "resource": 0, "doc_intent": 0, "note_intent": 0,
                              "classifier": f"llm:{llm.label}" if llm else clf.mode}
    defines: dict[str, list[str]] = {}
    for src, dst in store.conn.execute("SELECT src, dst FROM edges WHERE type='defines' AND src LIKE 'module:%'"):
        lst = defines.setdefault(src, [])
        if len(lst) < 15:
            lst.append(dst.rsplit(":", 1)[-1])
    modules = store.nodes("module")
    calls = [r for r in store.refs() if r["type"] == "external_call"]
    uniq = sorted({(r["context"].get("library", ""), r["target"].split(".", 1)[-1]) for r in calls})
    sections = store.nodes("doc_section")
    notes = store.nodes("code_note")
    module_labels = clf.classify_many("layer", [f"{m['name']} {(m['text'] or '')[:300]} "
                                                f"{' '.join(defines.get(m['id'], []))}" for m in modules], llm)
    labels = dict(zip(uniq, clf.classify_many("resource", [f"{lib} {call}" for lib, call in uniq], llm)))
    section_labels = clf.classify_many("doc_intent", [f"{s['name']}\n{(s['text'] or '')[:400]}" for s in sections],
                                       llm)
    note_labels = clf.classify_many("note_intent", [n["text"] or n["name"] for n in notes], llm)
    if warm_only:
        return counts

    for m, c in zip(modules, module_labels):
        attrs = dict(m["attrs"])
        attrs["layer"] = c.as_dict()
        _update_attrs(store, m["id"], attrs)
        counts["layer"] += 1

    store.delete_edges("json_extract(attrs, '$.classifier_task')='resource'")
    seen: set[tuple[str, str]] = set()
    evidence = ev.LLM_INFERENCE if llm else ev.ML_CLASSIFIER
    for ref in calls:
        lib = ref["context"].get("library", "")
        call = ref["target"].split(".", 1)[-1]
        c = labels[(lib, call)]
        if c.label in ("compute", "unknown") or c.confidence < 0.5:
            continue
        key = (ref["src"], c.label)
        if key in seen:
            continue
        seen.add(key)
        rid = f"resource:{c.label}"
        if not store.node(rid):
            store.upsert_node(rid, "resource", c.label, evidence=evidence, attrs={"category": c.label})
        verb = "writes" if re.search(r"(write|save|insert|add|put|post|send|publish|update|delete|create|set)",
                                     call, re.I) else "reads"
        store.add_edge(ref["src"], rid, verb, evidence=evidence, confidence=round(c.confidence * 0.9, 3),
                       file=ref["owner_file"], line=ref["line"], owner_file=ref["owner_file"], revision=revision,
                       attrs={"call": ref["target"], "library": lib, "classifier_task": "resource",
                              "classification": c.as_dict()})
        counts["resource"] += 1

    for s_, c in zip(sections, section_labels):
        attrs = dict(s_["attrs"])
        attrs["intent"] = c.as_dict()
        _update_attrs(store, s_["id"], attrs)
        counts["doc_intent"] += 1
    for n, c in zip(notes, note_labels):
        attrs = dict(n["attrs"])
        attrs["category"] = c.label
        attrs["classification"] = c.as_dict()
        _update_attrs(store, n["id"], attrs)
        counts["note_intent"] += 1
    return counts


def ml_enabled(rt: Any) -> bool:
    return rt.classifier.mode != "heuristic"


def graph_ml_pass(store: Store, rt: Any, ambiguous: list[dict[str, Any]], revision: str,
                  timer: _Timer | None = None) -> dict[str, Any]:
    """Embeddings, graph propagation, similarity edges and disambiguation (ML), or heuristics with static
    confidence when ML is disabled."""
    from promapkin.ml import graph_ml

    if not ml_enabled(rt):
        return {"embedded": 0,
                "similar_edges": graph_ml.similarity_edges(store, None, revision=revision),
                "disambiguated": graph_ml.disambiguate(store, ambiguous, None, revision),
                "graph_ml": "heuristic (static confidence)"}
    mark = timer.mark if timer else (lambda _p: None)
    embedded = embed_pass(store, rt)
    mark("embed")
    base = rt.embedder.name
    graph_ml.propagate(store, base)
    mark("propagate")
    gmodel = graph_ml.graph_model_name(base)
    threshold = 0.75 if rt.embedder.neural else 0.6
    similar = graph_ml.similarity_edges(store, gmodel, threshold=threshold, revision=revision)
    mark("similarity")
    disamb = graph_ml.disambiguate(store, ambiguous, gmodel, revision)
    mark("disambiguate")
    return {"embedded": embedded, "similar_edges": similar, "disambiguated": disamb,
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
    kinds = ",".join(f"'{k}'" for k in SHARED_KINDS | {"table"})   # tables inferred from code have no owner
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
