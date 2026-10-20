"""Runtime evidence: observe what actually executes, to cover reflection, generated code and
framework wiring that static analysis cannot see.

Promapkin never runs repository code on its own. `promapkin trace run -- <command>` runs the
command *you* give it (typically your test suite) with an observer attached:

* Python: a profiling hook (installed through a temporary `sitecustomize`) records caller → callee
  edges between functions inside the repository, plus which test reached which function.
* Node.js: V8 coverage (`NODE_V8_COVERAGE`) records which functions executed.

The trace is then ingested as `runtime` evidence: runtime call edges, `verifies` edges from tests,
and execution counts on nodes. Edges seen only at runtime are reported as dynamic links.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from promapkin import evidence as ev
from promapkin.config import Workspace

SITECUSTOMIZE = r'''
import atexit, importlib.util, json, os, sys, threading

def _promapkin_install():
    root = os.path.realpath(os.environ["PROMAPKIN_TRACE_ROOT"])
    out = os.environ["PROMAPKIN_TRACE_OUT"]
    here = os.path.dirname(os.path.abspath(__file__))
    skip = ("site-packages", "dist-packages", os.sep + ".venv" + os.sep, os.sep + "venv" + os.sep,
            os.sep + ".promapkin" + os.sep)
    edges, tests, inside_cache = {}, {}, {}
    state = threading.local()

    def inside(fn):
        r = inside_cache.get(fn)
        if r is None:
            real = os.path.realpath(fn) if fn and not fn.startswith("<") else ""
            r = bool(real) and real.startswith(root + os.sep) and not any(s in real for s in skip)
            inside_cache[fn] = r
        return r

    real_cache = {}

    def key(code):
        fn = code.co_filename
        real = real_cache.get(fn)
        if real is None:   # absolute, so ingestion does not depend on the traced process's cwd
            real = real_cache[fn] = os.path.realpath(fn)
        return (real, getattr(code, "co_qualname", code.co_name), code.co_firstlineno)

    def prof(frame, event, arg):
        code = frame.f_code
        if event == "return":
            if getattr(state, "test", None) is not None and getattr(state, "test_frame", None) is frame:
                state.test = None
                state.test_frame = None
            return
        # Only real function frames (CO_OPTIMIZED): skips module import and class-body execution.
        if event != "call" or not (code.co_flags & 0x1) or not inside(code.co_filename):
            return
        k = key(code)
        name = code.co_name
        base = os.path.basename(code.co_filename)
        if name.startswith("test") and (base.startswith("test_") or base.endswith("_test.py")) and \
                getattr(state, "test", None) is None:
            state.test = k
            state.test_frame = frame
        caller = frame.f_back
        hops = 0
        while caller is not None and not inside(caller.f_code.co_filename):
            caller = caller.f_back
            hops += 1
        if caller is not None:
            e = (key(caller.f_code), k, hops > 0)
            edges[e] = edges.get(e, 0) + 1
        t = getattr(state, "test", None)
        if t is not None and t != k:
            tests[(t, k)] = 1

    def dump():
        sys.setprofile(None)
        data = {"pid": os.getpid(), "root": root, "python": sys.version.split()[0],
                "edges": [[list(a), list(b), via, n] for (a, b, via), n in edges.items()],
                "tests": [[list(a), list(b)] for (a, b) in tests]}
        with open(f"{out}.{os.getpid()}.json", "w") as fh:
            json.dump(data, fh)

    atexit.register(dump)
    threading.setprofile(prof)
    sys.setprofile(prof)

    # Chain to any sitecustomize this one shadows.
    for entry in sys.path:
        if not entry or os.path.abspath(entry) == here:
            continue
        cand = os.path.join(entry, "sitecustomize.py")
        if os.path.isfile(cand):
            spec = importlib.util.spec_from_file_location("_promapkin_shadowed_sitecustomize", cand)
            if spec and spec.loader:
                spec.loader.exec_module(importlib.util.module_from_spec(spec))
            break

_promapkin_install()
'''


def run(ws: Workspace, command: list[str], kind: str = "auto", timeout: int = 3600) -> dict[str, Any]:
    """Run `command` with tracing attached, then ingest the trace. The command is the user's own."""
    if not command:
        raise ValueError("give the command to run, e.g. promapkin trace run -- python -m pytest -q")
    trace_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    out_dir = ws.dir / "traces" / trace_id
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="promapkin-trace-"))
    env = dict(os.environ)
    try:
        if kind in ("auto", "python"):
            (tmp / "sitecustomize.py").write_text(SITECUSTOMIZE)
            env["PYTHONPATH"] = os.pathsep.join(p for p in [str(tmp), env.get("PYTHONPATH", "")] if p)
            env["PROMAPKIN_TRACE_ROOT"] = str(ws.root)
            env["PROMAPKIN_TRACE_OUT"] = str(out_dir / "python")
        if kind in ("auto", "node"):
            env["NODE_V8_COVERAGE"] = str(out_dir / "v8")
        started = time.time()
        proc = subprocess.run(command, cwd=ws.root, env=env, timeout=timeout, check=False)
        seconds = round(time.time() - started, 2)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    result = ingest(ws, out_dir, trace_id)
    result.update({"command": command, "exit_code": proc.returncode, "seconds": seconds,
                   "trace_dir": str(out_dir.relative_to(ws.root))})
    return result


def ingest(ws: Workspace, trace_dir: Path, trace_id: str | None = None) -> dict[str, Any]:
    """Merge Python profile traces and V8 coverage from `trace_dir` into the graph as runtime evidence."""
    trace_dir = Path(trace_dir)
    if not trace_dir.is_absolute():
        trace_dir = ws.root / trace_dir
    trace_id = trace_id or trace_dir.name
    store = ws.store
    mapper = _Mapper(ws)
    edges: dict[tuple[str, str], dict[str, Any]] = {}
    verifies: set[tuple[str, str]] = set()
    executed: dict[str, int] = {}
    files = sorted(trace_dir.glob("python.*.json"))
    for f in files:
        data = json.loads(f.read_text())
        for a, b, via, n in data.get("edges", []):
            src, dst = mapper.node(a), mapper.node(b)
            if not src or not dst or src == dst:
                continue
            rec = edges.setdefault((src, dst), {"count": 0, "via_external": False})
            rec["count"] += n
            rec["via_external"] = rec["via_external"] or via
            executed[dst] = executed.get(dst, 0) + n
        for a, b in data.get("tests", []):
            t, d = mapper.node(a), mapper.node(b)
            if t and d and t != d:
                verifies.add((t, d))
    v8_files = sorted((trace_dir / "v8").glob("*.json")) if (trace_dir / "v8").is_dir() else []
    for f in v8_files:
        for nid, count in _v8_functions(ws, mapper, json.loads(f.read_text())):
            executed[nid] = executed.get(nid, 0) + count
    dynamic = []
    with store.transaction():
        for (src, dst), rec in edges.items():
            static = store.edges(src=src, dst=dst, where="type IN ('calls','instantiates','invokes','routes_to') "
                                                         "AND evidence != ?", params=(ev.RUNTIME,))
            store.add_edge(src, dst, "calls", evidence=ev.RUNTIME, confidence=1.0, revision=ws.revision(),
                           attrs={"count": rec["count"], "trace": trace_id, "via_external": rec["via_external"],
                                  "statically_known": bool(static)})
            if not static:
                dynamic.append({"source": src, "target": dst, "count": rec["count"],
                                "through_framework": rec["via_external"]})
        for t, d in verifies:
            store.add_edge(t, d, "verifies", evidence=ev.RUNTIME, confidence=1.0, revision=ws.revision(),
                           attrs={"trace": trace_id, "via": "executed during test"})
        for nid, count in executed.items():
            n = store.node(nid)
            if n:
                attrs = dict(n["attrs"])
                attrs["runtime"] = {"executions": count, "trace": trace_id}
                store.conn.execute("UPDATE nodes SET attrs=? WHERE id=?", (json.dumps(attrs, sort_keys=True), nid))
        store.record_artifact(f"trace:{trace_id}", ws.revision(), str(trace_dir),
                              {"edges": len(edges), "dynamic": len(dynamic), "verifies": len(verifies)})
    return {"trace": trace_id, "python_traces": len(files), "v8_coverage_files": len(v8_files),
            "runtime_edges": len(edges), "dynamic_links": dynamic[:100], "dynamic_link_count": len(dynamic),
            "test_links": len(verifies), "executed_nodes": len(executed), "unmapped_frames": mapper.unmapped}


class _Mapper:
    """Maps runtime frames (file, qualname, first line) to graph node ids."""

    def __init__(self, ws: Workspace):
        self.ws = ws
        self.root = os.path.realpath(ws.root)
        self.cache: dict[tuple, str | None] = {}
        self.unmapped = 0
        self.by_file: dict[str, list[dict[str, Any]]] = {}

    def rel(self, filename: str) -> str | None:
        real = os.path.realpath(filename)
        if not real.startswith(self.root + os.sep):
            return None
        return Path(real).relative_to(self.root).as_posix()

    def node(self, frame: list) -> str | None:
        key = tuple(frame)
        if key in self.cache:
            return self.cache[key]
        filename, qual, first = frame
        rel = self.rel(filename)
        nid = None
        if rel:
            if qual == "<module>":
                mod = next((e["dst"] for e in self.ws.store.edges(src=f"file:{rel}", types=["defines"])
                            if e["dst"].startswith("module:")), None)
                nid = mod
            else:
                clean = re.sub(r"\.?<locals>", "", qual)
                nid = self._match(rel, clean, int(first))
        if nid is None:
            self.unmapped += 1
        self.cache[key] = nid
        return nid

    def _match(self, rel: str, qual: str, line: int) -> str | None:
        if rel not in self.by_file:
            self.by_file[rel] = self.ws.store.nodes(
                where="file=? AND kind IN ('function','method','test','class')", params=(rel,))
        cands = self.by_file[rel]
        for n in cands:
            if n["qualname"] and n["qualname"].endswith("." + qual):
                return n["id"]
        name = qual.rsplit(".", 1)[-1]
        near = [n for n in cands if n["name"] == name and n["line_start"] is not None
                and line - 2 <= n["line_start"] <= line + 25]
        return near[0]["id"] if near else None

    def at_line(self, rel: str, line: int) -> str | None:
        if rel not in self.by_file:
            self.by_file[rel] = self.ws.store.nodes(
                where="file=? AND kind IN ('function','method','test','class')", params=(rel,))
        best = None
        for n in self.by_file[rel]:
            if n["line_start"] and n["line_end"] and n["line_start"] <= line <= n["line_end"]:
                if best is None or n["line_start"] >= best["line_start"]:
                    best = n
        return best["id"] if best else None


def _v8_functions(ws: Workspace, mapper: _Mapper, data: dict[str, Any]) -> list[tuple[str, int]]:
    out = []
    for script in data.get("result", []):
        url = script.get("url", "")
        if not url.startswith("file://"):
            continue
        path = url[len("file://"):]
        rel = mapper.rel(path)
        if not rel or "node_modules" in rel:
            continue
        try:
            text = Path(path).read_text(errors="replace")
        except OSError:
            continue
        for fn in script.get("functions", []):
            ranges = fn.get("ranges") or []
            if not ranges or not fn.get("functionName") or ranges[0].get("count", 0) <= 0:
                continue
            line = text.count("\n", 0, ranges[0]["startOffset"]) + 1
            nid = mapper.at_line(rel, line)
            if nid:
                out.append((nid, int(ranges[0]["count"])))
    return out
