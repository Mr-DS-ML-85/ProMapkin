"""LLM engine: the `--llm` mode of every Promapkin command.

In LLM mode the interpretive work that is otherwise done by regexes, rules and small
classifiers is done by an Anthropic or OpenAI model instead:

* code extraction for languages without a real parser (grounded against the source text);
* documentation analysis (relationship claims, decisions, requirements, constraints);
* every classification task (layers, design patterns, resource calls, doc/comment intent);
* query understanding and re-ranking for search and context building;
* specialist synthesis for blueprints, roadmap refinement, claim verification;
* explanations of any command's result, handoff drafting and briefing.

Guarantees: model output is schema-validated; extracted facts must be found in the source
(ungrounded facts are dropped); responses are cached by content hash so unchanged inputs are
never re-sent; secrets are redacted before anything leaves the machine; every request is
logged to .promapkin/egress.log; a per-invocation call budget bounds cost. Any failure falls
back to the deterministic implementation for that item.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterator

from promapkin import evidence as ev
from promapkin.extract import Facts, is_test_path
from promapkin.llm.clients import LLMClient, Usage, schema_errors
from promapkin.llm.errors import LLMBudgetExceeded, LLMError, LLMInvalidOutput, LLMUnavailable
from promapkin.security import UNTRUSTED_NOTICE

SYSTEM_BASE = (
    "You are the analysis engine of Promapkin, a codebase intelligence tool. " + UNTRUSTED_NOTICE +
    " Base every answer only on the material provided. Never invent identifiers, files or line numbers; "
    "if something is not supported by the material, leave it out."
)

_SECRET_PATTERNS = [
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S),
    re.compile(r"\b(AKIA|ASIA)[0-9A-Z]{16}\b"),
    re.compile(r"\b(sk|rk|pk)-(?:ant-|proj-|live-|test-)?[A-Za-z0-9_\-]{20,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b"),
    re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b"),
    re.compile(r"""(?i)((?:password|passwd|secret|api[_-]?key|token|auth)\s*[:=]\s*)(["'])[^"'\s]{6,}\2"""),
]


def redact(text: str) -> tuple[str, int]:
    n = 0
    for p in _SECRET_PATTERNS:
        def sub(m: re.Match[str]) -> str:
            nonlocal n
            n += 1
            if m.lastindex and m.lastindex >= 2 and m.group(2) in ("'", '"'):
                return f"{m.group(1)}{m.group(2)}[REDACTED]{m.group(2)}"
            return "[REDACTED]"
        text = p.sub(sub, text)
    return text, n


@dataclass
class EngineStats:
    cached: int = 0
    fallbacks: int = 0
    redactions: int = 0
    dropped_ungrounded: int = 0


class LLMEngine:
    def __init__(self, ws: Any, client: LLMClient, *, max_calls: int = 2000, concurrency: int = 4):
        self.ws = ws
        self.client = client
        self.usage: Usage = client.usage
        self.max_calls = max_calls
        self.concurrency = max(1, concurrency)
        self.stats = EngineStats()
        self._lock = threading.Lock()
        self._calls = 0
        self._cache_only = False

    # ------------------------------------------------------------------ plumbing
    @property
    def label(self) -> str:
        return f"{self.client.provider}/{self.client.model}"

    def describe(self) -> dict[str, Any]:
        return {"provider": self.client.provider, "model": self.client.model, "calls": self.usage.calls,
                "input_tokens": self.usage.input_tokens, "output_tokens": self.usage.output_tokens,
                "cached": self.stats.cached, "fallbacks": self.stats.fallbacks,
                "redactions": self.stats.redactions, "dropped_ungrounded": self.stats.dropped_ungrounded}

    def _key(self, task: str, *parts: str) -> str:
        h = hashlib.sha256(f"{self.client.provider}|{self.client.model}|{task}".encode())
        for p in parts:
            h.update(b"\0" + p.encode("utf-8", "replace"))
        return h.hexdigest()

    def _cache_get(self, key: str) -> Any:
        row = self.ws.store.conn.execute("SELECT response FROM llm_cache WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def _cache_put(self, key: str, task: str, value: Any) -> None:
        self.ws.store.conn.execute(
            "INSERT OR REPLACE INTO llm_cache(key, provider, model, task, response, created_at) VALUES (?,?,?,?,?,?)",
            (key, self.client.provider, self.client.model, task, json.dumps(value), time.time()))

    @contextmanager
    def cache_only(self) -> Iterator[None]:
        """Answer only from the response cache: used while a write transaction is open, so no network call
        ever holds the database lock. A miss raises LLMUnavailable, and callers fall back as usual."""
        self._cache_only = True
        try:
            yield
        finally:
            self._cache_only = False

    def _reserve(self) -> None:
        if self._cache_only:
            raise LLMUnavailable("not in the response cache (no network calls while the index is being written)")
        with self._lock:
            if self._calls >= self.max_calls:
                raise LLMBudgetExceeded(f"LLM call budget of {self.max_calls} reached for this command "
                                        f"(raise llm_max_calls in config)")
            self._calls += 1

    def _log(self, task: str, chars: int, subject: str) -> None:
        path = self.ws.dir / "egress.log"
        with self._lock:
            with path.open("a") as fh:
                fh.write(json.dumps({"time": time.time(), "provider": self.client.provider, "model": self.client.model,
                                     "task": task, "subject": subject[:200], "chars": chars}) + "\n")

    def _debug(self, task: str, subject: str, exc: Exception) -> Exception:
        """With PROMAPKIN_LLM_DEBUG=1, keep an invalid structured reply (secrets redacted) under
        .promapkin/llm-debug/ for diagnosis. Local only; never sent anywhere."""
        if not isinstance(exc, LLMInvalidOutput) or os.environ.get("PROMAPKIN_LLM_DEBUG", "") in ("", "0"):
            return exc
        d = self.ws.dir / "llm-debug"
        d.mkdir(exist_ok=True)
        path = d / f"{time.strftime('%Y%m%d-%H%M%S')}-{re.sub(r'[^a-z0-9]+', '-', task.lower())}-{id(exc):x}.json"
        path.write_text(json.dumps({"task": task, "subject": subject[:300], "provider": self.client.provider,
                                    "model": self.client.model, "kind": exc.kind, "finish_reason": exc.finish_reason,
                                    "error": str(exc), "raw_reply": redact(exc.raw)[0]}, indent=2))
        exc.args = (f"{exc.args[0]} [raw reply: {path.relative_to(self.ws.root)}]",)
        return exc

    def _valid_hit(self, hit: Any, schema: dict[str, Any]) -> bool:
        return isinstance(hit, dict) and not schema_errors(hit, schema)

    def _prepare(self, text: str) -> str:
        clean, n = redact(text)
        if n:
            with self._lock:
                self.stats.redactions += n
        return clean

    def call_json(self, task: str, system: str, user: str, schema: dict[str, Any], *, subject: str = "",
                  effort: str = "medium", max_tokens: int = 16000, use_cache: bool = True) -> dict[str, Any]:
        """One cached, budgeted, redacted structured call. Safe to call from worker threads only with
        use_cache=False (SQLite access stays on the calling thread)."""
        user = self._prepare(user)
        key = self._key(task, system, user, json.dumps(schema, sort_keys=True))
        if use_cache:
            hit = self._cache_get(key)
            if hit is not None and self._valid_hit(hit, schema):
                self.stats.cached += 1
                self.usage.cached_calls += 1
                return hit
        self._reserve()
        self._log(task, len(user), subject)
        try:
            out = self.client.json(SYSTEM_BASE + "\n\n" + system, user, schema, max_tokens=max_tokens,
                                   effort=effort, name=task)
        except LLMInvalidOutput as exc:
            raise self._debug(task, subject, exc) from exc.__cause__
        if use_cache:
            self._cache_put(key, task, out)
        return out

    def call_text(self, task: str, system: str, user: str, *, subject: str = "", effort: str = "medium",
                  max_tokens: int = 8000) -> str:
        user = self._prepare(user)
        key = self._key(task, system, user)
        hit = self._cache_get(key)
        if hit is not None:
            self.stats.cached += 1
            self.usage.cached_calls += 1
            return hit
        self._reserve()
        self._log(task, len(user), subject)
        out = self.client.text(SYSTEM_BASE + "\n\n" + system, user, max_tokens=max_tokens, effort=effort)
        self._cache_put(key, task, out)
        return out

    def map_json(self, task: str, jobs: list[tuple[str, str, str]], schema: dict[str, Any], *,
                 effort: str = "medium", max_tokens: int = 16000) -> list[dict[str, Any] | Exception]:
        """Run many (system, user, subject) jobs concurrently. Cache I/O stays on this thread."""
        results: list[Any] = [None] * len(jobs)
        pending: list[tuple[int, str, str, str, str]] = []
        for i, (system, user, subject) in enumerate(jobs):
            clean = self._prepare(user)
            key = self._key(task, system, clean, json.dumps(schema, sort_keys=True))
            hit = self._cache_get(key)
            if hit is not None and self._valid_hit(hit, schema):
                self.stats.cached += 1
                self.usage.cached_calls += 1
                results[i] = hit
            else:
                pending.append((i, key, system, clean, subject))

        def run(item: tuple[int, str, str, str, str]) -> tuple[int, str, Any]:
            i, key, system, user, subject = item
            try:
                self._reserve()
                self._log(task, len(user), subject)
                return i, key, self.client.json(SYSTEM_BASE + "\n\n" + system, user, schema, max_tokens=max_tokens,
                                                effort=effort, name=task)
            except Exception as exc:  # noqa: BLE001 - reported per item, caller falls back
                return i, key, exc

        if pending:
            with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
                for (i, key, out), job in zip(pool.map(run, pending), pending):
                    results[i] = self._debug(task, job[4], out) if isinstance(out, Exception) else out
                    if not isinstance(out, Exception):
                        self._cache_put(key, task, out)
        return results

    # ------------------------------------------------------------------ classification
    def classify_many(self, task_name: str, texts: list[str], batch: int = 40) -> list[dict[str, Any] | None]:
        """Classify texts with the LLM. Returns per-item {label, confidence, reason} or None (fallback)."""
        from promapkin.ml import TASKS
        task = TASKS[task_name]
        labels = sorted(task.labels) + ([task.abstain] if task.abstain not in task.labels else [])
        schema = {"type": "object", "properties": {"items": {"type": "array", "items": {
            "type": "object", "properties": {"index": {"type": "integer"}, "label": {"type": "string", "enum": labels},
                                             "confidence": {"type": "number"}, "reason": {"type": "string"}}}}}}
        system = (f"Classify each item for the task '{task_name}'. Labels and their meaning:\n" +
                  "\n".join(f"- {k}: {v}" for k, v in sorted(task.labels.items())) +
                  f"\nUse '{task.abstain}' when none applies. Confidence is your calibrated probability (0-1). "
                  "Keep reasons under 20 words.")
        out: list[dict[str, Any] | None] = [None] * len(texts)
        jobs, spans = [], []
        for start in range(0, len(texts), batch):
            chunk = texts[start:start + batch]
            jobs.append((system, json.dumps([{"index": i, "item": t[:1500]} for i, t in enumerate(chunk)]),
                         f"{task_name} batch@{start}"))
            spans.append(start)
        for start, res in zip(spans, self.map_json(f"classify:{task_name}", jobs, schema, effort="low",
                                                  max_tokens=8000)):
            if isinstance(res, Exception):
                if not self._cache_only:   # a cache-only miss already counted when the live call failed
                    self.stats.fallbacks += 1
                continue
            for item in res.get("items", []):
                idx = item.get("index")
                if isinstance(idx, int) and 0 <= start + idx < len(texts) and start + idx < start + batch:
                    out[start + idx] = {"label": item["label"], "confidence": max(0.0, min(1.0, float(
                        item.get("confidence", 0.5)))), "reason": item.get("reason", "")}
        return out

    # ------------------------------------------------------------------ code extraction
    CODE_SCHEMA = {"type": "object", "properties": {
        "module_doc": {"type": "string"},
        "symbols": {"type": "array", "items": {"type": "object", "properties": {
            "name": {"type": "string"}, "kind": {"type": "string", "enum": ["class", "function", "method", "interface"]},
            "parent": {"type": "string"}, "line_start": {"type": "integer"}, "line_end": {"type": "integer"},
            "doc": {"type": "string"}, "params": {"type": "array", "items": {"type": "string"}},
            "bases": {"type": "array", "items": {"type": "string"}}, "exported": {"type": "boolean"}}}},
        "imports": {"type": "array", "items": {"type": "object", "properties": {
            "target": {"type": "string"}, "line": {"type": "integer"}}}},
        "calls": {"type": "array", "items": {"type": "object", "properties": {
            "caller": {"type": "string"}, "callee": {"type": "string"}, "line": {"type": "integer"}}}},
        "routes": {"type": "array", "items": {"type": "object", "properties": {
            "method": {"type": "string"}, "path": {"type": "string"}, "handler": {"type": "string"},
            "line": {"type": "integer"}}}},
        "env": {"type": "array", "items": {"type": "object", "properties": {
            "key": {"type": "string"}, "line": {"type": "integer"}}}},
        "resources": {"type": "array", "items": {"type": "object", "properties": {
            "caller": {"type": "string"}, "resource": {"type": "string", "enum": [
                "database", "http", "queue", "cache", "filesystem", "subprocess"]},
            "operation": {"type": "string", "enum": ["reads", "writes", "invokes"]}, "line": {"type": "integer"}}}},
    }}

    def extract_code(self, path: str, language: str, text: str, chunk_lines: int = 700) -> Facts:
        """LLM structural extraction for languages without an installed parser. Every symbol and call
        must be found at (or next to) the line the model reports, otherwise it is dropped."""
        lines = text.splitlines()
        system = (f"Extract the structure of this {language} source file: declarations (with parent class/type "
                  "for members), imports, call sites (caller = enclosing symbol name, callee = the called "
                  "expression as written, e.g. repo.save or Command::new), HTTP routes, environment variable reads "
                  "and calls that touch databases, HTTP, queues, caches, files or subprocesses. Lines are numbered; "
                  "report the exact line numbers shown.")
        jobs = []
        for start in range(0, max(len(lines), 1), chunk_lines):
            chunk = lines[start:start + chunk_lines + 20]
            numbered = "\n".join(f"{start + i + 1:>6}| {l}" for i, l in enumerate(chunk))
            jobs.append((system, f"File: {path}\n\n{numbered}", f"{path}@{start + 1}"))
        results = self.map_json("extract_code", jobs, self.CODE_SCHEMA, effort="medium")
        merged: dict[str, list] = {k: [] for k in ("symbols", "imports", "calls", "routes", "env", "resources")}
        doc = ""
        for r in results:
            if isinstance(r, Exception):
                raise r
            doc = doc or r.get("module_doc", "")
            for k in merged:
                merged[k] += r.get(k, [])
        return self._facts_from_llm(path, language, lines, merged, doc)

    def _grounded(self, lines: list[str], line: int, token: str, slack: int = 2) -> bool:
        if not token or not isinstance(line, int):
            return False
        token = re.split(r"[.:(<\s]", token.strip().split("::")[-1].split(".")[-1])[0] or token
        for ln in range(max(1, line - slack), min(len(lines), line + slack) + 1):
            if token in lines[ln - 1]:
                return True
        return False

    def _facts_from_llm(self, path: str, language: str, lines: list[str], data: dict[str, list], doc: str
                        ) -> Facts:
        from promapkin.extract.treesitter_ext import module_name
        f = Facts(path, language)
        mod = module_name(path)
        mod_id = f"module:{mod}"
        is_test = is_test_path(path)
        E = ev.LLM_EXTRACTION
        f.node(mod_id, "module", mod, E, qualname=mod, file=path, line_start=1, line_end=max(len(lines), 1),
               text=doc or None, attrs={"test": is_test, "loc": len(lines), "parser": f"llm:{self.label}",
                                        "imports": {}})
        f.edge(f"file:{path}", mod_id, "defines", E)
        known: dict[str, str] = {}
        dropped = 0
        for s in sorted(data["symbols"], key=lambda s: s.get("line_start", 0)):
            name, line = s.get("name", ""), s.get("line_start", 0)
            if not re.fullmatch(r"[A-Za-z_$~][\w$]*", name or "") or not self._grounded(lines, line, name):
                dropped += 1
                continue
            parent = s.get("parent") or ""
            qual = f"{parent}.{name}" if parent else name
            sid = f"symbol:{mod}:{qual}"
            kind = "class" if s["kind"] in ("class", "interface") else ("method" if parent else "function")
            if is_test and kind != "class" and name.lower().startswith("test"):
                kind = "test"
            f.node(sid, kind, name, E, qualname=f"{mod}.{qual}", file=path, line_start=line,
                   line_end=max(line, s.get("line_end") or line), text=s.get("doc") or None,
                   attrs={"symbol_type": kind, "params": [{"name": p, "type": None} for p in s.get("params", [])],
                          "abstract": s["kind"] == "interface", "class": parent or None,
                          "public": s.get("exported", True), "extractor": "llm"})
            psid = f"symbol:{mod}:{parent}"
            f.edge(psid if parent and parent in known else mod_id, sid, "contains" if parent in known else "defines",
                   E, line=line)
            known[qual if not parent else name] = sid
            known.setdefault(name, sid)
            for b in s.get("bases", []):
                f.ref(sid, b.replace("::", "."), "inherits", line, module=mod)
        spans = sorted(((n["line_start"], n["line_end"], n["id"]) for n in f.nodes
                        if n["kind"] in ("function", "method", "test")), key=lambda t: t[0])

        def owner(line: int, hint: str) -> str:
            if hint and hint in known:
                return known[hint]
            best = None
            for a, b, sid in spans:
                if a <= line <= b:
                    best = sid
            return best or mod_id

        for imp in data["imports"]:
            if self._grounded(lines, imp.get("line", 0), imp.get("target", "").split("/")[-1].split(".")[-1]):
                f.ref(mod_id, imp["target"].replace("::", "."), "imports", imp["line"], module=mod)
            else:
                dropped += 1
        for c in data["calls"]:
            callee = c.get("callee", "").replace("::", ".").replace("->", ".")
            if not self._grounded(lines, c.get("line", 0), callee, slack=1):
                dropped += 1
                continue
            f.ref(owner(c["line"], c.get("caller", "")), re.sub(r"^this\.", "self.", callee), "calls", c["line"],
                  module=mod)
        for r in data["routes"]:
            if not self._grounded(lines, r.get("line", 0), r.get("path", "").strip("/").split("/")[0] or "/", 2):
                dropped += 1
                continue
            api_id = f"api:{r['method'].upper()} {r['path']}"
            f.node(api_id, "endpoint", f"{r['method'].upper()} {r['path']}", E, file=path, line_start=r["line"],
                   attrs={"method": r["method"].upper(), "route": r["path"], "auth": []})
            f.edge(mod_id, api_id, "exposes", E, line=r["line"])
            if r.get("handler"):
                f.ref(api_id, r["handler"].replace("::", "."), "routes_to", r["line"], module=mod)
        for e in data["env"]:
            if self._grounded(lines, e.get("line", 0), e.get("key", "")):
                f.node(f"config:{e['key']}", "config", e["key"], E, attrs={"source": "environment"})
                f.edge(owner(e["line"], ""), f"config:{e['key']}", "reads", E, line=e["line"])
        for r in data["resources"]:
            f.node(f"resource:{r['resource']}", "resource", r["resource"], ev.HEURISTIC,
                   attrs={"category": r["resource"]})
            f.edge(owner(r.get("line", 0), r.get("caller", "")), f"resource:{r['resource']}", r["operation"],
                   ev.LLM_EXTRACTION, line=r.get("line"), confidence=0.7, attrs={"classifier": f"llm:{self.label}"})
        self.stats.dropped_ungrounded += dropped
        return f

    # ------------------------------------------------------------------ documentation
    DOC_SCHEMA = {"type": "object", "properties": {
        "claims": {"type": "array", "items": {"type": "object", "properties": {
            "subject": {"type": "string"}, "relation": {"type": "string", "enum": [
                "depends_on", "calls", "reads", "writes", "invokes", "imports", "produces", "consumes"]},
            "object": {"type": "string"}, "quote": {"type": "string"}}}},
        "decisions": {"type": "array", "items": {"type": "object", "properties": {
            "title": {"type": "string"}, "decision": {"type": "string"}, "rationale": {"type": "string"},
            "status": {"type": "string"}, "quote": {"type": "string"}}}},
        "requirements": {"type": "array", "items": {"type": "object", "properties": {
            "text": {"type": "string"}, "kind": {"type": "string", "enum": ["requirement", "constraint"]},
            "quote": {"type": "string"}}}},
        "code_mentions": {"type": "array", "items": {"type": "string"}},
    }}

    def enrich_document(self, path: str, text: str, facts: Facts) -> None:
        """Replace regex claim mining with LLM analysis. Every item must quote the document verbatim."""
        if len(text.strip()) < 40:
            return
        system = ("From this project document extract: (1) explicit claims that one code element depends on, "
                  "calls, reads, writes, invokes, imports, produces or consumes another (use the identifiers as "
                  "written); (2) architecture decisions with their rationale and status; (3) requirements and "
                  "constraints the code must respect; (4) names of code elements mentioned. Every claim, decision "
                  "and requirement must include a short verbatim quote from the document.")
        snapshot = (list(facts.nodes), list(facts.edges), list(facts.refs))
        try:
            self._enrich(path, text, facts, system)
        except Exception:
            # All or nothing: a failed analysis leaves the deterministic Markdown facts exactly as extracted.
            facts.nodes, facts.edges, facts.refs = snapshot
            with self._lock:
                self.stats.fallbacks += 1
            raise

    def _enrich(self, path: str, text: str, facts: Facts, system: str) -> None:
        data = self.call_json("enrich_document", system, f"Document: {path}\n\n{text[:60000]}", self.DOC_SCHEMA,
                              subject=path, effort="low")
        norm = " ".join(text.split())

        def where(quote: str) -> int | None:
            q = " ".join((quote or "").split())
            if len(q) < 8 or q not in norm:
                return None
            idx = text.find(quote.strip()[:30])
            return text.count("\n", 0, idx) + 1 if idx >= 0 else 1

        facts.refs = [r for r in facts.refs if r["type"] != "claims"]   # LLM replaces the claim regex
        doc_id = f"doc:{path}"
        dropped = 0
        for c in data.get("claims", []):
            line = where(c.get("quote", ""))
            if line is None:
                dropped += 1
                continue
            facts.ref(doc_id, c["object"], "claims", line, subject=c["subject"],
                      verb={"depends_on": "depends on", "reads": "reads from", "writes": "writes to"}.get(
                          c["relation"], c["relation"]), sentence=c["quote"][:200], extractor="llm")
        for d in data.get("decisions", []):
            line = where(d.get("quote", ""))
            if line is None:
                dropped += 1
                continue
            did = f"decision:{path}#{re.sub(r'[^a-z0-9]+', '-', d['title'].lower())[:48]}"
            facts.node(did, "decision", d["title"][:160], ev.DOCUMENTATION, file=path, line_start=line,
                       text=f"{d['decision']} Rationale: {d['rationale']}"[:2000],
                       attrs={"status": d.get("status") or "unknown", "decision": d["decision"][:1200],
                              "context": "", "consequences": "", "rationale": d["rationale"][:1200],
                              "origin": f"llm-extracted ({self.label})"})
            facts.edge(doc_id, did, "records", ev.DOCUMENTATION)
        for rq in data.get("requirements", []):
            line = where(rq.get("quote", ""))
            if line is None:
                dropped += 1
                continue
            rid = f"{rq['kind']}:{path}:{line}"
            facts.node(rid, rq["kind"], rq["text"][:160], ev.DOCUMENTATION, file=path, line_start=line,
                       text=rq["text"][:1000], attrs={"origin": f"llm-extracted ({self.label})"})
            facts.edge(doc_id, rid, "records", ev.DOCUMENTATION)
        for name in data.get("code_mentions", [])[:200]:
            if name and name in text:
                facts.ref(doc_id, name.replace("::", "."), "mentions", 1)
        self.stats.dropped_ungrounded += dropped

    # ------------------------------------------------------------------ retrieval
    def expand_query(self, query: str) -> dict[str, list[str]]:
        schema = {"type": "object", "properties": {"identifiers": {"type": "array", "items": {"type": "string"}},
                                                   "keywords": {"type": "array", "items": {"type": "string"}}}}
        return self.call_json("expand_query", "Turn this developer question into likely code identifiers "
                              "(class, function, module names in common naming styles) and search keywords.",
                              query, schema, subject=query, effort="low", max_tokens=2000)

    def rerank(self, query: str, candidates: list[dict[str, Any]]) -> list[tuple[str, float]]:
        schema = {"type": "object", "properties": {"ranking": {"type": "array", "items": {"type": "object",
                  "properties": {"id": {"type": "string"}, "relevance": {"type": "number"}}}}}}
        items = [{"id": c["id"], "kind": c.get("kind"), "summary": c.get("summary", "")[:300]} for c in candidates]
        data = self.call_json("rerank", "Rank these code elements by relevance to the question (relevance 0-1). "
                              "Only return ids from the list.", json.dumps({"question": query, "candidates": items}),
                              schema, subject=query, effort="low", max_tokens=4000)
        valid = {c["id"] for c in candidates}
        return [(r["id"], float(r["relevance"])) for r in data.get("ranking", []) if r.get("id") in valid]

    # ------------------------------------------------------------------ explanations & synthesis
    def explain(self, command: str, result: Any, question: str | None = None) -> str:
        payload = json.dumps(result, default=str)[:60000]
        return self.call_text("explain", (
            "Explain this Promapkin result to a developer in plain language: what it shows, what matters most, "
            "and what to do next. Distinguish observed facts from inferences using the evidence fields. "
            "Cite identifiers and file:line locations from the result. Be concise."),
            f"Command: {command}\nQuestion: {question or 'Explain the result.'}\n\nResult JSON:\n{payload}",
            subject=command, effort="medium")

    def synthesize(self, contract: dict[str, Any]) -> dict[str, Any]:
        from promapkin.agents.contract import RESULT_SCHEMA
        user = json.dumps({k: contract[k] for k in ("specialist", "objective", "scope", "revision", "facts")},
                          default=str)
        return self.call_json(f"specialist:{contract['specialist']}", contract["evidence_requirements"] +
                              " Return a summary, grounded claims with citations (node ids or path:line from the "
                              "facts), relationships between existing node ids, and open questions.", user,
                              RESULT_SCHEMA, subject=contract.get("id", ""), effort="high")

    def draft_handoff(self, notes: str, context: dict[str, Any]) -> dict[str, Any]:
        schema = {"type": "object", "properties": {
            "task": {"type": "string"}, "status": {"type": "string", "enum": [
                "open", "investigating", "investigation_complete", "planned", "in_progress", "blocked", "review"]},
            "summary": {"type": "string"}, "next_action": {"type": "string"},
            "findings": {"type": "array", "items": {"type": "object", "properties": {
                "claim": {"type": "string"}, "sources": {"type": "array", "items": {"type": "string"}},
                "status": {"type": "string", "enum": ["verified", "unverified"]}}}},
            "constraints": {"type": "array", "items": {"type": "string"}},
            "validation": {"type": "array", "items": {"type": "string"}},
            "open_questions": {"type": "array", "items": {"type": "string"}},
            "symbols": {"type": "array", "items": {"type": "string"}}}}
        data = self.call_json("draft_handoff", "Turn these working notes into a structured handoff for the next "
                              "agent. Use only sources (path:line or node ids) and symbols that appear in the notes "
                              "or context.", json.dumps({"notes": notes, "context": context}, default=str)[:60000],
                              schema, subject="handoff", effort="medium")
        data["relevant_context"] = {"symbols": data.pop("symbols", []), "dependencies": [], "files": []}
        return data

    def judge_claims(self, checks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not checks:
            return []
        schema = {"type": "object", "properties": {"verdicts": {"type": "array", "items": {"type": "object",
                  "properties": {"index": {"type": "integer"}, "verdict": {"type": "string", "enum": [
                      "supported", "contradicted", "insufficient_evidence"]}, "explanation": {"type": "string"}}}}}}
        data = self.call_json("judge_claims", "For each documentation claim, judge whether the static evidence "
                              "supports it, contradicts it, or is insufficient. Explain briefly.",
                              json.dumps([{"index": i, **c} for i, c in enumerate(checks)], default=str)[:60000],
                              schema, subject="verify", effort="medium")
        return data.get("verdicts", [])

    def refine_roadmap(self, roadmap: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        schema = {"type": "object", "properties": {
            "summary": {"type": "string"},
            "steps": {"type": "array", "items": {"type": "object", "properties": {
                "title": {"type": "string"}, "detail": {"type": "string"},
                "components": {"type": "array", "items": {"type": "string"}},
                "prerequisites": {"type": "array", "items": {"type": "integer"}},
                "validation": {"type": "array", "items": {"type": "string"}},
                "risk": {"type": "string", "enum": ["low", "medium", "high"]}}}},
            "risks": {"type": "array", "items": {"type": "string"}}}}
        return self.call_json("refine_roadmap", "Improve this dependency-ordered implementation plan for the goal. "
                              "Keep steps concrete; components must be node ids from the input; prerequisites "
                              "refer to earlier step numbers.", json.dumps({"roadmap": roadmap, "context": context},
                                                                          default=str)[:60000],
                              schema, subject=roadmap.get("goal") or "roadmap", effort="high")


def fallback(fn: Callable[[], Any], default: Callable[[], Any], engine: LLMEngine | None) -> Any:
    """Run an LLM step; on any LLMError use the deterministic implementation instead."""
    if engine is None:
        return default()
    try:
        return fn()
    except LLMError:
        engine.stats.fallbacks += 1
        return default()
