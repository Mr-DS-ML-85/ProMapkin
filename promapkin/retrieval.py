"""Hybrid retrieval and the budget-aware context compiler.

Pipeline: task query → hybrid retrieval (symbol lookup + full-text BM25 + semantic embeddings)
→ evidence-aware graph expansion (callers, callees, tests, decisions, docs) → context
compiler (deduplication, token budgeting, evidence prioritisation) → compact package.

Critical items (constraints, validation requirements) are never dropped to fit a budget; if
they alone exceed it, the package says so and reports the minimum budget needed.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any

from promapkin import evidence as ev
from promapkin.config import Workspace
from promapkin.graph import Graph
from promapkin.security import UNTRUSTED_NOTICE, safe_path

EXPANSION = {   # (edge type, direction from the seed) -> (relevance, relation of the candidate to the seed)
    ("calls", "out"): (0.7, "called by"),
    ("calls", "in"): (0.65, "calls"),
    ("instantiates", "in"): (0.55, "instantiates"),
    ("instantiates", "out"): (0.5, "instantiated by"),
    ("verifies", "in"): (0.75, "tests"),
    ("documents", "in"): (0.6, "documents"),
    ("depends_on", "out"): (0.6, "dependency of"),
    ("depends_on", "in"): (0.55, "depends on"),
    ("routes_to", "in"): (0.7, "routes to"),
    ("inherits", "out"): (0.6, "base class of"),
    ("inherits", "in"): (0.5, "subclass of"),
    ("overrides", "out"): (0.5, "overridden by"),
    ("overrides", "in"): (0.55, "overrides"),
    ("reads", "out"): (0.55, "read by"),
    ("writes", "out"): (0.65, "written by"),
    ("implements_pattern", "out"): (0.5, "pattern used by"),
    ("about", "in"): (0.7, "finding about"),
    ("affects", "in"): (0.6, "affects"),
    ("constrains", "in"): (0.9, "constrains"),
    ("similar_to", "out"): (0.35, "similar to"),
    ("similar_to", "in"): (0.35, "similar to"),
    ("contains", "out"): (0.4, "member of"),
    ("contains", "in"): (0.45, "contains"),
}
SEARCH_KINDS = ("module", "class", "function", "method", "endpoint", "doc_section", "decision", "finding", "test",
                "pattern", "concept", "code_note", "config", "resource", "roadmap_step", "constraint", "requirement",
                "handoff", "risk", "bug", "task")


def estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


NOISY_DOCS = re.compile(r"(?i)(changelog|changes|history|release[-_ ]?notes|contributing|code[-_]of[-_]conduct|"
                        r"license|authors|security)[^/]*\.(md|rst|txt|markdown)")
RETRIEVAL_VERSION = 4   # part of the context-cache key: packages built by older ranking code are not reused
EXAMPLE_DIRS = {"examples", "example", "docs_src", "samples", "sample", "demo", "demos", "tutorial", "tutorials",
                "playground", "sandbox", "benchmarks", "bench", "fixtures"}
VENDORED_DIRS = {"vendor", "vendored", "third_party", "third-party", "node_modules", "external", "deps"}


def path_role(path: str) -> str:
    """'test', 'example', 'vendored' or 'source', from directory names."""
    parts = path.lower().split("/")[:-1]
    if any(p in ("tests", "test", "__tests__", "spec", "testing", "e2e") for p in parts):
        return "test"
    if any(p in EXAMPLE_DIRS for p in parts):
        return "example"
    if any(p in VENDORED_DIRS for p in parts):
        return "vendored"
    return "source"


class Retriever:
    def __init__(self, ws: Workspace, rt: Any = None):
        self.ws = ws
        self.store = ws.store
        self.graph = Graph(ws.store)
        if rt is None:
            from promapkin import ml
            rt = ml.runtime(ws)
        self.rt = rt

    @property
    def ml(self) -> bool:
        return self.rt.classifier.mode != "heuristic"

    # ---------------------------------------------------------------- hybrid search
    def search(self, query: str, limit: int = 20, kinds: list[str] | None = None) -> list[dict[str, Any]]:
        kinds = kinds or list(SEARCH_KINDS)
        signals: dict[str, dict[str, float]] = {}

        def add(signal: str, ranked: list[tuple[str, float]]) -> None:
            for rank, (nid, score) in enumerate(ranked):
                signals.setdefault(nid, {})[signal] = max(signals.get(nid, {}).get(signal, 0), 1.0 / (rank + 1))

        # 1. symbol lookup: identifiers mentioned in the query
        idents = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*(?:(?:\.|::)[A-Za-z_][A-Za-z0-9_]*)*", query))
        exact = []
        for ident in idents:
            if len(ident) < 3 or not (re.search(r"[A-Z_.]", ident) or "::" in ident):
                continue
            name = ident.replace("::", ".")
            hits = self.store.find_nodes_by_name(name)
            # Case-sensitive suffix match on qualified names (GLOB, unlike LIKE, respects case).
            hits += [dict(r) | {"attrs": {}, "suffix": True} for r in self.store.conn.execute(
                "SELECT id, kind, file FROM nodes WHERE qualname GLOB ? LIMIT 8", (f"*.{name}",))]
            for n in hits:
                # Definitions outrank tests and modules that merely share the name.
                rank = 1.0 if n["kind"] in ("class", "function", "method", "endpoint") else 0.8
                if n.get("suffix"):
                    rank *= 0.9
                if (n.get("attrs") or {}).get("test") or n["kind"] == "test" or "/tests/" in f"/{n['file'] or ''}":
                    rank *= 0.6
                exact.append((n["id"], rank))
        best: dict[str, float] = {}
        for nid, rank in exact:
            best[nid] = max(best.get(nid, 0.0), rank)
        add("symbol", sorted(best.items(), key=lambda t: -t[1]))
        # 2. lexical BM25
        add("lexical", self.store.search(query, limit=60, kinds=kinds))
        # 2b. documentation bridge: prose often answers a question in the words it is asked in, and doc sections
        # link to the code they describe. Their match is passed on to that code.
        code_kinds = [k for k in kinds if k not in ("document", "doc_section", "decision")]
        if code_kinds:
            bridged: dict[str, float] = {}
            doc_hits = [(d, sc) for d, sc in self.store.search(query, limit=25, kinds=["doc_section", "document"])
                        if not NOISY_DOCS.search(d)][:15]
            for rank, (did, _score) in enumerate(doc_hits):
                targets = [r[0] for r in self.store.conn.execute(
                    "SELECT dst FROM edges WHERE src = ? AND type = 'documents'", (did,))]
                targets = [t for t in targets if t.startswith(("symbol:", "module:", "api:"))][:12]
                for t in targets:
                    bridged[t] = max(bridged.get(t, 0.0), 1.0 / (rank + 1) / (1 + 0.15 * len(targets)))
            add("doc_bridge", sorted(bridged.items(), key=lambda t: -t[1]))
        # 3. semantic (ML tier only)
        if self.ml:
            from promapkin.ml.embedders import store_top_k
            emb = self.rt.embedder
            qv = emb.embed([query])[0]
            add("semantic", [(i, s) for i, s in store_top_k(self.store, emb.name, qv, 60) if s > 0.15])
            from promapkin.ml.graph_ml import graph_model_name
            add("graph_semantic", [(i, s) for i, s in store_top_k(self.store, graph_model_name(emb.name), qv, 40)
                                   if s > 0.2])
        # Neural embeddings carry real semantics; the built-in hashing embedder is mostly lexical, so it only
        # nudges the ranking (measured on examples/bench: full weight hurt MRR).
        neural = self.ml and self.rt.embedder.neural
        weights = {"symbol": 1.0, "lexical": 0.8, "doc_bridge": 0.45, "semantic": 0.7 if neural else 0.25,
                   "graph_semantic": 0.4 if neural else 0.1}
        wants_tests = bool(re.search(r"\btest", query, re.I))
        wants_examples = bool(re.search(r"\b(example|tutorial|sample|demo)", query, re.I))
        fused = []
        for nid, sig in signals.items():
            n = self.store.node(nid)
            if not n or n["kind"] not in kinds or n["status"] == "superseded":
                continue
            score = sum(weights[k] * v for k, v in sig.items())
            score *= 0.6 + 0.4 * ev.weight(n["evidence"], n["status"])
            role = path_role(n["file"] or "")
            if (n["attrs"].get("test") or n["kind"] == "test" or role == "test") and not wants_tests:
                score *= 0.55
            elif role == "example" and not wants_examples:
                score *= 0.6
            elif role == "vendored":
                score *= 0.5
            if n["kind"] in ("module", "document"):
                score *= 0.75   # prefer concrete symbols over whole modules
                if (n["attrs"].get("loc") or 99) < 5:
                    score *= 0.6   # near-empty package markers
            fused.append({"id": nid, "kind": n["kind"], "name": n["name"], "score": round(score, 4),
                          "location": f"{n['file']}:{n['line_start']}" if n["file"] and n["line_start"] else n["file"],
                          "signals": {k: round(v, 3) for k, v in sig.items()}, "evidence": n["evidence"],
                          "status": n["status"]})
        fused.sort(key=lambda h: -h["score"])
        llm = getattr(self.ws, "llm", None)
        if llm is not None and fused and not getattr(self, "_in_llm", False):
            fused = self._llm_rerank(query, fused, limit, kinds, llm)
        top = fused[0]["score"] if fused else 1
        for h in fused:
            h["score"] = round(h["score"] / top, 4)
        return fused[:limit]

    def _llm_rerank(self, query: str, fused: list[dict[str, Any]], limit: int, kinds: list[str], llm: Any
                    ) -> list[dict[str, Any]]:
        """--llm mode: the model expands the question into identifiers, then re-ranks the candidates."""
        from promapkin.llm.errors import LLMError
        self._in_llm = True
        try:
            exp = llm.expand_query(query)
            extra = " ".join(exp.get("identifiers", []) + exp.get("keywords", []))
            if extra:
                more = self.search(f"{query} {extra}", limit=40, kinds=kinds)
                have = {h["id"] for h in fused}
                fused += [h for h in more if h["id"] not in have]
            pool = fused[:max(30, limit)]
            cands = [{"id": h["id"], "kind": h["kind"],
                      "summary": f"{h['name']} {h.get('location') or ''} {(self.store.node(h['id']) or {}).get('text') or ''}"}
                     for h in pool]
            ranking = dict(llm.rerank(query, cands))
            for h in pool:
                if h["id"] in ranking:
                    h["signals"]["llm"] = round(ranking[h["id"]], 3)
                    h["score"] = 0.35 * h["score"] / (pool[0]["score"] or 1) + 0.65 * ranking[h["id"]]
            pool.sort(key=lambda h: -h["score"])
            return pool + fused[len(pool):]
        except LLMError:
            llm.stats.fallbacks += 1
            return fused
        finally:
            self._in_llm = False

    # ---------------------------------------------------------------- context compiler
    def build_context(self, task: str, budget: int | None = None, constraints: list[str] | None = None,
                      validation: list[str] | None = None, include_source: bool = True,
                      focus: list[str] | None = None, handoff: dict[str, Any] | None = None,
                      use_cache: bool = True) -> dict[str, Any]:
        budget = budget or self.ws.config.default_token_budget
        key = hashlib.sha1(json.dumps([RETRIEVAL_VERSION, task, budget, constraints, validation, include_source, focus,
                                       handoff.get("id") if handoff else None,
                                       handoff.get("version") if handoff else None, self.rt.classifier.mode,
                                       self.store.get_meta("indexed_at")], sort_keys=True).encode()).hexdigest()
        cached = self.store.get_meta(f"ctx:{key}") if use_cache else None
        if cached:
            pkg = json.loads(cached)
            pkg["cache"] = "hit"
            return pkg

        seeds = self.search(task, limit=15)
        for f in focus or []:
            try:
                nid = self.graph.resolve(f)
                seeds.insert(0, {"id": nid, "score": 1.0, "kind": (self.store.node(nid) or {}).get("kind"),
                                 "name": nid, "signals": {"focus": 1.0}})
            except LookupError:
                pass
        if handoff:
            for sym in handoff.get("relevant_context", {}).get("symbols", []):
                try:
                    nid = self.graph.resolve(sym)
                    seeds.insert(0, {"id": nid, "score": 1.0, "kind": (self.store.node(nid) or {}).get("kind"),
                                     "name": nid, "signals": {"handoff": 1.0}})
                except LookupError:
                    pass

        candidates: dict[str, dict[str, Any]] = {}
        relationships: list[dict[str, Any]] = []
        for sd in seeds[:10]:
            nid = sd["id"]
            candidates.setdefault(nid, {"id": nid, "score": sd["score"], "reasons": ["matches task"]})
            for (etype, direction), (rel, label) in EXPANSION.items():
                for other, e, trust in self.graph._adjacent(nid, direction, [etype]):
                    score = sd["score"] * rel * trust
                    if score < 0.12:
                        continue
                    c = candidates.setdefault(other, {"id": other, "score": 0.0, "reasons": []})
                    c["score"] = max(c["score"], score)
                    name = nid.rsplit(":", 1)[-1]
                    c["reasons"].append(f"{label} {name}")
                    if score >= 0.2:
                        relationships.append(self.graph.describe_edge(e))

        # Mandatory items.
        from promapkin.agents.specialists import Ctx, _constraints_for
        ctx = Ctx(self.ws, self.store, self.graph, self.rt, self.ws.revision())
        mandatory: list[dict[str, Any]] = []
        for c in _constraints_for(ctx, [s["id"] for s in seeds[:10]]):
            mandatory.append({"type": "constraint", "text": f"{c['constraint']}: {c['detail']}".strip(": "),
                              "source": c["source"], "evidence": c["evidence"]})
        for c in constraints or []:
            mandatory.append({"type": "constraint", "text": c, "source": "caller", "evidence": "request"})
        for v in validation or []:
            mandatory.append({"type": "validation", "text": v, "source": "caller", "evidence": "request"})
        if handoff:
            for c in handoff.get("constraints", []):
                mandatory.append({"type": "constraint", "text": c if isinstance(c, str) else json.dumps(c),
                                  "source": f"handoff:{handoff.get('id')}", "evidence": ev.HANDOFF})
            for v in handoff.get("validation", []):
                mandatory.append({"type": "validation", "text": v if isinstance(v, str) else json.dumps(v),
                                  "source": f"handoff:{handoff.get('id')}", "evidence": ev.HANDOFF})
        mandatory_tokens = sum(estimate_tokens(m["text"]) + 6 for m in mandatory)

        # Score, render and pack optional items.
        items = []
        for c in candidates.values():
            n = self.store.node(c["id"])
            if not n or n["status"] == "superseded":
                continue
            trust = ev.weight(n["evidence"], n["status"])
            value = c["score"] * (0.5 + 0.5 * trust)
            fact = _fact_line(n, c["reasons"])
            items.append({"id": n["id"], "kind": n["kind"], "value": round(value, 4), "fact": fact,
                          "tokens": estimate_tokens(fact), "node": n})
        items.sort(key=lambda i: -i["value"])
        remaining = budget - mandatory_tokens - 120   # header overhead
        facts, sources, omitted = [], [], []
        included_spans: list[tuple[str, int, int]] = []
        for it in items:
            if remaining <= 0:
                omitted.append({"id": it["id"], "reason": "budget"})
                continue
            if it["tokens"] > remaining:
                omitted.append({"id": it["id"], "reason": "budget"})
                continue
            facts.append({"id": it["id"], "kind": it["kind"], "fact": it["fact"], "relevance": it["value"],
                          "evidence": it["node"]["evidence"], "evidence_class": ev.classify(it["node"]["evidence"]),
                          "status": it["node"]["status"]})
            remaining -= it["tokens"]
        if include_source:
            for it in items[:12]:
                n = it["node"]
                if n["kind"] not in ("function", "method", "class", "test") or not n["file"]:
                    continue
                if any(f == n["file"] and a <= (n["line_start"] or 0) and (n["line_end"] or 0) <= b
                       for f, a, b in included_spans):
                    continue   # already inside an included enclosing snippet
                snippet = self._snippet(n, max_lines=40 if n["kind"] != "class" else 25)
                if not snippet:
                    continue
                cost = estimate_tokens(snippet) + 10
                if cost > remaining:
                    omitted.append({"id": n["id"], "reason": "source omitted for budget"})
                    continue
                sources.append({"id": n["id"], "location": f"{n['file']}:{n['line_start']}-{n['line_end']}",
                                "code": snippet})
                included_spans.append((n["file"], n["line_start"] or 0, n["line_end"] or 0))
                remaining -= cost
        used = budget - remaining if remaining > 0 else budget + (-remaining)
        insufficient = mandatory_tokens + 120 > budget
        next_actions = []
        if handoff and handoff.get("next_action"):
            next_actions.append(handoff["next_action"])
        top_tests = [f["id"] for f in facts if f["kind"] == "test"][:5]
        if top_tests:
            next_actions.append("run tests: " + ", ".join(t.rsplit(":", 1)[-1] for t in top_tests))
        pkg = {
            "task": task, "revision": self.ws.revision(), "indexed_revision": self.store.get_meta("indexed_revision"),
            "budget_tokens": budget, "estimated_tokens": min(used, max(used, mandatory_tokens)),
            "budget_insufficient": insufficient,
            **({"minimum_budget": mandatory_tokens + 120,
                "warning": "Critical constraints/validation exceed the budget; they were kept, not dropped."}
               if insufficient else {}),
            "constraints": [m for m in mandatory if m["type"] == "constraint"],
            "validation": [m for m in mandatory if m["type"] == "validation"],
            "facts": facts, "sources": sources, "relationships": _dedupe_rel(relationships)[:30],
            "next_actions": next_actions, "omitted": omitted[:40], "omitted_count": len(omitted),
            "retrieval": {"seeds": [{"id": s["id"], "score": s["score"], "signals": s.get("signals")}
                                    for s in seeds[:8]],
                          "mode": "hybrid (symbol + lexical + semantic + graph)" if self.ml else
                                  "hybrid (symbol + lexical + graph; ML disabled)"},
            "security_notice": UNTRUSTED_NOTICE, "cache": "miss",
        }
        self.store.set_meta(f"ctx:{key}", json.dumps(pkg))
        return pkg

    def _snippet(self, n: dict[str, Any], max_lines: int) -> str | None:
        try:
            path = safe_path(self.ws.root, n["file"])
            lines = path.read_text(errors="replace").splitlines()
        except (OSError, ValueError):
            return None
        a = max(1, n["line_start"] or 1)
        b = min(len(lines), n["line_end"] or a)
        chunk = lines[a - 1:min(b, a - 1 + max_lines)]
        if b - a + 1 > max_lines:
            chunk.append(f"    … ({b - a + 1 - max_lines} more lines)")
        return "\n".join(chunk)


def _fact_line(n: dict[str, Any], reasons: list[str]) -> str:
    where = f" @ {n['file']}:{n['line_start']}" if n.get("file") and n.get("line_start") else (
        f" @ {n['file']}" if n.get("file") else "")
    sig = ""
    if n["kind"] in ("function", "method") and n["attrs"].get("params") is not None:
        sig = "(" + ", ".join(p["name"] for p in n["attrs"]["params"]) + ")"
    doc = f" — {n['text'][:160]}" if n.get("text") else ""
    why = f" [{'; '.join(dict.fromkeys(reasons))[:120]}]" if reasons else ""
    return f"{n['kind']} {n.get('qualname') or n['name']}{sig}{where}{doc}{why}"


def _dedupe_rel(rels: list[dict]) -> list[dict]:
    seen, out = set(), []
    for r in sorted(rels, key=lambda r: -r["trust"]):
        k = (r["source"], r["type"], r["target"])
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out


def render_context(pkg: dict[str, Any]) -> str:
    lines = [f"# Context: {pkg['task']}", f"revision {pkg['revision']} · ~{pkg['estimated_tokens']} / "
             f"{pkg['budget_tokens']} tokens", ""]
    if pkg.get("budget_insufficient"):
        lines += [f"> WARNING: {pkg['warning']} Minimum budget: {pkg['minimum_budget']} tokens.", ""]
    if pkg["constraints"]:
        lines.append("## Constraints (must hold)")
        lines += [f"- {c['text']}  _(source: {c['source']})_" for c in pkg["constraints"]]
        lines.append("")
    if pkg["validation"]:
        lines.append("## Validation")
        lines += [f"- {v['text']}" for v in pkg["validation"]]
        lines.append("")
    lines.append("## Relevant facts")
    lines += [f"- [{f['evidence_class']}] {f['fact']}" for f in pkg["facts"]]
    if pkg["sources"]:
        lines += ["", "## Source excerpts", f"_{pkg['security_notice']}_"]
        for s in pkg["sources"]:
            lines += [f"### {s['location']}", "```", s["code"], "```"]
    if pkg["next_actions"]:
        lines += ["", "## Next actions"] + [f"- {a}" for a in pkg["next_actions"]]
    if pkg["omitted_count"]:
        lines += ["", f"_{pkg['omitted_count']} lower-ranked items omitted for budget._"]
    return "\n".join(lines)

