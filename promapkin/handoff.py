"""Agent Handoff Engine: durable, versioned task records shared between agents.

A handoff is an index into project knowledge (findings, constraints, symbols,
decisions, next action, validation) rather than a transcript. Every update creates
a new version; concurrent writers use optimistic concurrency (expected_version),
list fields are appended rather than overwritten, contradictory findings are recorded
as conflicts, and tasks can branch, be abandoned or superseded.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from promapkin import evidence as ev
from promapkin import gitutil, knowledge
from promapkin.config import Workspace
from promapkin.graph import Graph

HANDOFF_VERSION = 1
STATUSES = {"open", "investigating", "investigation_complete", "planned", "in_progress", "blocked", "review",
            "complete", "abandoned", "superseded"}
LIST_FIELDS = ("findings", "constraints", "decisions", "completed", "validation", "open_questions", "artifacts")
FINDING_STATUSES = {"verified", "unverified", "refuted", "stale", "superseded"}


class HandoffError(ValueError):
    pass


class VersionConflict(HandoffError):
    def __init__(self, latest: dict[str, Any]):
        super().__init__(f"handoff {latest['id']} is at version {latest['version']}; reload and retry")
        self.latest = latest


def _norm_list(v: Any) -> list:
    if v is None:
        return []
    return v if isinstance(v, list) else [v]


def normalize(data: dict[str, Any]) -> dict[str, Any]:
    out = {
        "handoff_version": HANDOFF_VERSION,
        "task": str(data.get("task") or "").strip(),
        "status": data.get("status") or "open",
        "summary": data.get("summary") or "",
        "findings": [], "constraints": [], "decisions": [], "completed": [], "validation": [],
        "open_questions": [], "artifacts": [],
        "relevant_context": {"symbols": [], "dependencies": [], "files": []},
        "next_action": data.get("next_action") or "",
    }
    if not out["task"]:
        raise HandoffError("task is required")
    if out["status"] not in STATUSES:
        raise HandoffError(f"status must be one of {sorted(STATUSES)}")
    for f in _norm_list(data.get("findings")):
        out["findings"].append(_finding(f))
    for k in ("constraints", "completed", "validation", "open_questions", "artifacts"):
        out[k] = [x if isinstance(x, (str, dict)) else str(x) for x in _norm_list(data.get(k))]
    for d in _norm_list(data.get("decisions")):
        out["decisions"].append(d if isinstance(d, dict) else {"decision": str(d), "rationale": "unknown"})
    rc = data.get("relevant_context") or {}
    for k in ("symbols", "dependencies", "files"):
        out["relevant_context"][k] = [str(x) for x in _norm_list(rc.get(k))]
    return out


def _finding(f: Any) -> dict[str, Any]:
    if isinstance(f, str):
        f = {"claim": f}
    if not f.get("claim"):
        raise HandoffError("each finding needs a claim")
    status = f.get("status", "unverified")
    if status not in FINDING_STATUSES:
        raise HandoffError(f"finding status must be one of {sorted(FINDING_STATUSES)}")
    evidence = f.get("evidence", "agent_inference")
    sources = _norm_list(f.get("sources") or f.get("source"))
    return {"claim": str(f["claim"]), "sources": [str(s) for s in sources], "evidence": str(evidence),
            "status": status, "confidence": float(f.get("confidence", 0.6 if status == "verified" else 0.4)),
            **({"supersedes": f["supersedes"]} if f.get("supersedes") else {}),
            **({"notes": f["notes"]} if f.get("notes") else {})}


class Handoffs:
    def __init__(self, ws: Workspace):
        self.ws = ws
        self.store = ws.store
        self.graph = Graph(ws.store)

    # ---------------------------------------------------------------- create / branch
    def create(self, data: dict[str, Any], agent: str = "unknown", parent_id: str | None = None) -> dict[str, Any]:
        rec = normalize(data)
        if parent_id:
            parent = self.store.latest_handoff(parent_id)
            if not parent:
                raise HandoffError(f"no handoff {parent_id}")
            p = parent["data"]
            rec["constraints"] = _merge_unique(p.get("constraints", []), rec["constraints"])
            for k in ("symbols", "dependencies", "files"):
                rec["relevant_context"][k] = _merge_unique(p["relevant_context"].get(k, []),
                                                           rec["relevant_context"][k])
        hid = "ho-" + knowledge.slugify(rec["task"], 32).replace("_", "-") + "-" + \
            hashlib.sha1(f"{rec['task']}{time.time()}{agent}".encode()).hexdigest()[:6]
        rec["id"] = hid
        rec["parent_id"] = parent_id
        rec["repository"] = self._repo_state()
        rec["agent"] = agent
        rec["history"] = [{"version": 1, "agent": agent, "at": time.time(), "change": "created"}]
        with self.store.transaction():
            self.store.insert_handoff({"id": hid, "version": 1, "task": rec["task"], "status": rec["status"],
                                       "parent_id": parent_id, "agent": agent, "revision": rec["repository"]["revision"],
                                       "created_at": time.time(), "data": rec})
            graph_report = self._project(rec, agent)
        self._export(rec, 1)
        return {"id": hid, "version": 1, "handoff": rec, "graph": graph_report}

    def branch(self, parent_id: str, task: str, agent: str = "unknown", **fields: Any) -> dict[str, Any]:
        return self.create({"task": task, **fields}, agent=agent, parent_id=parent_id)

    # ---------------------------------------------------------------- update
    def update(self, hid: str, patch: dict[str, Any], agent: str = "unknown",
               expected_version: int | None = None) -> dict[str, Any]:
        with self.store.transaction():
            latest = self.store.latest_handoff(hid)
            if not latest:
                raise HandoffError(f"no handoff {hid}")
            if expected_version is not None and expected_version != latest["version"]:
                raise VersionConflict({"id": hid, "version": latest["version"], "status": latest["status"],
                                       "agent": latest["agent"]})
            rec = json.loads(json.dumps(latest["data"]))
            changes = []
            if "status" in patch:
                if patch["status"] not in STATUSES:
                    raise HandoffError(f"status must be one of {sorted(STATUSES)}")
                rec["status"] = patch["status"]
                changes.append(f"status={patch['status']}")
            for k in ("next_action", "summary"):
                if k in patch:
                    rec[k] = patch[k]
                    changes.append(k)
            conflicts = []
            for f in _norm_list(patch.get("findings")):
                nf = _finding(f)
                for old in rec["findings"]:
                    if old["claim"].strip().lower() == nf["claim"].strip().lower() and old["status"] != nf["status"] \
                            and "supersedes" not in nf:
                        conflicts.append({"claim": nf["claim"], "existing": old["status"], "new": nf["status"],
                                          "agent": agent})
                        old.setdefault("conflicts", []).append({"status": nf["status"], "agent": agent})
                        nf.setdefault("conflicts", []).append({"status": old["status"]})
                    if nf.get("supersedes") and old["claim"] == nf["supersedes"]:
                        old["status"] = "superseded"
                rec["findings"].append(nf)
                changes.append("finding")
            for k in ("constraints", "completed", "validation", "open_questions", "artifacts"):
                if k in patch:
                    rec[k] = _merge_unique(rec.get(k, []), _norm_list(patch[k]))
                    changes.append(k)
            for d in _norm_list(patch.get("decisions")):
                rec["decisions"].append(d if isinstance(d, dict) else {"decision": str(d), "rationale": "unknown"})
                changes.append("decision")
            rc = patch.get("relevant_context") or {}
            for k in ("symbols", "dependencies", "files"):
                if rc.get(k):
                    rec["relevant_context"][k] = _merge_unique(rec["relevant_context"].get(k, []), rc[k])
                    changes.append(f"context.{k}")
            if not changes:
                raise HandoffError("patch contains no recognised fields")
            if conflicts:
                rec.setdefault("conflicts", []).extend(conflicts)
            version = latest["version"] + 1
            rec["repository"] = self._repo_state()
            rec["agent"] = agent
            rec.setdefault("history", []).append({"version": version, "agent": agent, "at": time.time(),
                                                  "change": ", ".join(changes)})
            self.store.insert_handoff({"id": hid, "version": version, "task": rec["task"], "status": rec["status"],
                                       "parent_id": rec.get("parent_id"), "agent": agent,
                                       "revision": rec["repository"]["revision"], "created_at": time.time(),
                                       "data": rec})
            graph_report = self._project(rec, agent)
        self._export(rec, version)
        return {"id": hid, "version": version, "conflicts": conflicts, "handoff": rec, "graph": graph_report}

    # ---------------------------------------------------------------- resume
    def resume(self, hid: str, budget: int | None = None, agent: str | None = None, claim: bool = False,
               include_source: bool = True) -> dict[str, Any]:
        latest = self.store.latest_handoff(hid)
        if not latest:
            raise HandoffError(f"no handoff {hid}")
        rec = latest["data"]
        freshness = self.freshness(rec)
        from promapkin.retrieval import Retriever
        pkg = Retriever(self.ws).build_context(rec["task"], budget=budget, handoff=rec, include_source=include_source)
        lineage = []
        parent = rec.get("parent_id")
        while parent and len(lineage) < 10:
            p = self.store.latest_handoff(parent)
            if not p:
                break
            lineage.append({"id": p["id"], "task": p["task"], "status": p["status"]})
            parent = p["data"].get("parent_id")
        children = [h["id"] for h in self.store.list_handoffs() if h["parent_id"] == hid]
        out = {"id": hid, "version": latest["version"], "handoff": rec, "freshness": freshness, "context": pkg,
               "lineage": lineage, "branches": children,
               "instructions": [
                   "Validate the findings marked stale or changed before relying on them.",
                   "Respect every constraint; they were kept regardless of the token budget.",
                   "Append new findings with update_handoff (expected_version guards concurrent edits).",
               ]}
        if claim and agent:
            upd = self.update(hid, {"status": "in_progress"}, agent=agent, expected_version=latest["version"])
            out["version"] = upd["version"]
            out["claimed_by"] = agent
        return out

    def freshness(self, rec: dict[str, Any]) -> dict[str, Any]:
        then = (rec.get("repository") or {}).get("revision", "")
        now = self.ws.revision()
        changed = gitutil.changed_since(self.ws.root, then) if then and then != now else []
        changed = set(changed or [])
        symbols = []
        for sym in rec["relevant_context"].get("symbols", []):
            try:
                nid = self.graph.resolve(sym)
                n = self.store.node(nid)
                symbols.append({"symbol": sym, "id": nid, "exists": True, "file": n.get("file") if n else None,
                                "changed_since_handoff": bool(n and n.get("file") in changed)})
            except LookupError:
                symbols.append({"symbol": sym, "exists": False, "changed_since_handoff": True})
        findings = []
        for f in rec.get("findings", []):
            cites = [knowledge.resolve_citation(self.store, self.ws.root, s) for s in f.get("sources", [])]
            touched = any(c.get("file") in changed for c in cites if c.get("valid"))
            broken = [c["ref"] for c in cites if not c["valid"]]
            state = f["status"]
            if f["status"] == "verified" and (touched or broken):
                state = "needs_revalidation"
            findings.append({"claim": f["claim"], "recorded_status": f["status"], "current": state,
                             "sources_changed": touched, "broken_sources": broken})
        return {"recorded_revision": then, "current_revision": now, "revision_changed": then != now,
                "files_changed_since": sorted(changed)[:100], "symbols": symbols, "findings": findings,
                "stale": any(s["changed_since_handoff"] for s in symbols) or any(
                    f["current"] == "needs_revalidation" for f in findings)}

    # ---------------------------------------------------------------- queries
    def get(self, hid: str, version: int | None = None) -> dict[str, Any]:
        versions = self.store.handoff_versions(hid)
        if not versions:
            raise HandoffError(f"no handoff {hid}")
        if version is None:
            return versions[-1]
        for v in versions:
            if v["version"] == version:
                return v
        raise HandoffError(f"handoff {hid} has no version {version}")

    def history(self, hid: str) -> list[dict[str, Any]]:
        return [{"version": v["version"], "status": v["status"], "agent": v["agent"], "revision": v["revision"],
                 "at": v["created_at"], "change": (v["data"].get("history") or [{}])[-1].get("change")}
                for v in self.store.handoff_versions(hid)]

    def list(self, status: str | None = None) -> list[dict[str, Any]]:
        out = []
        for h in self.store.list_handoffs():
            if status and h["status"] != status:
                continue
            out.append({"id": h["id"], "task": h["task"], "status": h["status"], "version": h["version"],
                        "agent": h["agent"], "parent_id": h["parent_id"], "revision": h["revision"],
                        "updated_at": h["created_at"], "next_action": h["data"].get("next_action")})
        return out

    # ---------------------------------------------------------------- graph projection
    def _project(self, rec: dict[str, Any], agent: str) -> dict[str, Any]:
        """Mirror the handoff into the graph so retrieval and impact analysis can see it."""
        s = self.store
        hid = f"handoff:{rec['id']}"
        rev = rec["repository"]["revision"]
        s.upsert_node(hid, "handoff", rec["task"], evidence=ev.HANDOFF, confidence=0.6, revision=rev,
                      text=f"{rec['task']}. {rec.get('summary', '')} Next: {rec.get('next_action', '')}",
                      status="superseded" if rec["status"] in ("abandoned", "superseded") else "verified",
                      attrs={"status": rec["status"], "agent": agent, "handoff_id": rec["id"],
                             "parent_id": rec.get("parent_id")})
        if rec.get("parent_id"):
            s.add_edge(hid, f"handoff:{rec['parent_id']}", "depends_on", evidence=ev.HANDOFF)
        linked, unresolved = [], []
        for sym in rec["relevant_context"].get("symbols", []) + rec["relevant_context"].get("dependencies", []):
            try:
                nid = self.graph.resolve(sym)
            except LookupError:
                unresolved.append(sym)
                continue
            s.add_edge(hid, nid, "affects", evidence=ev.HANDOFF, revision=rev, attrs={"agent": agent})
            linked.append(nid)
        findings = []
        for f in rec["findings"]:
            fid = "finding:" + knowledge.slugify(f["claim"], 40) + "-" + hashlib.sha1(
                f"{rec['id']}{f['claim']}".encode()).hexdigest()[:8]
            out = knowledge.add_node(
                s, self.ws.root, kind="finding", name=f["claim"][:160], text=f.get("notes", f["claim"]),
                evidence=ev.RUNTIME if f["evidence"] in ("benchmark", "runtime", "test_run") else ev.HANDOFF,
                confidence=f["confidence"], citations=f["sources"] + linked[:3], agent=agent, revision=rev,
                node_id=fid, attrs={"handoff": rec["id"], "finding_status": f["status"],
                                    "reported_evidence": f["evidence"]})
            if f["status"] in ("refuted", "superseded"):
                s.set_node_status(fid, "superseded")
            s.add_edge(hid, fid, "supports", evidence=ev.HANDOFF, revision=rev)
            for c in f.get("conflicts", []):
                s.add_edge(fid, hid, "conflicts_with", evidence=ev.HANDOFF, revision=rev, attrs=c)
            findings.append(out["id"])
        for c in rec["constraints"]:
            text = c if isinstance(c, str) else json.dumps(c)
            cid = "constraint:" + knowledge.slugify(text, 48)
            s.upsert_node(cid, "constraint", text[:160], evidence=ev.HANDOFF, revision=rev, text=text,
                          attrs={"handoff": rec["id"], "agent": agent})
            s.add_edge(hid, cid, "contains", evidence=ev.HANDOFF)
            for nid in linked:
                s.add_edge(cid, nid, "constrains", evidence=ev.HANDOFF, revision=rev)
        return {"node": hid, "linked_symbols": linked, "unresolved_symbols": unresolved, "findings": findings}

    def _repo_state(self) -> dict[str, Any]:
        return {"revision": self.ws.revision(), "indexed_revision": self.store.get_meta("indexed_revision")}

    def _export(self, rec: dict[str, Any], version: int) -> None:
        d = self.ws.dir / "handoffs"
        d.mkdir(exist_ok=True)
        (d / f"{rec['id']}.yaml").write_text(to_yaml({**{k: v for k, v in rec.items() if k != "history"},
                                                      "version": version}))


def _merge_unique(a: list, b: list) -> list:
    out = list(a)
    seen = {json.dumps(x, sort_keys=True) for x in a}
    for x in b:
        k = json.dumps(x, sort_keys=True)
        if k not in seen:
            seen.add(k)
            out.append(x)
    return out


def to_yaml(obj: Any, indent: int = 0) -> str:
    """Minimal YAML emitter for handoff exports (dicts, lists, scalars)."""
    pad = "  " * indent
    if isinstance(obj, dict):
        if not obj:
            return pad + "{}\n"
        out = ""
        for k, v in obj.items():
            if isinstance(v, (dict, list)) and v:
                out += f"{pad}{k}:\n" + to_yaml(v, indent + 1)
            else:
                out += f"{pad}{k}: {_scalar(v)}\n"
        return out
    if isinstance(obj, list):
        if not obj:
            return pad + "[]\n"
        out = ""
        for v in obj:
            if isinstance(v, dict) and v:
                body = to_yaml(v, indent + 1)
                out += f"{pad}- " + body.lstrip()
            elif isinstance(v, list) and v:
                out += f"{pad}-\n" + to_yaml(v, indent + 1)
            else:
                out += f"{pad}- {_scalar(v)}\n"
        return out
    return pad + _scalar(obj) + "\n"


def _scalar(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, (dict, list)):
        return json.dumps(v)
    s = str(v)
    if s == "" or any(c in s for c in ":#{}[],&*!|>'\"%@`\n") or s.strip() != s or s.lower() in (
            "true", "false", "null", "yes", "no"):
        return json.dumps(s)
    return s
