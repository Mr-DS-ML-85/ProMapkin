"""Agent-writable knowledge: findings, concepts, decisions, relationships and annotations.

Agents never overwrite shared knowledge silently. They append facts with provenance
(agent, evidence class, citations, revision) and may supersede earlier agent facts.
Statically extracted facts are read-only: an agent cannot claim AST evidence, delete
extracted nodes, or attach edges to nodes that do not exist.
"""

from __future__ import annotations

import hashlib
import re
import time
from typing import Any

from promapkin import evidence as ev
from promapkin.security import PathOutsideRepository, safe_path
from promapkin.store import Store

AGENT_NODE_KINDS = {
    "finding", "concept", "pattern", "decision", "requirement", "bug", "task", "roadmap_step", "risk", "question",
    "constraint", "annotation", "event", "data_entity", "component", "subsystem", "note", "map",
}
RELATIONSHIP_TYPES = {
    "contains", "defines", "imports", "calls", "references", "reads", "writes", "invokes", "produces", "consumes",
    "configures", "implements", "supports", "conflicts_with", "verifies", "constrains", "affects", "depends_on",
    "supersedes", "about", "annotated_as", "implements_pattern", "documents", "routes_to", "part_of", "blocks", "similar_to",
}
AGENT_EVIDENCE = {ev.AGENT_INFERENCE, ev.RUNTIME, ev.DOCUMENTATION, ev.HANDOFF, ev.HEURISTIC}


class KnowledgeError(ValueError):
    pass


def slugify(text: str, n: int = 48) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:n] or "item"


def resolve_citation(store: Store, root: Any, citation: str) -> dict[str, Any]:
    """A citation is a node id, or 'path[:line[-line]]' inside the repository. Returns a verified citation."""
    citation = citation.strip()
    node = store.node(citation)
    if node:
        return {"ref": citation, "kind": "node", "valid": True, "file": node.get("file"), "status": node["status"]}
    m = re.match(r"^(?P<path>[^:]+?)(?::(?P<a>\d+)(?:-(?P<b>\d+))?)?$", citation)
    if m and root is not None:
        try:
            p = safe_path(root, m.group("path"))
        except PathOutsideRepository as exc:
            return {"ref": citation, "kind": "file", "valid": False, "error": str(exc)}
        if p.is_file():
            lines = p.read_text(errors="replace").count("\n") + 1
            a = int(m.group("a") or 1)
            valid = 1 <= a <= lines
            return {"ref": citation, "kind": "file", "valid": valid, "file": m.group("path"),
                    **({} if valid else {"error": f"line {a} beyond end of file ({lines} lines)"})}
        return {"ref": citation, "kind": "file", "valid": False, "error": "no such file or node"}
    return {"ref": citation, "kind": "unknown", "valid": False, "error": "no such node"}


def add_node(store: Store, root: Any, *, kind: str, name: str, text: str = "", evidence: str = ev.AGENT_INFERENCE,
             confidence: float = 0.6, citations: list[str] | None = None, relation: str = "about",
             attrs: dict[str, Any] | None = None, agent: str = "unknown", revision: str | None = None,
             node_id: str | None = None, require_citation: bool = True) -> dict[str, Any]:
    if kind not in AGENT_NODE_KINDS:
        raise KnowledgeError(f"agents may create nodes of kinds {sorted(AGENT_NODE_KINDS)}, not {kind!r}")
    if evidence not in AGENT_EVIDENCE:
        raise KnowledgeError(f"evidence must be one of {sorted(AGENT_EVIDENCE)} (static evidence is extractor-only)")
    if relation not in RELATIONSHIP_TYPES:
        raise KnowledgeError(f"unknown relationship type {relation!r}")
    if not name.strip():
        raise KnowledgeError("name is required")
    confidence = max(0.0, min(float(confidence), 1.0))
    cites = [resolve_citation(store, root, c) for c in (citations or [])]
    valid = [c for c in cites if c["valid"]]
    if require_citation and kind in ("finding", "decision", "risk", "bug") and not valid:
        # Unsupported claims are kept, but clearly marked and down-weighted.
        confidence = min(confidence, 0.3)
    nid = node_id or f"{kind}:{slugify(name)}-{hashlib.sha1(f'{name}{text}{agent}'.encode()).hexdigest()[:8]}"
    if node_id and (existing := store.node(node_id)) and existing["evidence"] in (ev.AST, ev.LEXICAL, ev.MANIFEST):
        raise KnowledgeError("cannot overwrite a statically extracted node")
    files = sorted({c["file"] for c in valid if c.get("file")})
    store.upsert_node(
        nid, kind, name[:200], evidence=evidence, confidence=confidence, revision=revision, text=text[:4000],
        status="verified" if valid or not require_citation else "unsupported",
        attrs={**(attrs or {}), "agent": agent, "citations": cites, "files": files, "created_at": time.time()},
    )
    for c in valid:
        if c["kind"] == "node":
            store.add_edge(nid, c["ref"], relation, evidence=evidence, confidence=confidence, revision=revision,
                           attrs={"agent": agent})
        elif c.get("file") and store.node(f"file:{c['file']}"):
            store.add_edge(nid, f"file:{c['file']}", relation, evidence=evidence, confidence=confidence,
                           revision=revision, attrs={"agent": agent, "at": c["ref"]})
    return {"id": nid, "kind": kind, "status": "verified" if valid or not require_citation else "unsupported",
            "confidence": confidence, "citations": cites}


def add_relationship(store: Store, *, src: str, dst: str, type: str, evidence: str = ev.AGENT_INFERENCE,
                     confidence: float = 0.6, rationale: str = "", agent: str = "unknown",
                     revision: str | None = None) -> dict[str, Any]:
    if type not in RELATIONSHIP_TYPES:
        raise KnowledgeError(f"relationship type must be one of {sorted(RELATIONSHIP_TYPES)}")
    if evidence not in AGENT_EVIDENCE:
        raise KnowledgeError(f"evidence must be one of {sorted(AGENT_EVIDENCE)}")
    for n in (src, dst):
        if not store.node(n):
            raise KnowledgeError(f"node does not exist: {n}")
    existing = store.edges(src=src, dst=dst, types=[type])
    store.add_edge(src, dst, type, evidence=evidence, confidence=max(0.0, min(float(confidence), 1.0)),
                   revision=revision, attrs={"agent": agent, "rationale": rationale[:1000]})
    corroborated_by = [e["evidence"] for e in existing if e["evidence"] != evidence]
    return {"source": src, "type": type, "target": dst, "evidence": evidence,
            "corroborated_by": corroborated_by,
            "conflicts": _conflicts(store, src, dst, type)}


def _conflicts(store: Store, src: str, dst: str, type: str) -> list[dict[str, Any]]:
    out = []
    for e in store.edges(src=src, dst=dst, types=["conflicts_with"]) + store.edges(src=dst, dst=src,
                                                                                 types=["conflicts_with"]):
        out.append({"source": e["src"], "target": e["dst"], "evidence": e["evidence"]})
    return out


def supersede(store: Store, old_id: str, new_id: str, agent: str = "unknown", reason: str = "") -> dict[str, Any]:
    old, new = store.node(old_id), store.node(new_id)
    if not old or not new:
        raise KnowledgeError("both nodes must exist")
    if old["evidence"] in (ev.AST, ev.LEXICAL, ev.MANIFEST, ev.STATIC_RESOLVED):
        raise KnowledgeError("static facts are superseded by re-scanning, not by agents")
    store.set_node_status(old_id, "superseded")
    store.add_edge(new_id, old_id, "supersedes", evidence=new["evidence"], attrs={"agent": agent, "reason": reason})
    return {"superseded": old_id, "by": new_id}


def annotate(store: Store, node_id: str, concept: str, *, confidence: float = 0.6, agent: str = "unknown",
             evidence: str = ev.AGENT_INFERENCE, revision: str | None = None) -> dict[str, Any]:
    if not store.node(node_id):
        raise KnowledgeError(f"node does not exist: {node_id}")
    cid = f"concept:{slugify(concept)}"
    if not store.node(cid):
        store.upsert_node(cid, "concept", concept, evidence=evidence, confidence=confidence, revision=revision,
                          attrs={"agent": agent})
    store.add_edge(node_id, cid, "annotated_as", evidence=evidence, confidence=confidence, revision=revision,
                   attrs={"agent": agent})
    return {"node": node_id, "concept": cid, "origin": evidence, "confidence": confidence}
