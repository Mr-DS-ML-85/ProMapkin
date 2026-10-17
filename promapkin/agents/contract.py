"""Provider-independent task contract and result validation for specialist agents."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from promapkin import evidence as ev
from promapkin import knowledge
from promapkin.security import UNTRUSTED_NOTICE

RESULT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["summary", "claims"],
    "properties": {
        "summary": {"type": "string", "maxLength": 4000},
        "claims": {
            "type": "array", "maxItems": 50,
            "items": {
                "type": "object",
                "required": ["claim", "citations"],
                "properties": {
                    "claim": {"type": "string", "maxLength": 1000},
                    "kind": {"type": "string", "enum": ["finding", "pattern", "decision", "risk", "concept",
                                                         "requirement", "constraint", "question", "roadmap_step"]},
                    "citations": {"type": "array", "items": {"type": "string"}, "maxItems": 20},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "evidence": {"type": "string", "enum": [ev.AGENT_INFERENCE, ev.RUNTIME, ev.DOCUMENTATION]},
                    "rationale": {"type": "string", "maxLength": 2000},
                },
            },
        },
        "relationships": {
            "type": "array", "maxItems": 100,
            "items": {
                "type": "object",
                "required": ["source", "target", "type"],
                "properties": {
                    "source": {"type": "string"}, "target": {"type": "string"},
                    "type": {"type": "string", "enum": sorted(knowledge.RELATIONSHIP_TYPES)},
                    "rationale": {"type": "string"}, "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
            },
        },
        "annotations": {
            "type": "array", "maxItems": 100,
            "items": {"type": "object", "required": ["node", "concept"],
                      "properties": {"node": {"type": "string"}, "concept": {"type": "string"},
                                     "confidence": {"type": "number", "minimum": 0, "maximum": 1}}},
        },
        "questions": {"type": "array", "items": {"type": "string"}, "maxItems": 20},
    },
}


@dataclass
class TaskContract:
    specialist: str
    objective: str
    scope: list[str]
    revision: str
    facts: dict[str, Any]
    allowed_tools: list[str] = field(default_factory=lambda: [
        "get_node", "search_graph", "get_source", "trace_dataflow", "analyze_impact"])
    permitted_operations: list[str] = field(default_factory=lambda: ["read repository", "submit_analysis"])
    output_schema: dict[str, Any] = field(default_factory=lambda: RESULT_SCHEMA)
    evidence_requirements: str = (
        "Every claim must cite graph node ids or repository paths (path:line). Claims that cannot be grounded "
        "must be labelled as inferred with low confidence or asked as questions. Never present a rationale as "
        "historical fact unless a document states it."
    )
    budget_tokens: int = 6000
    validation: list[str] = field(default_factory=lambda: [
        "output matches output_schema", "citations resolve to existing nodes or files", "no static-evidence claims"])
    retry_limit: int = 2
    security_notice: str = UNTRUSTED_NOTICE
    id: str = ""
    created_at: float = 0.0

    def __post_init__(self) -> None:
        if not self.id:
            digest = hashlib.sha1(json.dumps([self.specialist, self.objective, self.scope, self.revision],
                                             sort_keys=True).encode()).hexdigest()[:10]
            self.id = f"task:{self.specialist}:{digest}"
        if not self.created_at:
            self.created_at = time.time()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ------------------------------------------------------------------------------- validation

def validate(value: Any, schema: dict[str, Any], path: str = "$") -> list[str]:
    """A small JSON-schema subset validator (type, required, properties, items, enum, bounds)."""
    errors: list[str] = []
    t = schema.get("type")
    types = {"object": dict, "array": list, "string": str, "number": (int, float), "integer": int, "boolean": bool}
    if t and not isinstance(value, types[t]) or (t in ("number", "integer") and isinstance(value, bool)):
        return [f"{path}: expected {t}"]
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} not in {schema['enum']}")
    if t == "object":
        for r in schema.get("required", []):
            if r not in value:
                errors.append(f"{path}.{r}: required")
        for k, sub in schema.get("properties", {}).items():
            if k in value:
                errors += validate(value[k], sub, f"{path}.{k}")
    elif t == "array":
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append(f"{path}: more than {schema['maxItems']} items")
        for i, item in enumerate(value):
            errors += validate(item, schema.get("items", {}), f"{path}[{i}]")
    elif t == "string":
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errors.append(f"{path}: longer than {schema['maxLength']}")
    elif t in ("number", "integer"):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: below {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: above {schema['maximum']}")
    return errors


def merge_result(store: Any, root: Any, contract: dict[str, Any], result: dict[str, Any], agent: str) -> dict[str, Any]:
    """Validate a specialist result and merge its claims into the graph as agent inference."""
    errors = validate(result, RESULT_SCHEMA)
    if errors:
        return {"accepted": False, "errors": errors[:20]}
    revision = contract.get("revision")
    merged: dict[str, list] = {"claims": [], "relationships": [], "annotations": [], "rejected": []}
    with store.transaction():
        for c in result.get("claims", []):
            kind = c.get("kind", "finding")
            if kind not in knowledge.AGENT_NODE_KINDS:
                kind = "finding"
            out = knowledge.add_node(
                store, root, kind=kind, name=c["claim"][:120], text=c.get("rationale", c["claim"]),
                evidence=c.get("evidence", ev.AGENT_INFERENCE), confidence=c.get("confidence", 0.5),
                citations=c.get("citations", []), agent=agent, revision=revision,
                attrs={"task": contract.get("id"), "specialist": contract.get("specialist"), "claim": c["claim"]},
            )
            merged["claims"].append(out)
        for r in result.get("relationships", []):
            try:
                merged["relationships"].append(knowledge.add_relationship(
                    store, src=r["source"], dst=r["target"], type=r["type"], rationale=r.get("rationale", ""),
                    confidence=r.get("confidence", 0.5), agent=agent, revision=revision))
            except knowledge.KnowledgeError as exc:
                merged["rejected"].append({"relationship": r, "error": str(exc)})
        for a in result.get("annotations", []):
            try:
                merged["annotations"].append(knowledge.annotate(store, a["node"], a["concept"],
                                                                confidence=a.get("confidence", 0.5), agent=agent,
                                                                revision=revision))
            except knowledge.KnowledgeError as exc:
                merged["rejected"].append({"annotation": a, "error": str(exc)})
        if result.get("summary"):
            sid = f"finding:summary_{contract.get('specialist')}_{(contract.get('id') or '')[-10:]}"
            store.upsert_node(sid, "finding", f"{contract.get('specialist')} synthesis", evidence=ev.AGENT_INFERENCE,
                              confidence=0.5, revision=revision, text=result["summary"][:4000],
                              attrs={"agent": agent, "task": contract.get("id"), "synthesis": True,
                                     "specialist": contract.get("specialist"),
                                     "questions": result.get("questions", [])})
    unsupported = sum(1 for c in merged["claims"] if c["status"] == "unsupported")
    return {"accepted": True, "merged_claims": len(merged["claims"]), "unsupported_claims": unsupported,
            "relationships": len(merged["relationships"]), "annotations": len(merged["annotations"]),
            "rejected": merged["rejected"], "claims": merged["claims"]}
