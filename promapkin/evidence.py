"""Evidence classification.

Every node and edge in the graph records *how* it was obtained. Traversal and
retrieval weight facts by the trustworthiness of that evidence, so a direct AST
call edge outranks an agent's architectural guess.
"""

from __future__ import annotations

# Evidence kinds, from most to least trustworthy.
AST = "ast"                      # parsed directly from a syntax tree
RUNTIME = "runtime"              # observed during execution (benchmarks, traces)
STATIC_RESOLVED = "static_resolved"  # cross-file resolution through imports
LEXICAL = "lexical"              # lexical/regex extraction (non-AST languages)
MANIFEST = "manifest"            # package manifests / build configuration
DOCUMENTATION = "documentation"  # claims made by docs, ADRs, comments
NAME_MATCH = "name_match"        # resolved only by matching an unqualified name
HEURISTIC = "heuristic"          # rule-based interpretation of structural facts
ML_CLASSIFIER = "ml_classifier"  # statistical classifier / embedding model over structural facts
AGENT_INFERENCE = "agent_inference"  # conclusions produced by an AI agent
HANDOFF = "handoff"              # recorded by an agent in a handoff
UNKNOWN = "unknown"

WEIGHTS: dict[str, float] = {
    AST: 1.0,
    RUNTIME: 0.95,
    STATIC_RESOLVED: 0.9,
    MANIFEST: 0.9,
    LEXICAL: 0.75,
    DOCUMENTATION: 0.7,
    NAME_MATCH: 0.6,
    HEURISTIC: 0.55,
    ML_CLASSIFIER: 0.6,
    HANDOFF: 0.6,
    AGENT_INFERENCE: 0.45,
    UNKNOWN: 0.2,
}

# Coarse classes used in reports: observed / documented / inferred.
CLASS_OF: dict[str, str] = {
    AST: "observed",
    RUNTIME: "observed",
    STATIC_RESOLVED: "observed",
    MANIFEST: "observed",
    LEXICAL: "observed",
    DOCUMENTATION: "documented",
    NAME_MATCH: "inferred",
    HEURISTIC: "inferred",
    ML_CLASSIFIER: "inferred",
    HANDOFF: "reported",
    AGENT_INFERENCE: "inferred",
    UNKNOWN: "unknown",
}

STALE_PENALTY = 0.5


def weight(evidence: str, status: str = "verified", corroborations: int = 0) -> float:
    """Trust weight of a fact given its evidence kind, freshness and corroboration."""
    w = WEIGHTS.get(evidence, WEIGHTS[UNKNOWN])
    if status == "stale":
        w *= STALE_PENALTY
    elif status == "superseded":
        w *= 0.1
    if corroborations:
        # Independent corroboration moves a fact toward certainty, never past 1.
        w = 1 - (1 - w) * (0.6 ** corroborations)
    return round(min(w, 1.0), 4)


def classify(evidence: str) -> str:
    return CLASS_OF.get(evidence, "unknown")
