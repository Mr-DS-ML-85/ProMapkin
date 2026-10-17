"""Two-tier classification: heuristic rules alongside a statistical / neural model.

Every decision runs *both* classifiers when a model tier is available:

* rules: deterministic, explainable patterns (always available);
* model: a neural embedding prototype classifier when a small embedding model is present,
  otherwise a built-in multinomial naive Bayes model trained on seed examples.

Agreement raises confidence; disagreement keeps the stronger answer and records the
alternative, so downstream evidence reflects the uncertainty. Users and agents can add
labelled examples (``.promapkin/training/<task>.jsonl``) that both model tiers learn from.
With ``mode="heuristic"`` only the rules run.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from promapkin.extract.resources import classify_call
from promapkin.ml.embedders import Embedder, HashingEmbedder, cosine, normalize


@dataclass
class Task:
    name: str
    labels: dict[str, str]                       # label -> description
    seeds: dict[str, list[str]]                  # label -> example texts
    rules: Callable[[str], tuple[str, float] | None]
    abstain: str = "unknown"


# ------------------------------------------------------------------------------- rules

def _resource_rules(text: str) -> tuple[str, float] | None:
    call = text.split(" ", 1)[-1]
    lib = text.split(" ", 1)[0]
    hit = classify_call(call, lib) or classify_call(lib + "." + call.split(".", 1)[-1])
    return (hit[0], 0.75) if hit else None


_LAYER_WORDS = {
    "interface": r"api|apis|routes?|routers?|views?|controllers?|handlers?|endpoints?|cli|commands?|web|http|"
                 r"server|ui|pages?|components?|graphql|rest|resources|mcp",
    "application": r"services?|usecases?|use_cases|application|workflows?|orchestrat\w*|jobs?|tasks?|commands_bus|"
                   r"managers?|pipelines?",
    "domain": r"domain|models?|entities|entity|core|logic|engine|rules|policies|aggregates?",
    "persistence": r"db|database|repository|repositories|repos?|storage|store|stores|dao|migrations?|orm|schema|"
                   r"persistence|cache",
    "integration": r"clients?|adapters?|gateways?|integrations?|external|providers?|sdk|connectors?|webhooks?",
    "infrastructure": r"config|settings|infra|deploy|logging|middleware|bootstrap|wsgi|asgi|setup",
    "shared": r"utils?|helpers?|common|shared|lib|misc|types",
}


def _layer_rules(text: str) -> tuple[str, float] | None:
    name = text.split(" ", 1)[0].lower()
    parts = re.split(r"[./\\]", name)
    if any(p in ("tests", "test", "__tests__", "spec") or p.startswith("test_") for p in parts):
        return "test", 0.95
    # The most specific (last) path segment decides; earlier segments break ties.
    for weight, part in zip((0.85, 0.7, 0.6, 0.55), reversed(parts)):
        for label, pattern in _LAYER_WORDS.items():
            if re.fullmatch(f"(_?)({pattern})", part):
                return label, weight
    return None


_DOC_WORDS = {
    "architecture": r"architecture|design|overview|components?|structure|modules?|system|diagram|data ?flow",
    "decision": r"decisions?|adr|rationale|why|trade-?offs?|alternatives",
    "api": r"\bapi\b|endpoints?|reference|usage|interface|routes?|schema",
    "setup": r"install\w*|setup|getting started|quick ?start|configur\w*|deploy\w*|build\w*|development",
    "requirement": r"requirements?|constraints?|specification|acceptance|non-functional|must",
    "operations": r"operations?|monitoring|runbook|troubleshoot\w*|faq|security",
}


def _doc_rules(text: str) -> tuple[str, float] | None:
    heading = text.split("\n", 1)[0].lower()
    for label, pattern in _DOC_WORDS.items():
        if re.search(pattern, heading):
            return label, 0.8
    return None


def _note_rules(text: str) -> tuple[str, float] | None:
    low = text.lower()
    if re.search(r"\b(todo|fixme|xxx)\b", low):
        return "todo", 0.95
    if re.search(r"\b(hack|workaround|warning|careful|security|danger|do not)\b", low):
        return "warning", 0.8
    if re.search(r"\b(because|we (chose|decided|use|prefer)|rationale|trade-?off|deliberately|intentionally|so that)\b",
                 low):
        return "rationale", 0.8
    return None


def _feature(desc: str, key: str) -> str:
    m = re.search(rf"\b{key}: ([^;]*)", desc)
    return m.group(1).strip() if m else ""


def _pattern_rules(desc: str) -> tuple[str, float] | None:
    """Static-confidence design-pattern rules over a structural class description (see describe_class)."""
    name = _feature(desc, "class").lower()
    if not name:   # free-form description: take the first CamelCase identifier
        m = re.search(r"\b([A-Z][A-Za-z0-9]+)\b", desc)
        name = m.group(1).lower() if m else ""
    methods = set(_feature(desc, "methods").lower().split())
    subclasses = int(_feature(desc, "subclasses") or 0)
    abstract = _feature(desc, "abstract") == "yes"
    injects = [x for x in _feature(desc, "injects").split() if x != "none"]
    touches = _feature(desc, "touches")
    instantiates = int(_feature(desc, "instantiates") or 0)
    if re.search(r"(repository|repo|dao|store)$", name) and len(methods & {"get", "add", "save", "find", "find_all",
                                                                          "delete", "remove", "list", "update",
                                                                          "find_by_id", "all"}) >= 2:
        return "repository", 0.85 if "database" in touches else 0.7
    if abstract and subclasses >= 2:
        return "strategy", 0.8
    if re.search(r"(adapter|wrapper|gateway)$", name):
        return "adapter", 0.8 if touches not in ("", "none") else 0.65
    if re.search(r"client$", name) and touches in ("http", "queue", "cache"):
        return "adapter", 0.6
    if re.search(r"factory$", name) or (instantiates >= 2 and re.search(r"^(create|make|build)", name)):
        return "factory", 0.7
    if re.search(r"builder$", name):
        return "builder", 0.7
    if "_instance" in methods or "get_instance" in methods or "instance" in methods:
        return "singleton", 0.6
    if methods & {"subscribe", "notify", "add_listener", "emit", "on", "attach", "detach", "publish"}:
        return "observer", 0.6
    if re.search(r"(facade|service|manager)$", name) and len(injects) >= 3:
        return "facade", 0.5
    if injects:
        return "dependency_injection", 0.65
    return None


TASKS: dict[str, Task] = {
    "resource": Task(
        "resource",
        {
            "database": "database query sql orm table record insert select persistence",
            "http": "http request rest api client url fetch get post web service",
            "queue": "message queue broker publish subscribe event stream topic worker",
            "cache": "cache key value store redis memcached ttl",
            "filesystem": "file path read write disk directory open save load",
            "subprocess": "process shell command execute spawn system",
            "compute": "pure computation math transformation parsing formatting no side effects",
        },
        {
            "database": ["sqlalchemy session query", "psycopg2 cursor execute", "pymongo collection find",
                         "prisma user findMany", "django objects filter", "peewee model select", "duckdb sql"],
            "http": ["requests get", "httpx AsyncClient post", "axios get", "urllib request urlopen",
                     "boto3 client invoke", "grpc channel stub", "openai chat completions create"],
            "queue": ["kafka producer send", "pika channel basic_publish", "celery task delay", "sqs send_message",
                      "nats publish", "rq queue enqueue"],
            "cache": ["redis set", "redis get", "memcache client get", "lru cache store"],
            "filesystem": ["pathlib path read_text", "shutil copy", "os walk", "glob glob", "tarfile open",
                           "zipfile ZipFile", "yaml safe_load file"],
            "subprocess": ["subprocess run", "os system", "child_process exec", "shlex split command"],
            "compute": ["math sqrt", "json dumps", "re compile", "itertools chain", "numpy array mean",
                        "datetime now", "hashlib sha256", "logging getLogger", "collections Counter"],
        },
        _resource_rules,
    ),
    "layer": Task(
        "layer",
        {
            "interface": "entry point http routes api handlers cli commands controllers user facing interface",
            "application": "application services use cases orchestration workflow coordinating domain operations",
            "domain": "core domain model business rules entities algorithms engine logic",
            "persistence": "data access repository database storage index persistence queries",
            "integration": "adapters clients for external systems third party providers gateways",
            "infrastructure": "configuration settings logging bootstrapping middleware deployment",
            "shared": "generic utilities helpers common shared functions",
            "test": "tests fixtures assertions test cases",
        },
        {
            "interface": ["routes register endpoint request response", "cli main argparse command",
                          "views render template", "handler controller"],
            "application": ["service orchestrates repository and gateway", "use case workflow process order"],
            "domain": ["entity value object business rule", "algorithm ranking scoring engine"],
            "persistence": ["repository save find by id sql", "store index storage persist"],
            "integration": ["adapter client wraps external api", "provider gateway sdk"],
            "infrastructure": ["settings load environment config", "logging setup middleware"],
            "shared": ["utils string helpers", "common helpers formatting"],
            "test": ["test case assert fixture", "pytest test function"],
        },
        _layer_rules,
    ),
    "doc_intent": Task(
        "doc_intent",
        {
            "architecture": "architecture overview components modules structure design data flow",
            "decision": "architecture decision rationale trade-offs alternatives considered",
            "api": "api reference endpoints usage parameters responses",
            "setup": "installation setup getting started configuration build deploy",
            "requirement": "requirements constraints acceptance criteria specification",
            "operations": "operations monitoring troubleshooting runbook security",
            "other": "general notes changelog license contributing miscellaneous",
        },
        {
            "architecture": ["the system is composed of three layers", "component diagram and data flow"],
            "decision": ["we decided to use sqlite because", "alternatives considered postgres"],
            "api": ["GET /items returns a list", "call search(query) to retrieve results"],
            "setup": ["pip install the package", "run make build then start the server"],
            "requirement": ["the service must respond within 100ms", "must preserve backwards compatibility"],
            "operations": ["check the logs when the job fails", "rotate keys every 90 days"],
            "other": ["licensed under MIT", "thanks to all contributors", "changelog version 1.2"],
        },
        _doc_rules,
        abstain="other",
    ),
    "design_pattern": Task(
        "design_pattern",
        {
            "strategy": "abstract interface with several interchangeable implementations selected at runtime",
            "adapter": "wraps an external api or incompatible interface and translates calls adapter client",
            "repository": "collection like data access object get add find save delete backed by database storage",
            "factory": "creates and returns instances of other classes build create make",
            "builder": "step by step construction of a complex object with chained setters build",
            "singleton": "single shared instance get_instance class level instance",
            "observer": "subscribers listeners notified of events subscribe notify emit",
            "facade": "simplified entry point coordinating several subsystems services",
            "dependency_injection": "collaborators passed into the constructor instead of created internally",
            "none": "plain class data holder utility with no recognizable design pattern",
        },
        {
            "strategy": ["class Ranker methods: score subclasses: 3 abstract: yes",
                         "class PaymentMethod methods: pay subclasses: 2 abstract: yes"],
            "adapter": ["class EmbeddingAdapter methods: embed touches: http",
                        "class S3StorageAdapter methods: put get touches: http"],
            "repository": ["class UserRepository methods: get add find_all delete touches: database",
                           "class OrderStore methods: save find_by_id touches: database"],
            "factory": ["class ConnectionFactory methods: create instantiates: 3",
                        "function make_parser instantiates: 2"],
            "builder": ["class QueryBuilder methods: where order_by limit build"],
            "singleton": ["class Config methods: get_instance _instance"],
            "observer": ["class EventBus methods: subscribe publish notify"],
            "facade": ["class CheckoutService injects: Cart Payment Shipping Inventory"],
            "dependency_injection": ["class SearchService injects: IndexRepository Ranker methods: search"],
            "none": ["class Point methods: x y", "class Settings methods: none touches: none"],
        },
        _pattern_rules,
        abstain="none",
    ),
    "note_intent": Task(
        "note_intent",
        {
            "rationale": "explains why the code is written this way design choice reason",
            "todo": "unfinished work todo fixme to be done later",
            "warning": "caution hack workaround danger security pitfall",
        },
        {
            "rationale": ["we use a heap here because it keeps the top k cheap", "chosen for determinism"],
            "todo": ["todo handle pagination", "fixme this breaks on empty input"],
            "warning": ["hack around a library bug", "do not call this from the request thread"],
        },
        _note_rules,
    ),
}


# ------------------------------------------------------------------------------- models

class NaiveBayes:
    """Multinomial naive Bayes over hashed text features (pure Python, trains in milliseconds)."""

    def __init__(self, examples: dict[str, list[str]], featurize: Callable[[str], list[str]]):
        self.featurize = featurize
        self.counts: dict[str, dict[str, float]] = {}
        self.totals: dict[str, float] = {}
        vocab: set[str] = set()
        for label, texts in examples.items():
            c: dict[str, float] = {}
            for t in texts:
                for f in featurize(t):
                    c[f] = c.get(f, 0.0) + 1.0
                    vocab.add(f)
            self.counts[label] = c
            self.totals[label] = sum(c.values())
        self.v = max(len(vocab), 1)

    def predict(self, text: str) -> list[tuple[str, float]]:
        feats = self.featurize(text) or ["<empty>"]
        scores = {}
        for label, c in self.counts.items():
            denom = self.totals[label] + self.v
            ll = sum(math.log((c.get(f, 0.0) + 1.0) / denom) for f in feats)
            scores[label] = ll / len(feats) * 4.0   # per-feature likelihood, tempered for calibration
        return _softmax(scores)


class PrototypeClassifier:
    """Nearest-prototype classification in a (neural) embedding space."""

    def __init__(self, examples: dict[str, list[str]], embedder: Embedder):
        self.embedder = embedder
        self.protos: dict[str, list[float]] = {}
        for label, texts in examples.items():
            vecs = embedder.embed(texts)
            self.protos[label] = normalize([sum(col) / len(vecs) for col in zip(*vecs)])

    def predict(self, text: str) -> list[tuple[str, float]]:
        v = self.embedder.embed([text])[0]
        return _softmax({label: cosine(v, p) / 0.05 for label, p in self.protos.items()})


def _softmax(scores: dict[str, float]) -> list[tuple[str, float]]:
    m = max(scores.values())
    exps = {k: math.exp(v - m) for k, v in scores.items()}
    z = sum(exps.values())
    return sorted(((k, e / z) for k, e in exps.items()), key=lambda t: -t[1])


@dataclass
class Classification:
    label: str
    confidence: float
    tier: str
    model: str | None = None
    agreement: bool | None = None
    alternatives: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"label": self.label, "confidence": round(self.confidence, 3), "tier": self.tier,
                "model": self.model, "agreement": self.agreement, "alternatives": self.alternatives}


class HybridClassifier:
    def __init__(self, embedder: Embedder | None = None, mode: str = "hybrid", training_dir: Path | None = None,
                 cache: Any = None):
        """mode: 'hybrid' (rules + model), 'heuristic' (rules only) or 'model' (model only)."""
        self.mode = mode
        self.embedder = embedder
        self.training_dir = training_dir
        self.cache = cache   # Store with cached_classification/cache_classification
        self._models: dict[str, Any] = {}
        self._hasher = HashingEmbedder()

    @property
    def model_tier(self) -> str:
        return "neural" if self.embedder is not None and self.embedder.neural else "builtin-nb"

    @property
    def model_name(self) -> str:
        return self.embedder.name if self.embedder is not None and self.embedder.neural else "naive-bayes:hash"

    def examples(self, task: Task) -> dict[str, list[str]]:
        ex = {label: [desc] + task.seeds.get(label, []) for label, desc in task.labels.items()}
        if self.training_dir:
            f = self.training_dir / f"{task.name}.jsonl"
            if f.exists():
                for line in f.read_text().splitlines():
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if rec.get("label") in ex and rec.get("text"):
                        ex[rec["label"]].append(rec["text"])
        return ex

    def _model(self, task: Task) -> Any:
        if task.name not in self._models:
            ex = self.examples(task)
            if self.embedder is not None and self.embedder.neural:
                self._models[task.name] = PrototypeClassifier(ex, self.embedder)
            else:
                self._models[task.name] = NaiveBayes(ex, self._hasher.features)
        return self._models[task.name]

    def classify(self, task_name: str, text: str) -> Classification:
        task = TASKS[task_name]
        tier_key = f"{self.mode}:{self.model_tier}"
        key = hashlib.sha1(text.encode()).hexdigest()
        if self.cache is not None:
            hit = self.cache.cached_classification(task_name, key, tier_key)
            if hit and hit["model"] == self.model_name:
                d = hit["detail"]
                return Classification(hit["label"], hit["confidence"], d.get("tier", tier_key), hit["model"],
                                      d.get("agreement"), d.get("alternatives", []))
        result = self._classify(task, text)
        if self.cache is not None:
            self.cache.cache_classification(task_name, key, tier_key, result.label, result.confidence,
                                            self.model_name, {"tier": result.tier, "agreement": result.agreement,
                                                              "alternatives": result.alternatives})
        return result

    def _classify(self, task: Task, text: str) -> Classification:
        rule = task.rules(text)
        if self.mode == "heuristic":
            if rule:
                return Classification(rule[0], rule[1], "heuristic", "rules")
            return Classification(task.abstain, 0.0, "heuristic", "rules")
        ranked = self._model(task).predict(text)
        m_label, m_conf = ranked[0]
        tier = self.model_tier
        if self.mode == "model" or rule is None:
            return Classification(m_label, m_conf, tier, self.model_name, None,
                                  [{"label": l, "confidence": round(c, 3)} for l, c in ranked[1:3]])
        r_label, r_conf = rule
        if r_label == m_label:
            return Classification(r_label, 1 - (1 - r_conf) * (1 - m_conf), f"heuristic+{tier}", self.model_name, True)
        model_alt = {"label": m_label, "confidence": round(m_conf, 3), "tier": tier}
        rule_alt = {"label": r_label, "confidence": round(r_conf, 3), "tier": "heuristic"}
        # Rules encode precise, explainable signals; the model must be clearly more confident to override.
        if m_conf > r_conf + 0.15:
            return Classification(m_label, m_conf * 0.9, f"{tier}>heuristic", self.model_name, False, [rule_alt])
        return Classification(r_label, r_conf * 0.9, f"heuristic>{tier}", "rules", False, [model_alt])

    def add_example(self, task_name: str, text: str, label: str) -> Path:
        if task_name not in TASKS or label not in TASKS[task_name].labels:
            raise ValueError(f"unknown task/label: {task_name}/{label}")
        if not self.training_dir:
            raise ValueError("no training directory configured")
        self.training_dir.mkdir(parents=True, exist_ok=True)
        path = self.training_dir / f"{task_name}.jsonl"
        with path.open("a") as fh:
            fh.write(json.dumps({"text": text, "label": label}) + "\n")
        self._models.pop(task_name, None)
        return path
