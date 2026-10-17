"""Machine-learning tier: embeddings for semantic retrieval and two-tier classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from promapkin.ml.classify import TASKS, HybridClassifier
from promapkin.ml.embedders import Embedder, HashingEmbedder
from promapkin.ml.models import Decision, ModelManager

if TYPE_CHECKING:
    from promapkin.config import Workspace

__all__ = ["MLRuntime", "runtime", "TASKS", "HybridClassifier", "HashingEmbedder", "ModelManager"]


@dataclass
class MLRuntime:
    decision: Decision
    classifier: HybridClassifier
    hasher: HashingEmbedder

    @property
    def embedder(self) -> Embedder:
        return self.decision.embedder

    def embedders(self) -> list[Embedder]:
        """Embedders whose vectors are maintained: always the built-in one, plus a neural one if present."""
        return [self.hasher] + ([self.decision.embedder] if self.decision.embedder.neural else [])

    def describe(self) -> dict:
        return {"tier": self.decision.tier, "embedder": self.decision.embedder.name, "reason": self.decision.reason,
                "classifier_mode": self.classifier.mode, "classifier_model": self.classifier.model_name,
                "download_available": self.decision.download_available}


def manager(ws: "Workspace") -> ModelManager:
    cfg = ws.config

    def remember(answer: str) -> None:
        cfg.ml_consent = answer
        if ws.initialized:
            cfg.save(ws.config_path)

    return ModelManager(cfg.ml_mode, cfg.ml_backend, cfg.ml_model, cfg.ml_consent, on_consent=remember)


def runtime(ws: "Workspace", interactive: bool = False, ask: Callable[[str], bool] | None = None) -> MLRuntime:
    cached = getattr(ws, "_ml_runtime", None)
    if cached is not None:
        return cached
    decision = manager(ws).resolve(interactive=interactive, ask=ask)
    # ml_backend="heuristic" disables every statistical model and uses only the rules.
    mode = "heuristic" if ws.config.ml_backend == "heuristic" else "hybrid"
    clf = HybridClassifier(decision.embedder, mode=mode, training_dir=ws.dir / "training", cache=ws.store)
    rt = MLRuntime(decision, clf, decision.embedder if not decision.embedder.neural else HashingEmbedder())
    ws._ml_runtime = rt  # type: ignore[attr-defined]
    return rt
