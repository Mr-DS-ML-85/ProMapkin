"""Ranking strategies used to order candidate documents."""
from abc import ABC, abstractmethod


class Ranker(ABC):
    """Strategy interface for ranking candidates."""

    @abstractmethod
    def score(self, query: str, document: dict) -> float:
        ...


class BM25Ranker(Ranker):
    """Lexical ranking with BM25."""

    def score(self, query: str, document: dict) -> float:
        terms = query.lower().split()
        text = document.get("text", "").lower()
        return float(sum(text.count(t) for t in terms))


class VectorRanker(Ranker):
    """Ranks by cosine similarity between embeddings."""

    def __init__(self, embedder):
        self.embedder = embedder

    def score(self, query: str, document: dict) -> float:
        q = self.embedder.embed(query)
        d = document.get("vector") or self.embedder.embed(document.get("text", ""))
        return cosine_similarity(q, d)


def cosine_similarity(a, b) -> float:
    num = sum(x * y for x, y in zip(a, b))
    den = (sum(x * x for x in a) ** 0.5) * (sum(y * y for y in b) ** 0.5)
    return num / den if den else 0.0


def select_candidates(documents: list, limit: int) -> list:
    # We deliberately cap candidates before ranking because ranking is the expensive step.
    return documents[: limit * 4]
