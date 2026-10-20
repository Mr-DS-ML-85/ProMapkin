"""Text embedders, from a dependency-free hashing model to small neural models.

* ``builtin-hash``: feature-hashed word/identifier/char-trigram vectors (pure Python,
  always available, no download).
* ``fastembed``: ONNX small embedding models (default ``BAAI/bge-small-en-v1.5``).
* ``sentence-transformers``: PyTorch small models (default ``all-MiniLM-L6-v2``).
"""

from __future__ import annotations

import math
import zlib
import re
from array import array
from typing import Iterable, Sequence

from promapkin.store import split_identifier

try:  # numpy speeds up similarity search but is optional
    import numpy as _np
except ModuleNotFoundError:  # pragma: no cover - depends on environment
    _np = None

Vector = Sequence[float]


def pack(vec: Vector) -> bytes:
    return array("f", vec).tobytes()


def unpack(blob: bytes) -> list[float]:
    a = array("f")
    a.frombytes(blob)
    return a.tolist()


def normalize(vec: Iterable[float]) -> list[float]:
    v = list(vec)
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def cosine(a: Vector, b: Vector) -> float:
    return sum(x * y for x, y in zip(a, b))


def top_k(query: Vector, items: list[tuple[str, bytes]], k: int) -> list[tuple[str, float]]:
    """Cosine top-k over normalized stored vectors."""
    if not items:
        return []
    if _np is not None:
        mat = _np.frombuffer(b"".join(v for _, v in items), dtype=_np.float32).reshape(len(items), -1)
        q = _np.asarray(query, dtype=_np.float32)
        if mat.shape[1] != q.shape[0]:
            return []
        scores = mat @ q
        idx = _np.argsort(-scores)[:k]
        return [(items[i][0], float(scores[i])) for i in idx]
    scored = [(id, cosine(query, unpack(v))) for id, v in items]
    scored.sort(key=lambda t: -t[1])
    return scored[:k]


_HASH_CACHE: dict[str, int] = {}


def _hash(feature: str) -> int:
    """Stable 64-bit feature hash (crc32 of the feature and of its reverse; C speed, memoized)."""
    h = _HASH_CACHE.get(feature)
    if h is None:
        b = feature.encode()
        h = (zlib.crc32(b) << 32) | zlib.crc32(b[::-1])
        if len(_HASH_CACHE) < 500_000:
            _HASH_CACHE[feature] = h
    return h


class Embedder:
    name = "abstract"
    dim = 0
    neural = False

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


class HashingEmbedder(Embedder):
    """Feature hashing over words, split identifiers and character trigrams, sublinear TF, L2 normalized."""

    name = "builtin-hash-512-v2"
    neural = False

    def __init__(self, dim: int = 512):
        self.dim = dim

    def features(self, text: str) -> list[str]:
        feats: list[str] = []
        for word in re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+", text):
            low = word.lower()
            feats.append("w:" + low)
            parts = split_identifier(word)
            if len(parts) > 1:
                feats.extend("w:" + p for p in parts)
            padded = f"#{low}#"
            feats.extend("c:" + padded[i:i + 3] for i in range(len(padded) - 2))
        return feats

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            counts: dict[int, float] = {}
            for f in self.features(text):
                h = _hash(f)
                idx, sign = h % self.dim, (1.0 if (h >> 32) & 1 else -1.0)
                weight = 1.0 if f.startswith("w:") else 0.35
                counts[idx] = counts.get(idx, 0.0) + sign * weight
            vec = [0.0] * self.dim
            for idx, val in counts.items():
                vec[idx] = math.copysign(1 + math.log(abs(val)), val) if val else 0.0
            out.append(normalize(vec))
        return out


class FastEmbedEmbedder(Embedder):
    neural = True
    default_model = "BAAI/bge-small-en-v1.5"

    def __init__(self, model: str | None, cache_dir: str):
        from fastembed import TextEmbedding  # imported lazily: optional dependency

        self.model_name = model or self.default_model
        self._model = TextEmbedding(model_name=self.model_name, cache_dir=cache_dir)
        self.name = f"fastembed:{self.model_name}"
        self.dim = len(next(iter(self._model.embed(["probe"]))))

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [normalize(v.tolist()) for v in self._model.embed(texts)]


class SentenceTransformerEmbedder(Embedder):
    neural = True
    default_model = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, model: str | None, cache_dir: str):
        from sentence_transformers import SentenceTransformer  # optional dependency

        self.model_name = model or self.default_model
        self._model = SentenceTransformer(self.model_name, cache_folder=cache_dir)
        self.name = f"st:{self.model_name}"
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [normalize(v) for v in self._model.encode(texts, normalize_embeddings=True).tolist()]
