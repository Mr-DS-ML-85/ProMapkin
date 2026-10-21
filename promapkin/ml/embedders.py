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
from typing import Any, Iterable, Sequence

from promapkin.store import split_identifier

try:  # numpy speeds up similarity search but is optional
    import numpy as _np
except ModuleNotFoundError:  # pragma: no cover - depends on environment
    _np = None

Vector = Sequence[float]


# Large indexes store vectors as float16 behind this header: half the disk and I/O (a kernel-sized index has two
# million vectors), with cosine error around 1e-3. Plain float32 blobs (no header) remain the default and both
# formats are read everywhere. As a float32 the header would be ~4e-38, a value no normalized vector holds.
HALF_MAGIC = b"PMh\x01"
COMPACT_MIN_VECTORS = 200_000


def pack(vec: Vector, half: bool = False) -> bytes:
    if half and _np is not None:
        return HALF_MAGIC + _np.asarray(vec, dtype=_np.float16).tobytes()
    return array("f", vec).tobytes()


def is_half(blob: bytes) -> bool:
    return blob[:4] == HALF_MAGIC and len(blob) % 2 == 0


def blob_dim(blob: bytes) -> int:
    return (len(blob) - 4) // 2 if is_half(blob) else len(blob) // 4


def unpack(blob: bytes) -> list[float]:
    if is_half(blob):
        if _np is not None:
            return _np.frombuffer(blob, dtype=_np.float16, offset=4).astype(_np.float32).tolist()
        import struct
        return list(struct.unpack(f"={(len(blob) - 4) // 2}e", blob[4:]))
    a = array("f")
    a.frombytes(blob)
    return a.tolist()


def decode(blob: bytes) -> Any:
    """A stored vector as a float32 numpy array (numpy required)."""
    if is_half(blob):
        return _np.frombuffer(blob, dtype=_np.float16, offset=4).astype(_np.float32)
    return _np.frombuffer(blob, dtype=_np.float32)


def decode_many(blobs: list[bytes]) -> Any:
    """Equal-dimension stored vectors as one float32 matrix (numpy required)."""
    if blobs and not any(is_half(b) for b in blobs):
        return _np.frombuffer(b"".join(blobs), dtype=_np.float32).reshape(len(blobs), -1)
    if blobs and all(is_half(b) for b in blobs):
        return _np.frombuffer(b"".join(b[4:] for b in blobs), dtype=_np.float16).reshape(len(blobs), -1).astype(
            _np.float32)
    return _np.stack([decode(b) for b in blobs]) if blobs else _np.zeros((0, 0), dtype=_np.float32)


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
        q = _np.asarray(query, dtype=_np.float32)
        items = [(i, v) for i, v in items if blob_dim(v) == q.shape[0]]
        if not items:
            return []
        mat = decode_many([v for _, v in items])
        scores = mat @ q
        idx = _np.argsort(-scores)[:k]
        return [(items[i][0], float(scores[i])) for i in idx]
    scored = [(id, cosine(query, unpack(v))) for id, v in items]
    scored.sort(key=lambda t: -t[1])
    return scored[:k]


_HASH_CACHE: dict[str, int] = {}


def store_top_k(store: Any, model: str, query: Vector, k: int, chunk: int = 20000) -> list[tuple[str, float]]:
    """Cosine top-k over every stored vector of `model`, streamed in chunks: memory is O(chunk + k) however many
    vectors the index holds (top_k over store.embeddings() would load them all at once)."""
    import heapq
    rows = store.conn.execute("SELECT e.id, e.vec FROM embeddings e JOIN nodes n ON n.id=e.id WHERE e.model=?",
                              (model,))
    if _np is None:
        def scored() -> Iterable[tuple[float, str]]:
            for nid, blob in rows:
                yield cosine(query, unpack(blob)), nid
        return [(nid, s) for s, nid in sorted(heapq.nlargest(k, scored()), key=lambda t: (-t[0], t[1]))]
    q = _np.asarray(query, dtype=_np.float32)
    best: list[tuple[float, str]] = []
    while True:
        fetched = rows.fetchmany(chunk)
        if not fetched:
            break
        batch = [r for r in fetched if blob_dim(r[1]) == q.shape[0]]   # another dimension cannot match
        if not batch:
            continue
        mat = decode_many([r[1] for r in batch])
        scores = mat @ q
        top = _np.argpartition(-scores, min(k, len(batch)) - 1)[:k] if len(batch) > k else range(len(batch))
        best = heapq.nlargest(k, best + [(float(scores[i]), batch[i][0]) for i in top])
    return [(nid, s) for s, nid in sorted(best, key=lambda t: (-t[0], t[1]))]   # ties in id order


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
