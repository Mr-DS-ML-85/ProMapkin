"""Graph machine learning: structure-aware node embeddings, similarity edges, clustering and
ML disambiguation of ambiguous references.

With ML enabled, node text embeddings (neural or built-in) are smoothed over the code graph
with a parameter-free message-passing step (a GNN-style propagation), so a symbol's vector
reflects both what it says and what it is connected to. With ML disabled, each function
falls back to a deterministic heuristic with static confidence values.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any, Iterable

from promapkin import evidence as ev
from promapkin.ml.embedders import COMPACT_MIN_VECTORS, HALF_MAGIC, _np, blob_dim, decode, normalize, pack, top_k, \
    unpack
from promapkin.store import Store, split_identifier

STRUCTURAL_EDGES = ["calls", "imports", "inherits", "implements", "depends_on", "instantiates", "contains",
                    "defines", "routes_to", "overrides", "references"]
SIMILAR_KINDS = ("class", "function", "method", "module")
STATIC_SIMILARITY_CONFIDENCE = 0.4
STATIC_DISAMBIGUATION_CONFIDENCE = 0.45


def graph_model_name(base: str) -> str:
    return f"graph:{base}"


PYTHON_MAX_NODES = 60_000      # without numpy, larger graphs skip propagation and vector similarity
EXACT_MAX_NODES = 20_000       # above this, similarity search uses locality-sensitive hashing
BLOCK_FLOATS = 16_000_000      # ~64 MB of float32 per block of work
RAM_STATE_BYTES = 512 << 20    # propagation state kept in float32 up to this size, float16 above


def propagate(store: Store, base_model: str, alpha: float = 0.5, iterations: int = 2) -> int:
    """h_{t+1} = normalize((1-alpha) * h_0 + alpha * mean(h_t over neighbours)). Stored as graph:<model>.

    With numpy the matrices live in temporary files and are processed in row blocks, so memory does not grow
    with the repository (the Linux kernel has about a million embedded symbols)."""
    name = graph_model_name(base_model)
    if _np is None:
        n = store.conn.execute("SELECT count(*) FROM embeddings WHERE model=?", (base_model,)).fetchone()[0]
        if n > PYTHON_MAX_NODES:
            raise RuntimeError(f"{n} embedded nodes: graph propagation at this size needs numpy "
                               "(pip install numpy)")
        base = {nid: unpack(v) for nid, v in store.embeddings(base_model)}
        if not base:
            return 0
        adj: dict[str, set[str]] = defaultdict(set)
        for src, dst in _structural_pairs(store):
            if src in base and dst in base:
                adj[src].add(dst)
                adj[dst].add(src)
        h = _propagate_python(base, adj, alpha, iterations)
        store.conn.execute("DELETE FROM embeddings WHERE model=?", (name,))
        store.conn.executemany("INSERT OR REPLACE INTO embeddings(id, model, sha, vec) VALUES (?,?,?,?)",
                               ((nid, name, "propagated", pack(vec)) for nid, vec in h.items()))
        return len(h)
    with Matrix(store, base_model) as m:
        if not m.ids:
            store.conn.execute("DELETE FROM embeddings WHERE model=?", (name,))
            return 0
        indptr, indices = _csr(store, m.index)
        n, dim = len(m.ids), m.dim
        h0 = m.vecs
        # Neighbours are read at random, so the current state lives in memory: float32 while that is small (exact
        # results), float16 for kernel-sized graphs (half the memory; aggregation still runs in float32). The
        # inputs and outputs are streamed through the memory-mapped files block by block.
        cur = m.in_memory()
        nxt = m.scratch()
        threads = _threads()

        def step(block: tuple[int, int]) -> None:
            a, b = block
            lo, hi = indptr[a], indptr[b]
            deg = _np.diff(indptr[a:b + 1]).astype(_np.float32)
            mixed = _np.array(h0[a:b], dtype=_np.float32)
            _drop(h0, a, b)
            has = deg > 0
            if hi > lo:
                gathered = cur[indices[lo:hi]].astype(_np.float32, copy=False)
                starts = (indptr[a:b][has] - lo)
                agg = _np.add.reduceat(gathered, starts, axis=0)
                mixed[has] = (1 - alpha) * mixed[has] + alpha * agg / deg[has, None]
            norms = _np.linalg.norm(mixed, axis=1, keepdims=True)
            nxt[a:b] = mixed / _np.where(norms == 0, 1, norms)
            _drop(nxt, a, b)

        # Blocks are independent within an iteration and write disjoint rows; numpy releases the GIL in the
        # gathers and reductions, so threads scale (3.3x on 4 cores) with the same total block memory.
        blocks = list(_row_blocks(indptr, dim, BLOCK_FLOATS // threads))
        for it in range(iterations):
            if threads > 1:
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor(threads) as pool:
                    list(pool.map(step, blocks))
            else:
                for block in blocks:
                    step(block)
            if it < iterations - 1:
                for a in range(0, n, 20000):
                    cur[a:a + 20000] = nxt[a:a + 20000]
                    _drop(nxt, a, a + 20000)
        cur = None       # free the in-memory state before the vectors are written
        # Rewritten in place (same-size rows): 3x faster than delete + insert on a million vectors. Vectors of
        # deleted nodes were already pruned by the embedding pass.
        half = n >= COMPACT_MIN_VECTORS
        for a in range(0, n, 5000):
            block = _np.asarray(nxt[a:a + 5000], dtype=_np.float16 if half else _np.float32)
            _drop(nxt, a, a + 5000)
            prefix = HALF_MAGIC if half else b""
            store.conn.executemany("INSERT INTO embeddings(id, model, sha, vec) VALUES (?,?,?,?) "
                                   "ON CONFLICT(id, model) DO UPDATE SET sha=excluded.sha, vec=excluded.vec",
                                   ((m.ids[a + i], name, "propagated", prefix + block[i].tobytes())
                                    for i in range(block.shape[0])))
        return n


class Matrix:
    """The embeddings of one model as a float32 matrix in a temporary file (numpy memmap): passes over millions
    of vectors keep only the ids in memory. `keep` restricts it to a set of node ids."""

    def __init__(self, store: Store, model: str, keep: Any = None):
        import tempfile
        self._dir = tempfile.TemporaryDirectory(prefix="vectors-", dir=_tmp_root(store))
        self.ids: list[str] = []
        self.index: dict[str, int] = {}
        self.dim = 0
        self.vecs: Any = None
        self._n = 0
        total = store.conn.execute("SELECT count(*) FROM embeddings e JOIN nodes n ON n.id=e.id WHERE e.model=?",
                                   (model,)).fetchone()[0]
        if not total:
            return
        rows = store.conn.execute("SELECT e.id, e.vec FROM embeddings e JOIN nodes n ON n.id=e.id WHERE e.model=? "
                                  "ORDER BY e.id", (model,))
        for nid, blob in rows:
            if keep is not None and nid not in keep:
                continue
            if self.vecs is None:
                self.dim = blob_dim(blob)
                self.vecs = self._file("vecs", total)
            if blob_dim(blob) != self.dim:
                continue      # a vector from another dimension (stale model change): skip it
            self.vecs[len(self.ids)] = decode(blob)
            self.index[nid] = len(self.ids)
            self.ids.append(nid)
            if len(self.ids) % 20000 == 0:
                _drop(self.vecs, len(self.ids) - 20000, len(self.ids))
        if self.vecs is not None:
            self.vecs = self.vecs[:len(self.ids)]

    def _file(self, name: str, rows: int) -> Any:
        self._n += 1
        return _np.memmap(f"{self._dir.name}/{name}-{self._n}.f32", mode="w+", dtype=_np.float32,
                          shape=(max(rows, 1), self.dim))

    def in_memory(self) -> Any:
        """The matrix as an in-memory array for random access: float32 up to RAM_STATE_BYTES, else float16."""
        n = len(self.ids)
        out = _np.empty((n, self.dim), dtype=_np.float32 if n * self.dim * 4 <= RAM_STATE_BYTES else _np.float16)
        for a in range(0, n, 20000):
            out[a:a + 20000] = self.vecs[a:a + 20000]
            _drop(self.vecs, a, a + 20000)
        return out

    def scratch(self) -> Any:
        return self._file("scratch", len(self.ids))[:len(self.ids)]

    def __enter__(self) -> "Matrix":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.vecs = None
        self._dir.cleanup()


def _drop(arr: Any, a: int, b: int) -> None:
    """Release rows a..b of a memory-mapped matrix from this process (written data stays in the file and the
    page cache): a sequential pass over a 2 GB file then never holds more than one block resident."""
    import mmap
    mm = getattr(arr, "_mmap", None)
    if mm is None or not hasattr(mm, "madvise"):
        return
    row = arr.shape[1] * arr.itemsize
    base = getattr(arr, "offset", 0) or 0
    start = -(-(base + a * row) // mmap.PAGESIZE) * mmap.PAGESIZE
    end = (base + min(b, arr.shape[0]) * row) // mmap.PAGESIZE * mmap.PAGESIZE
    if end > start:
        try:
            mm.madvise(mmap.MADV_DONTNEED, start, end - start)
        except (OSError, ValueError, AttributeError):
            pass


def _tmp_root(store: Store) -> str | None:
    from pathlib import Path
    parent = Path(store.path).parent
    return str(parent) if store.path != ":memory:" and parent.is_dir() else None


def _structural_pairs(store: Store) -> Any:
    marks = ",".join("?" * len(STRUCTURAL_EDGES))
    return store.conn.execute(f"SELECT src, dst FROM edges WHERE type IN ({marks})", STRUCTURAL_EDGES)


def _csr(store: Store, index: dict[str, int]) -> tuple[Any, Any]:
    """Undirected, de-duplicated adjacency of the structural edges as CSR arrays (indptr, indices)."""
    from array import array
    a, b = array("q"), array("q")
    for src, dst in _structural_pairs(store):
        i, j = index.get(src), index.get(dst)
        if i is not None and j is not None and i != j:
            a.append(i)
            b.append(j)
    n = len(index)
    src = _np.frombuffer(a, dtype=_np.int64) if a else _np.zeros(0, dtype=_np.int64)
    dst = _np.frombuffer(b, dtype=_np.int64) if b else _np.zeros(0, dtype=_np.int64)
    keys = _np.unique(_np.concatenate([src * n + dst, dst * n + src]))
    rows, cols = keys // n, (keys % n).astype(_np.int32)
    indptr = _np.zeros(n + 1, dtype=_np.int64)
    _np.cumsum(_np.bincount(rows, minlength=n), out=indptr[1:])
    return indptr, cols


def _threads() -> int:
    import os
    env = os.environ.get("PROMAPKIN_WORKERS")
    want = int(env) if env and env.isdigit() else 0
    return max(1, want if want > 0 else min(8, os.cpu_count() or 1))


def _row_blocks(indptr: Any, dim: int, floats: int | None = None) -> Any:
    """Row ranges whose gathered neighbour vectors stay within `floats` (default BLOCK_FLOATS)."""
    n = len(indptr) - 1
    budget = max(1, (floats or BLOCK_FLOATS) // max(dim, 1))
    a = 0
    while a < n:
        b = int(_np.searchsorted(indptr, indptr[a] + budget, side="right")) - 1
        b = min(n, max(b, a + 1), a + budget)
        yield a, b
        a = b


def _propagate_python(base: dict[str, list[float]], adj: dict[str, set[str]], alpha: float, iterations: int
                      ) -> dict[str, list[float]]:
    h = dict(base)
    for _ in range(iterations):
        nxt = {}
        for nid, h0 in base.items():
            nbrs = adj.get(nid)
            if not nbrs:
                nxt[nid] = h0
                continue
            dim = len(h0)
            mean = [0.0] * dim
            for n in nbrs:
                v = h[n]
                for i in range(dim):
                    mean[i] += v[i]
            k = len(nbrs)
            nxt[nid] = normalize([(1 - alpha) * h0[i] + alpha * mean[i] / k for i in range(dim)])
        h = nxt
    return h


def similarity_edges(store: Store, model: str | None, k: int = 3, threshold: float = 0.82,
                     revision: str | None = None) -> int:
    """`similar_to` edges between semantically related symbols in different modules.

    ML: cosine similarity of graph embeddings (evidence ml_classifier, confidence = similarity); exact top-k up to
    EXACT_MAX_NODES symbols, locality-sensitive hashing above.
    Heuristic: identifier-token Jaccard overlap (evidence heuristic, static confidence)."""
    store.delete_edges("type='similar_to'")
    kinds = ",".join(f"'{x}'" for x in SIMILAR_KINDS)
    nodes = {nid: (kind, name) for nid, kind, name in store.conn.execute(
        f"SELECT id, kind, name FROM nodes WHERE kind IN ({kinds}) AND json_extract(attrs, '$.test') IS NOT 1 "
        f"AND NOT (kind = 'module' AND COALESCE(json_extract(attrs, '$.loc'), 0) < 5) ORDER BY id")}
    count = 0
    if model:
        if _np is None:
            if len(nodes) > PYTHON_MAX_NODES:
                raise RuntimeError(f"{len(nodes)} symbols: similarity search at this size needs numpy "
                                   "(pip install numpy)")
            items = [(nid, v) for nid, v in store.embeddings(model) if nid in nodes]
            pairs = _pairs_python(items, k + 1)
        else:
            with Matrix(store, model, keep=nodes) as m:
                ids = m.ids
                nbr_idx, nbr_sim = _neighbours(m, k + 1)
            pairs = _pairs(ids, nbr_idx, nbr_sim)
        for nid, other, sim in pairs:
            if sim < threshold or _same_module(nid, other) or nodes[nid][0] != nodes[other][0]:
                continue
            a, b = sorted((nid, other))
            store.add_edge(a, b, "similar_to", evidence=ev.ML_CLASSIFIER, confidence=round(sim, 3),
                           revision=revision, attrs={"method": f"cosine({model})", "similarity": round(sim, 3)})
            count += 1
        return count
    tokens = {nid: set(split_identifier(name)) - {"get", "set", "init", "self"} for nid, (_k, name) in nodes.items()}
    by_token: dict[str, list[str]] = defaultdict(list)
    for nid, toks in tokens.items():
        for t in toks:
            by_token[t].append(nid)
    # A token shared by thousands of names (`free`, `alloc` in a kernel) says nothing and makes this quadratic.
    common = {t for t, ids in by_token.items() if len(ids) > 5000}
    seen: set[tuple[str, str]] = set()
    for nid, toks in tokens.items():
        if len(toks) < 2:
            continue
        cands = Counter(o for t in toks - common for o in by_token[t] if o != nid)
        for other, shared in cands.most_common(k):
            union = len(toks | tokens[other])
            if shared / union < 0.6 or _same_module(nid, other) or nodes[nid][0] != nodes[other][0]:
                continue
            key = tuple(sorted((nid, other)))
            if key in seen:
                continue
            seen.add(key)
            store.add_edge(key[0], key[1], "similar_to", evidence=ev.HEURISTIC,
                           confidence=STATIC_SIMILARITY_CONFIDENCE, revision=revision,
                           attrs={"method": "identifier-token jaccard", "jaccard": round(shared / union, 3)})
            count += 1
    return count


def _pairs(ids: list[str], nbr_idx: Any, nbr_sim: Any) -> Any:
    """Each unordered neighbour pair once: (i, j) is reported from row i unless row j lists i too and j < i."""
    for i in range(len(ids)):
        for j, sim in zip(nbr_idx[i].tolist(), nbr_sim[i].tolist()):
            if j < 0 or j == i:
                continue
            if j < i and i in nbr_idx[j]:
                continue
            yield ids[i], ids[j], sim


def _pairs_python(items: list[tuple[str, bytes]], k: int) -> Any:
    seen: set[tuple[str, str]] = set()
    for nid, blob in items:
        for other, sim in top_k(unpack(blob), items, k):
            if other == nid:
                continue
            key = (min(nid, other), max(nid, other))
            if key not in seen:
                seen.add(key)
                yield nid, other, sim


def _neighbours(m: Matrix, k: int) -> tuple[Any, Any]:
    """(indices, similarities) of each row's top-k cosine neighbours (self included), -1 where fewer exist."""
    n = len(m.ids)
    if n == 0:
        return _np.zeros((0, k), dtype=_np.int64), _np.zeros((0, k), dtype=_np.float32)
    if n <= EXACT_MAX_NODES:
        return _exact_top_k(m.in_memory(), k)
    return _lsh_top_k(m.in_memory(), k)


def _exact_top_k(mat: Any, k: int) -> tuple[Any, Any]:
    n = mat.shape[0]
    kk = min(k, n)
    idx = _np.full((n, k), -1, dtype=_np.int64)
    sim = _np.full((n, k), -_np.inf, dtype=_np.float32)
    full = _np.asarray(mat, dtype=_np.float32)
    step = max(1, BLOCK_FLOATS // n)
    for start in range(0, n, step):
        block = full[start:start + step] @ full.T
        part = _np.argpartition(-block, kk - 1, axis=1)[:, :kk]
        vals = _np.take_along_axis(block, part, axis=1)
        order = _np.argsort(-vals, axis=1, kind="stable")
        idx[start:start + block.shape[0], :kk] = _np.take_along_axis(part, order, axis=1)
        sim[start:start + block.shape[0], :kk] = _np.take_along_axis(vals, order, axis=1)
    return idx, sim


LSH_TABLES = 12
LSH_WINDOW = 256
LSH_BITS = 20


def _lsh_top_k(mat: Any, k: int, tables: int | None = None, window: int | None = None) -> tuple[Any, Any]:
    """Approximate top-k by random-hyperplane hashing. In each of `tables` tables the vectors are sorted by their
    hash code and compared exactly within half-overlapping windows of the sorted order, so near neighbours (long
    shared code prefixes) meet even when their codes differ in a low bit. Windows that do not overlap are
    compared in one batched matrix product. Deterministic (fixed seed); time and memory are linear in the
    number of vectors."""
    tables, window = tables or LSH_TABLES, window or LSH_WINDOW
    n, dim = mat.shape
    window = min(window, n)
    rng = _np.random.default_rng(20240607)
    best_idx = _np.full((n, k), -1, dtype=_np.int64)
    best_sim = _np.full((n, k), -_np.inf, dtype=_np.float32)
    weights = (1 << _np.arange(LSH_BITS - 1, -1, -1, dtype=_np.int64))
    rows_per_block = max(1, BLOCK_FLOATS // dim)
    threads = _threads()
    group = max(1, BLOCK_FLOATS // (window * dim * threads))       # windows per batched product

    def compare(ids: Any) -> None:
        v = mat[ids.ravel()].astype(_np.float32, copy=False).reshape(ids.shape[0], window, dim)
        _merge_rows(best_idx, best_sim, ids, v @ v.transpose(0, 2, 1), k)

    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(threads) if threads > 1 else None
    for _t in range(tables):
        planes = rng.standard_normal((dim, LSH_BITS)).astype(_np.float32)
        codes = _np.empty(n, dtype=_np.int64)
        for a in range(0, n, rows_per_block):
            codes[a:a + rows_per_block] = ((_np.asarray(mat[a:a + rows_per_block]) @ planes) > 0).astype(
                _np.int64) @ weights
        order = _np.argsort(codes, kind="stable")
        for offset in (0, window // 2):
            full = (n - offset) // window
            # Windows of one offset never share a row, so groups of them can be compared in parallel.
            groups = [order[offset + g * window: offset + min(full, g + group) * window].reshape(-1, window)
                      for g in range(0, full, group)]
            if pool is not None:
                list(pool.map(compare, groups))
            else:
                for ids in groups:
                    compare(ids)
            tail = order[offset + full * window:] if offset == 0 else order[:offset]
            if len(tail) > 1:
                tail = _np.sort(tail)
                v = mat[tail].astype(_np.float32, copy=False)
                _merge_rows(best_idx, best_sim, tail[None, :], (v @ v.T)[None], k)
    if pool is not None:
        pool.shutdown()
    return best_idx, best_sim


def _merge_rows(best_idx: Any, best_sim: Any, ids: Any, sims: Any, k: int) -> None:
    """Fold each window's exact top-k (ids: windows x members, sims: windows x members x members) into the
    running per-row top-k. A row appears at most once per call."""
    w, b = ids.shape
    kk = min(k, b)
    part = _np.argpartition(sims, b - kk, axis=2)[:, :, b - kk:]     # the kk largest, without negating a copy
    cand_sim = _np.take_along_axis(sims, part, axis=2).reshape(w * b, kk)
    cand_idx = _np.take_along_axis(_np.broadcast_to(ids[:, None, :], (w, b, b)), part, axis=2).reshape(w * b, kk)
    rows = ids.reshape(-1)
    cur_idx, cur_sim = best_idx[rows], best_sim[rows]
    dup = (cand_idx[:, :, None] == cur_idx[:, None, :]).any(axis=2)
    cand_sim = _np.where(dup, -_np.inf, cand_sim)
    all_idx = _np.concatenate([cur_idx, cand_idx], axis=1)
    all_sim = _np.concatenate([cur_sim, cand_sim], axis=1)
    top = _np.argsort(-all_sim, axis=1, kind="stable")[:, :k]
    new_sim = _np.take_along_axis(all_sim, top, axis=1)
    best_sim[rows] = new_sim
    best_idx[rows] = _np.where(_np.isneginf(new_sim), -1, _np.take_along_axis(all_idx, top, axis=1))


def _same_module(a: str, b: str) -> bool:
    """Same module, or a package and its own submodule (trivially similar, not informative)."""
    ma = a.split(":")[1] if a.count(":") >= 1 else a
    mb = b.split(":")[1] if b.count(":") >= 1 else b
    return ma == mb or ma.startswith(mb + ".") or mb.startswith(ma + ".") or \
        ma.startswith(mb + "/") or mb.startswith(ma + "/")


def disambiguate(store: Store, ambiguous: Iterable[dict[str, Any]], model: str | None, revision: str | None) -> int:
    """Resolve references whose name matched several symbols.

    ML: pick the candidate closest to the caller in graph-embedding space when the margin is clear.
    Heuristic: pick the only candidate whose module the caller's module imports (static confidence)."""
    import itertools
    imports_of: dict[str, set[str]] = {}
    resolved = 0
    refs = iter(ambiguous)      # a list, or the scanner's disk-backed list: read in batches either way
    while batch := list(itertools.islice(refs, 500)):
        vecs = _vectors(store, model, {i for a in batch for i in [a["src"], *a["candidates"]]}) if model else {}
        resolved += _disambiguate_batch(store, batch, vecs, imports_of, revision)
    return resolved


def _vectors(store: Store, model: str, ids: set[str]) -> dict[str, Any]:
    """Just the vectors a batch needs: never the whole model (a million vectors in a kernel-sized index). numpy
    arrays when numpy is installed, lists otherwise."""
    out: dict[str, Any] = {}
    ordered = sorted(ids)
    for i in range(0, len(ordered), 500):
        chunk = ordered[i:i + 500]
        marks = ",".join("?" * len(chunk))
        for nid, blob in store.conn.execute(f"SELECT id, vec FROM embeddings WHERE model=? AND id IN ({marks})",
                                            [model, *chunk]):
            out[nid] = decode(blob) if _np is not None else unpack(blob)
    return out


def _dot(a: Any, b: Any) -> float:
    if _np is not None and not isinstance(a, list):
        return float(a @ b) if a.shape == b.shape else 0.0
    return sum(x * y for x, y in zip(a, b))


def _disambiguate_batch(store: Store, ambiguous: list[dict[str, Any]], vecs: dict[str, Any],
                        imports_of: dict[str, set[str]], revision: str | None) -> int:
    resolved = 0
    for amb in ambiguous:
        src = amb["src"]
        cands = [c for c in dict.fromkeys(amb["candidates"]) if c != src]
        if not cands:
            continue
        choice, conf, evidence, method = None, 0.0, ev.HEURISTIC, ""
        if vecs and src in vecs and all(c in vecs for c in cands):
            sims = sorted(((_dot(vecs[src], vecs[c]), c) for c in cands), reverse=True)
            if len(sims) == 1:
                # A single remaining candidate still needs semantic support before we link it.
                if sims[0][0] > 0.4:
                    choice, conf, evidence = sims[0][1], round(min(0.7, 0.3 + sims[0][0] * 0.4), 3), ev.ML_CLASSIFIER
                    method = f"graph-embedding similarity {sims[0][0]:.3f} (single candidate)"
            else:
                margin = sims[0][0] - sims[1][0]
                if margin > 0.04:
                    choice, conf, evidence = sims[0][1], round(min(0.75, 0.4 + margin * 2), 3), ev.ML_CLASSIFIER
                    method = f"graph-embedding similarity (margin {margin:.3f})"
        if choice is None:
            src_mod = "module:" + src.split(":")[1] if src.count(":") >= 2 else src
            if src_mod not in imports_of:
                imports_of[src_mod] = {r[0] for r in store.conn.execute(
                    "SELECT dst FROM edges WHERE src=? AND type='imports'", (src_mod,))}
            hits = [c for c in cands if c.count(":") >= 2 and "module:" + c.split(":")[1] in imports_of[src_mod]]
            if len(hits) == 1:
                choice, conf, evidence, method = hits[0], STATIC_DISAMBIGUATION_CONFIDENCE, ev.HEURISTIC, \
                    "only candidate in an imported module"
        if choice:
            store.add_edge(src, choice, amb["type"], evidence=evidence, confidence=conf, file=amb.get("file"),
                           line=amb.get("line"), owner_file=amb.get("file"), revision=revision,
                           attrs={"resolved": True, "disambiguation": method, "candidates": cands[:6]})
            resolved += 1
    return resolved


def cluster_modules(store: Store, model: str | None, min_similarity: float = 0.6) -> list[dict[str, Any]]:
    """Group modules into semantic components.

    ML: label propagation over import edges plus embedding-similarity edges.
    Heuristic: group by directory (static confidence)."""
    mods = {n["id"]: n for n in store.nodes("module") if not n["attrs"].get("test")}
    if not mods:
        return []
    if not model:
        groups: dict[str, list[str]] = defaultdict(list)
        for mid, n in mods.items():
            parts = (n["file"] or n["name"]).split("/")
            groups["/".join(parts[:-1][:2]) or "(root)"].append(mid)
        return [{"component": k, "modules": sorted(v), "method": "directory grouping", "confidence": 0.5,
                 "evidence": ev.HEURISTIC, "keywords": _keywords(store, v)} for k, v in sorted(groups.items())]
    weights: dict[str, dict[str, float]] = defaultdict(dict)
    for e in store.edges(types=["imports"]):
        if e["src"] in mods and e["dst"] in mods:
            weights[e["src"]][e["dst"]] = weights[e["src"]].get(e["dst"], 0) + 1.0
            weights[e["dst"]][e["src"]] = weights[e["dst"]].get(e["src"], 0) + 1.0
    if _np is None:
        items = [(nid, v) for nid, v in store.embeddings(model) if nid in mods]
        neighbours = ((nid, other, sim) for nid, blob in items for other, sim in top_k(unpack(blob), items, 4))
    else:
        with Matrix(store, model, keep=mods) as m:
            ids = m.ids
            nbr_idx, nbr_sim = _neighbours(m, 4)
        neighbours = ((ids[i], ids[j], s) for i in range(len(ids))
                      for j, s in zip(nbr_idx[i].tolist(), nbr_sim[i].tolist()) if j >= 0)
    for nid, other, sim in neighbours:
        if other != nid and sim >= min_similarity:
            weights[nid][other] = weights[nid].get(other, 0) + sim
            weights[other][nid] = weights[other].get(nid, 0) + sim
    label = {m: m for m in mods}
    order = sorted(mods)
    for _ in range(10):
        changed = False
        for m in order:
            if not weights.get(m):
                continue
            score: dict[str, float] = defaultdict(float)
            for n, w in weights[m].items():
                score[label[n]] += w
            best = max(sorted(score), key=lambda l: score[l])
            if score[best] > score.get(label[m], 0) and best != label[m]:
                label[m] = best
                changed = True
        if not changed:
            break
    groups = defaultdict(list)
    for m, l in label.items():
        groups[l].append(m)
    out = []
    for members in groups.values():
        kw = _keywords(store, members)
        internal = sum(w for a in members for b, w in weights.get(a, {}).items() if b in members) / 2
        external = sum(w for a in members for b, w in weights.get(a, {}).items() if b not in members)
        cohesion = internal / (internal + external) if internal + external else 0.0
        out.append({"component": " / ".join(kw[:2]) or members[0].removeprefix("module:"), "modules": sorted(members),
                    "method": f"label propagation (imports + {model} similarity)", "confidence": round(
                        0.4 + 0.5 * cohesion, 3) if len(members) > 1 else 0.4, "evidence": ev.ML_CLASSIFIER,
                    "cohesion": round(cohesion, 3), "keywords": kw})
    out.sort(key=lambda c: (-len(c["modules"]), c["component"]))
    return out


_STOP = {"the", "and", "for", "with", "from", "this", "that", "init", "self", "none", "str", "int", "list", "dict",
         "py", "ts", "js", "src", "lib", "index", "main", "test", "tests"}


def _keywords(store: Store, module_ids: list[str], n: int = 4) -> list[str]:
    c: Counter[str] = Counter()
    for mid in module_ids:
        node = store.node(mid)
        if not node:
            continue
        for tok in split_identifier(node["name"].replace("/", " ").replace(".", " ")):
            if len(tok) > 2 and tok not in _STOP:
                c[tok] += 2
        for e in store.edges(src=mid, types=["defines"]):
            for tok in split_identifier(e["dst"].rsplit(":", 1)[-1]):
                if len(tok) > 2 and tok not in _STOP:
                    c[tok] += 1
        for word in re.findall(r"[a-z]{4,}", (node["text"] or "").lower())[:30]:
            if word not in _STOP:
                c[word] += 1
    return [w for w, _ in c.most_common(n)]
