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
from typing import Any

from promapkin import evidence as ev
from promapkin.ml.embedders import _np, normalize, pack, top_k, unpack
from promapkin.store import Store, split_identifier

STRUCTURAL_EDGES = ["calls", "imports", "inherits", "implements", "depends_on", "instantiates", "contains",
                    "defines", "routes_to", "overrides", "references"]
SIMILAR_KINDS = ("class", "function", "method", "module")
STATIC_SIMILARITY_CONFIDENCE = 0.4
STATIC_DISAMBIGUATION_CONFIDENCE = 0.45


def graph_model_name(base: str) -> str:
    return f"graph:{base}"


def propagate(store: Store, base_model: str, alpha: float = 0.5, iterations: int = 2) -> int:
    """h_{t+1} = normalize((1-alpha) * h_0 + alpha * mean(h_t over neighbours)). Stored as graph:<model>."""
    base = {nid: unpack(v) for nid, v in store.embeddings(base_model)}
    if not base:
        return 0
    adj: dict[str, set[str]] = defaultdict(set)
    for e in store.edges(types=STRUCTURAL_EDGES):
        if e["src"] in base and e["dst"] in base:
            adj[e["src"]].add(e["dst"])
            adj[e["dst"]].add(e["src"])
    if _np is not None:
        h = _propagate_numpy(base, adj, alpha, iterations)
    else:
        h = _propagate_python(base, adj, alpha, iterations)
    name = graph_model_name(base_model)
    store.conn.execute("DELETE FROM embeddings WHERE model=?", (name,))
    store.conn.executemany("INSERT OR REPLACE INTO embeddings(id, model, sha, vec) VALUES (?,?,?,?)",
                           ((nid, name, "propagated", vec.astype("float32").tobytes() if hasattr(vec, "astype")
                             else pack(vec)) for nid, vec in h.items()))
    return len(h)


def _propagate_numpy(base: dict[str, list[float]], adj: dict[str, set[str]], alpha: float, iterations: int
                     ) -> dict[str, Any]:
    ids = list(base)
    index = {nid: i for i, nid in enumerate(ids)}
    h0 = _np.asarray([base[i] for i in ids], dtype=_np.float32)
    src = _np.fromiter((index[a] for a, bs in adj.items() for _ in bs), dtype=_np.int64)
    dst = _np.fromiter((index[b] for a, bs in adj.items() for b in bs), dtype=_np.int64)
    deg = _np.bincount(src, minlength=len(ids)).astype(_np.float32)
    h = h0
    for _ in range(iterations):
        agg = _np.zeros_like(h0)
        _np.add.at(agg, src, h[dst])
        has = deg > 0
        mixed = h0.copy()
        mixed[has] = (1 - alpha) * h0[has] + alpha * agg[has] / deg[has, None]
        norms = _np.linalg.norm(mixed, axis=1, keepdims=True)
        h = mixed / _np.where(norms == 0, 1, norms)
    return {nid: h[i] for i, nid in enumerate(ids)}


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

    ML: cosine similarity of graph embeddings (evidence ml_classifier, confidence = similarity).
    Heuristic: identifier-token Jaccard overlap (evidence heuristic, static confidence)."""
    store.delete_edges("type='similar_to'")
    kinds = ",".join(f"'{x}'" for x in SIMILAR_KINDS)
    nodes = {n["id"]: n for n in store.nodes(where=f"kind IN ({kinds}) AND json_extract(attrs, '$.test') IS NOT 1")
             if not (n["kind"] == "module" and (n["attrs"].get("loc") or 0) < 5)}
    count = 0
    seen: set[tuple[str, str]] = set()
    if model:
        items = [(nid, v) for nid, v in store.embeddings(model) if nid in nodes]
        for nid, neighbours in _all_top_k(items, k + 1):
            for other, sim in neighbours:
                if other == nid or sim < threshold or _same_module(nid, other) or \
                        nodes[nid]["kind"] != nodes[other]["kind"]:
                    continue
                key = tuple(sorted((nid, other)))
                if key in seen:
                    continue
                seen.add(key)
                store.add_edge(key[0], key[1], "similar_to", evidence=ev.ML_CLASSIFIER, confidence=round(sim, 3),
                               revision=revision, attrs={"method": f"cosine({model})", "similarity": round(sim, 3)})
                count += 1
        return count
    tokens = {nid: set(split_identifier(n["name"])) - {"get", "set", "init", "self"} for nid, n in nodes.items()}
    by_token: dict[str, list[str]] = defaultdict(list)
    for nid, toks in tokens.items():
        for t in toks:
            by_token[t].append(nid)
    for nid, toks in tokens.items():
        if len(toks) < 2:
            continue
        cands = Counter(o for t in toks for o in by_token[t] if o != nid)
        for other, shared in cands.most_common(k):
            union = len(toks | tokens[other])
            if shared / union < 0.6 or _same_module(nid, other) or nodes[nid]["kind"] != nodes[other]["kind"]:
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


def _all_top_k(items: list[tuple[str, bytes]], k: int) -> list[tuple[str, list[tuple[str, float]]]]:
    """Top-k cosine neighbours for every item: one blocked matrix product with numpy, else pairwise."""
    if not items:
        return []
    if _np is None:
        return [(nid, top_k(unpack(blob), items, k)) for nid, blob in items]
    ids = [i for i, _ in items]
    mat = _np.frombuffer(b"".join(v for _, v in items), dtype=_np.float32).reshape(len(items), -1)
    out = []
    kk = min(k, len(ids))
    for start in range(0, len(ids), 1024):
        block = mat[start:start + 1024] @ mat.T
        part = _np.argpartition(-block, kk - 1, axis=1)[:, :kk]
        for r in range(block.shape[0]):
            cols = part[r][_np.argsort(-block[r, part[r]])]
            out.append((ids[start + r], [(ids[c], float(block[r, c])) for c in cols]))
    return out


def _same_module(a: str, b: str) -> bool:
    """Same module, or a package and its own submodule (trivially similar, not informative)."""
    ma = a.split(":")[1] if a.count(":") >= 1 else a
    mb = b.split(":")[1] if b.count(":") >= 1 else b
    return ma == mb or ma.startswith(mb + ".") or mb.startswith(ma + ".") or \
        ma.startswith(mb + "/") or mb.startswith(ma + "/")


def disambiguate(store: Store, ambiguous: list[dict[str, Any]], model: str | None, revision: str | None) -> int:
    """Resolve references whose name matched several symbols.

    ML: pick the candidate closest to the caller in graph-embedding space when the margin is clear.
    Heuristic: pick the only candidate whose module the caller's module imports (static confidence)."""
    vecs = {nid: unpack(v) for nid, v in store.embeddings(model)} if model else {}
    imports_of: dict[str, set[str]] = {}
    resolved = 0
    for amb in ambiguous:
        src = amb["src"]
        cands = [c for c in dict.fromkeys(amb["candidates"]) if c != src]
        if not cands:
            continue
        choice, conf, evidence, method = None, 0.0, ev.HEURISTIC, ""
        if vecs and src in vecs and all(c in vecs for c in cands):
            sims = sorted(((sum(a * b for a, b in zip(vecs[src], vecs[c])), c) for c in cands), reverse=True)
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
                imports_of[src_mod] = {e["dst"] for e in store.edges(src=src_mod, types=["imports"])}
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
    items = [(nid, v) for nid, v in store.embeddings(model) if nid in mods]
    for nid, blob in items:
        for other, sim in top_k(unpack(blob), items, 4):
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
