"""Large-repository code paths (streaming, blocked, approximate) must agree with the small-repository ones.

The thresholds that switch them on are lowered here so the fixture repo exercises them."""

import unittest
from unittest import mock

from tests.helpers import RepoTestCase, git
from promapkin import evidence as ev
from promapkin.ml import graph_ml

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - numpy is optional
    np = None


def _vectors(store, model):
    return {nid: bytes(v) for nid, v in store.embeddings(model)}


class StreamingStoreTest(RepoTestCase):
    def test_iter_refs_and_iter_nodes_page_through_everything(self):
        s = self.ws.store
        self.assertEqual(sorted(map(repr, s.iter_refs(page=3))), sorted(map(repr, s.refs())))
        calls = [r for r in s.refs() if r["type"] == "calls"]
        self.assertEqual(len(list(s.iter_refs(where="type = 'calls'", page=2))), len(calls))
        files = {r["owner_file"] for r in calls[:3]}
        self.assertEqual(sorted(map(repr, s.iter_refs(files))), sorted(map(repr, s.refs(files))))
        self.assertEqual({n["id"] for n in s.iter_nodes(page=4)}, {n["id"] for n in s.nodes()})
        self.assertNotIn("_rowid", next(s.iter_nodes()))

    def test_spool_round_trip(self):
        from promapkin.extract import Facts
        from promapkin.scanner import _Spool
        spool = _Spool(self.ws.dir)
        f = Facts(path="a.py", language="python")
        f.node("module:a", "module", "a", ev.AST)
        spool.add("a.py", False, f, "python-ast")
        spool.add("b.bin", True, None, None)
        self.assertIn("a.py", spool)
        generated, facts, parser = spool.get("a.py")
        self.assertEqual((generated, parser, facts.nodes[0]["id"]), (False, "python-ast", "module:a"))
        self.assertEqual(spool.get("b.bin"), (True, None, None))
        spool.close()


class ResolverRegressionTest(RepoTestCase):
    def test_call_through_a_submodule_name_does_not_crash(self):
        # Regression (Linux scripts/sbom): `sub.missing()` inside package `pkg`, where `pkg.sub` is a module, made
        # the resolver look up the module's id among classes and abort the scan with KeyError.
        (self.root / "pkg").mkdir()
        (self.root / "pkg" / "__init__.py").write_text("def entry():\n    return sub.not_defined_here()\n")
        (self.root / "pkg" / "sub.py").write_text("def helper():\n    return 1\n")
        git(self.root, "add", "-A")
        git(self.root, "commit", "-qm", "pkg")
        from promapkin.scanner import scan
        r = scan(self.ws)
        self.assertGreater(r["changed"], 0)


class TwoPhaseCommitTest(RepoTestCase):
    def test_structural_index_survives_a_killed_enrichment_pass(self):
        from promapkin import scanner
        s = self.ws.store
        (self.root / "searchapp" / "extra.py").write_text("def brand_new_helper():\n    return 1\n")
        git(self.root, "add", "-A")
        git(self.root, "commit", "-qm", "extra")

        def killed(*_a, **_k):
            raise KeyboardInterrupt   # stands in for SIGKILL/OOM: not an Exception, so nothing catches it

        with mock.patch.object(scanner, "graph_ml_pass", killed), self.assertRaises(KeyboardInterrupt):
            scanner.scan(self.ws)
        # The extraction and resolution were committed before enrichment started.
        self.assertIsNotNone(s.node("symbol:searchapp.extra:brand_new_helper"))
        self.assertEqual(s.get_meta("enrichment_pending"), "1")
        with mock.patch.object(scanner, "resolve", wraps=scanner.resolve) as res:
            r = scanner.scan(self.ws)
        self.assertEqual(r["changed"], 0)
        self.assertIsNone(res.call_args.kwargs.get("files"))      # pending enrichment: full re-resolution
        self.assertIsNone(s.get_meta("enrichment_pending"))
        self.assertIn("classified", r)


@unittest.skipIf(np is None, "numpy not installed")
class LargeRepoGraphMLTest(RepoTestCase):
    def embedded(self):
        from promapkin import ml
        from promapkin.scanner import embed_pass
        rt = ml.runtime(self.ws)
        with self.ws.store.transaction():
            embed_pass(self.ws.store, rt)
        return rt.embedder.name

    def test_parallel_embedding_matches_sequential(self):
        from promapkin import ml
        from promapkin import scanner
        rt = ml.runtime(self.ws)
        s = self.ws.store
        with s.transaction():
            scanner.embed_pass(s, rt)
        sequential = _vectors(s, rt.embedder.name)
        with s.transaction():
            s.conn.execute("DELETE FROM embeddings")
            with mock.patch.object(scanner, "PARALLEL_EMBED_MIN", 1), \
                    mock.patch.object(scanner, "_scan_workers_for", lambda: 2):
                n = scanner.embed_pass(s, rt)
        self.assertEqual(n, len(sequential))
        self.assertEqual(_vectors(s, rt.embedder.name), sequential)

    def test_blocked_propagation_matches_one_block(self):
        base = self.embedded()
        s = self.ws.store
        gm = graph_ml.graph_model_name(base)
        with s.transaction():
            graph_ml.propagate(s, base)
            whole = _vectors(s, gm)
            with mock.patch.object(graph_ml, "BLOCK_FLOATS", 1):    # one row per block
                graph_ml.propagate(s, base)
            blocked = _vectors(s, gm)
        self.assertEqual(whole.keys(), blocked.keys())
        for nid in whole:
            np.testing.assert_allclose(np.frombuffer(whole[nid], np.float32), np.frombuffer(blocked[nid], np.float32),
                                       atol=1e-6)

    def test_half_precision_state_stays_close(self):
        base = self.embedded()
        s = self.ws.store
        gm = graph_ml.graph_model_name(base)
        with s.transaction():
            graph_ml.propagate(s, base)
            full = _vectors(s, gm)
            with mock.patch.object(graph_ml, "RAM_STATE_BYTES", 0):     # what a kernel-sized graph uses
                graph_ml.propagate(s, base)
            half = _vectors(s, gm)
        worst = max(float(np.abs(np.frombuffer(full[k], np.float32) - np.frombuffer(half[k], np.float32)).max())
                    for k in full)
        self.assertLess(worst, 2e-3)

    def test_lsh_similarity_mode_produces_ml_edges(self):
        base = self.embedded()
        s = self.ws.store
        gm = graph_ml.graph_model_name(base)
        with s.transaction():
            graph_ml.propagate(s, base)
            exact = graph_ml.similarity_edges(s, gm, threshold=0.5)
            exact_pairs = {(e["src"], e["dst"]) for e in s.edges(types=["similar_to"])}
            with mock.patch.object(graph_ml, "EXACT_MAX_NODES", 0):
                approx = graph_ml.similarity_edges(s, gm, threshold=0.5)
            approx_pairs = {(e["src"], e["dst"]) for e in s.edges(types=["similar_to"])}
        self.assertGreater(exact, 0)
        self.assertTrue(approx_pairs <= exact_pairs | approx_pairs)
        self.assertGreaterEqual(len(approx_pairs & exact_pairs), exact // 2)
        self.assertTrue(all(e["evidence"] == ev.ML_CLASSIFIER for e in s.edges(types=["similar_to"])))

    def test_lsh_finds_planted_near_duplicates(self):
        rng = np.random.default_rng(7)
        n, dim = 4000, 64
        base = rng.standard_normal((n // 2, dim)).astype(np.float32)
        twins = base + 0.15 * rng.standard_normal(base.shape).astype(np.float32)
        mat = np.concatenate([base, twins])
        mat /= np.linalg.norm(mat, axis=1, keepdims=True)
        idx, sim = graph_ml._lsh_top_k(mat, 2)
        found = sum(1 for i in range(n // 2) if (i + n // 2) in idx[i])
        self.assertGreater(found / (n // 2), 0.97)
        exact_idx, _ = graph_ml._exact_top_k(mat, 2)
        self.assertTrue(all(exact_idx[i][1] == i + n // 2 for i in range(0, n // 2, 97)))

    def test_disambiguation_reads_vectors_in_batches(self):
        base = self.embedded()
        s = self.ws.store
        with s.transaction():
            graph_ml.propagate(s, base)
            gm = graph_ml.graph_model_name(base)
            src = "symbol:searchapp.services.search_service:SearchService.search"
            amb = [{"src": src, "candidates": [src, "symbol:searchapp.storage.index_repository:IndexRepository.get",
                                               "symbol:web/client:SearchClient.search"], "type": "calls",
                    "file": None, "line": None}] * 1200
            with mock.patch.object(graph_ml, "_vectors", wraps=graph_ml._vectors) as fetch:
                graph_ml.disambiguate(s, amb, gm, None)
        self.assertEqual(fetch.call_count, 3)      # 500 references per batch


if __name__ == "__main__":
    unittest.main()
