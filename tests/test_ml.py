import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests.helpers import RepoTestCase
from promapkin import evidence as ev
from promapkin.ml import graph_ml
from promapkin.ml.classify import TASKS, HybridClassifier, NaiveBayes
from promapkin.ml.embedders import Embedder, HashingEmbedder, cosine
from promapkin.ml.models import ModelManager


class FakeNeural(Embedder):
    """Stand-in for a downloaded neural model (exercises the DL code paths offline)."""
    name = "fake-neural"
    neural = True

    def __init__(self):
        self.inner = HashingEmbedder(256)
        self.dim = 256

    def embed(self, texts):
        return self.inner.embed(texts)


class ClassifierTest(unittest.TestCase):
    def test_hybrid_agreement_raises_confidence(self):
        clf = HybridClassifier(HashingEmbedder())
        c = clf.classify("layer", "myapp.storage.user_repository save find sql")
        self.assertEqual(c.label, "persistence")
        self.assertTrue(c.agreement)
        self.assertTrue(c.tier.startswith("heuristic+"))
        rule_only = HybridClassifier(mode="heuristic").classify("layer", "myapp.storage.user_repository")
        self.assertEqual(rule_only.tier, "heuristic")
        self.assertGreater(c.confidence, rule_only.confidence)

    def test_heuristic_mode_uses_static_confidence(self):
        clf = HybridClassifier(mode="heuristic")
        self.assertEqual(clf.classify("note_intent", "TODO: paginate").as_dict()["confidence"], 0.95)
        self.assertEqual(clf.classify("layer", "zzz qqq").label, "unknown")

    def test_model_only_answer_when_rules_abstain(self):
        c = HybridClassifier(HashingEmbedder()).classify("resource", "boto3 put_object bucket upload")
        self.assertIn(c.tier, ("builtin-nb",))
        self.assertTrue(0 < c.confidence <= 1)

    def test_design_pattern_rules_and_ml(self):
        clf = HybridClassifier(HashingEmbedder())
        c = clf.classify("design_pattern", "class: PaymentRepository; methods: get add delete; touches: database")
        self.assertEqual(c.label, "repository")
        c = clf.classify("design_pattern", "class PaymentGatewayAdapter; methods: charge; touches: http")
        self.assertEqual(c.label, "adapter")

    def test_neural_prototype_tier(self):
        clf = HybridClassifier(FakeNeural())
        self.assertEqual(clf.model_tier, "neural")
        c = clf.classify("design_pattern", "class: Shape; methods: area; subclasses: 3; abstract: yes")
        self.assertEqual(c.label, "strategy")
        self.assertIn("neural", c.tier)

    def test_feedback_examples_are_learned(self):
        with tempfile.TemporaryDirectory() as d:
            clf = HybridClassifier(HashingEmbedder(), training_dir=Path(d))
            for _ in range(5):
                clf.add_example("resource", "zephyrlib frobnicate widget", "queue")
            c = clf.classify("resource", "zephyrlib frobnicate widget")
            self.assertEqual(c.label, "queue")
            with self.assertRaises(ValueError):
                clf.add_example("resource", "x", "not-a-label")

    def test_naive_bayes_probabilities(self):
        nb = NaiveBayes({"a": ["apple banana"], "b": ["car truck"]}, HashingEmbedder().features)
        ranked = nb.predict("banana apple")
        self.assertEqual(ranked[0][0], "a")
        self.assertAlmostEqual(sum(p for _, p in ranked), 1.0, places=6)

    def test_every_task_has_rules_and_seeds(self):
        for t in TASKS.values():
            self.assertTrue(t.labels and t.seeds)


class EmbedderTest(unittest.TestCase):
    def test_hashing_embedder_semantics(self):
        e = HashingEmbedder()
        a, b, c = e.embed(["VectorIndex search", "vector_index_search", "cooking recipes"])
        self.assertGreater(cosine(a, b), cosine(a, c))
        self.assertAlmostEqual(cosine(a, a), 1.0, places=5)


class ModelPolicyTest(unittest.TestCase):
    def test_never_mode_does_not_download(self):
        with mock.patch.dict(os.environ, {"PROMAPKIN_ML_MODE": "never"}):
            mgr = ModelManager()
            with mock.patch.object(mgr, "download") as dl:
                d = mgr.resolve(interactive=True, ask=lambda q: True)
            dl.assert_not_called()
        self.assertEqual(d.tier, "builtin")

    def test_ask_mode_declined_falls_back_and_is_remembered(self):
        answers = []
        with mock.patch.dict(os.environ, {"PROMAPKIN_ML_MODE": "ask"}), \
                mock.patch.object(ModelManager, "cached", return_value=False):
            mgr = ModelManager(on_consent=answers.append)
            with mock.patch.object(mgr, "download") as dl:
                d = mgr.resolve(interactive=True, ask=lambda q: False)
            dl.assert_not_called()
        self.assertEqual(d.tier, "builtin")
        self.assertTrue(d.download_available)
        self.assertEqual(answers, ["declined"])

    def test_non_interactive_never_prompts(self):
        with mock.patch.dict(os.environ, {"PROMAPKIN_ML_MODE": "ask"}), \
                mock.patch.object(ModelManager, "cached", return_value=False):
            mgr = ModelManager()
            asked = []
            d = mgr.resolve(interactive=False, ask=lambda q: asked.append(q) or True)
        self.assertEqual(asked, [])
        self.assertEqual(d.tier, "builtin")

    def test_consent_granted_downloads(self):
        with mock.patch.dict(os.environ, {"PROMAPKIN_ML_MODE": "ask"}), \
                mock.patch.object(ModelManager, "cached", return_value=False):
            mgr = ModelManager()
            with mock.patch.object(mgr, "download", return_value=(True, "ok")) as dl, \
                    mock.patch.object(mgr, "_load", return_value=FakeNeural()):
                d = mgr.resolve(interactive=True, ask=lambda q: True)
            dl.assert_called()
        self.assertEqual(d.tier, "neural")


class GraphMLTest(RepoTestCase):
    def test_propagation_similarity_and_clusters(self):
        s = self.ws.store
        from promapkin.scanner import embed_pass
        from promapkin import ml
        rt = ml.runtime(self.ws)
        with s.transaction():
            embed_pass(s, rt)
            n = graph_ml.propagate(s, rt.embedder.name)
            self.assertGreater(n, 10)
            gm = graph_ml.graph_model_name(rt.embedder.name)
            count = graph_ml.similarity_edges(s, gm, threshold=0.5)
        self.assertGreater(count, 0)
        for e in s.edges(types=["similar_to"]):
            self.assertEqual(e["evidence"], ev.ML_CLASSIFIER)
        clusters = graph_ml.cluster_modules(s, gm)
        self.assertTrue(clusters)
        self.assertEqual(sum(len(c["modules"]) for c in clusters),
                         len([m for m in s.nodes("module") if not m["attrs"].get("test")]))

    def test_heuristic_similarity_uses_static_confidence(self):
        s = self.ws.store
        with s.transaction():
            graph_ml.similarity_edges(s, None)
        for e in s.edges(types=["similar_to"]):
            self.assertEqual(e["evidence"], ev.HEURISTIC)
            self.assertEqual(e["confidence"], graph_ml.STATIC_SIMILARITY_CONFIDENCE)
        clusters = graph_ml.cluster_modules(s, None)
        self.assertTrue(all(c["method"] == "directory grouping" for c in clusters))

    def test_disambiguation(self):
        s = self.ws.store
        amb = [{"src": "symbol:searchapp.services.search_service:SearchService.search",
                "candidates": ["symbol:searchapp.storage.index_repository:IndexRepository.get",
                               "symbol:web/client:SearchClient.search"], "type": "calls", "file": None,
                "line": None}]
        with s.transaction():
            n = graph_ml.disambiguate(s, amb, None, None)
        self.assertEqual(n, 1)   # only one candidate lives in an imported module
        e = s.edges(src=amb[0]["src"], dst="symbol:searchapp.storage.index_repository:IndexRepository.get")
        self.assertEqual(e[0]["confidence"], graph_ml.STATIC_DISAMBIGUATION_CONFIDENCE)

    def test_disambiguation_never_crashes_on_short_candidate_lists(self):
        # Regression: a single remaining candidate used to raise IndexError (sims[1]) and roll back the scan.
        s = self.ws.store
        from promapkin.scanner import embed_pass
        from promapkin import ml
        rt = ml.runtime(self.ws)
        src = "symbol:searchapp.services.search_service:SearchService.search"
        with s.transaction():
            embed_pass(s, rt)
            graph_ml.propagate(s, rt.embedder.name)
            gm = graph_ml.graph_model_name(rt.embedder.name)
            amb = [{"src": src, "candidates": [src, "symbol:searchapp.storage.index_repository:IndexRepository.get"],
                    "type": "calls", "file": None, "line": None},
                   {"src": src, "candidates": [src], "type": "calls", "file": None, "line": None},
                   {"src": src, "candidates": [], "type": "calls", "file": None, "line": None}]
            graph_ml.disambiguate(s, amb, gm, None)
            graph_ml.disambiguate(s, amb, None, None)

    def test_layers_are_classified_during_scan(self):
        mod = self.ws.store.node("module:searchapp.storage.index_repository")
        self.assertEqual(mod["attrs"]["layer"]["label"], "persistence")

    def test_heuristic_backend_disables_models(self):
        from promapkin.config import Workspace
        from promapkin.scanner import scan
        self.ws.config.ml_backend = "heuristic"
        self.ws.config.save(self.ws.config_path)
        ws = Workspace(self.root)
        r = scan(ws, full=True)
        self.assertEqual(r["ml"]["classifier_mode"], "heuristic")
        self.assertEqual(r["graph_ml"], "heuristic (static confidence)")
        mod = ws.store.node("module:searchapp.storage.index_repository")
        self.assertEqual(mod["attrs"]["layer"]["tier"], "heuristic")
        ws.close()


if __name__ == "__main__":
    unittest.main()
