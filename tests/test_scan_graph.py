import unittest
from unittest import mock

from tests.helpers import RepoTestCase, git
from promapkin import evidence as ev, knowledge
from promapkin.graph import Graph
from promapkin.scanner import scan


class ScanTest(RepoTestCase):
    def test_revision_and_counts(self):
        r = self.scan_result
        self.assertEqual(r["files"], 19)
        self.assertNotIn("dirty", r["revision"])
        self.assertEqual(self.ws.store.get_meta("indexed_revision"), r["revision"])
        self.assertEqual(r["errors"], [])

    def test_static_resolution(self):
        s = self.ws.store
        calls = {(e["src"], e["dst"]): e["evidence"] for e in s.edges(types=["calls"])}
        self.assertEqual(calls[("symbol:searchapp.services.search_service:SearchService.search",
                                "symbol:searchapp.storage.index_repository:IndexRepository.find_all")],
                         ev.STATIC_RESOLVED)
        self.assertIn(("symbol:searchapp.api.routes:search_endpoint",
                       "symbol:searchapp.services.search_service:SearchService.search"), calls)
        overrides = {(e["src"], e["dst"]) for e in s.edges(types=["overrides"])}
        self.assertIn(("symbol:searchapp.domain.ranking:BM25Ranker.score",
                       "symbol:searchapp.domain.ranking:Ranker.score"), overrides)
        imports = {(e["src"], e["dst"]) for e in s.edges(types=["imports"])}
        self.assertIn(("module:searchapp.api.routes", "package:fastapi"), imports)
        self.assertTrue(s.node("package:os")["attrs"]["stdlib"])

    def test_incremental_scan_only_touches_changed_files(self):
        self.edit("searchapp/storage/index_repository.py", "def get(self", "def get_one(self")
        r = scan(self.ws)
        self.assertEqual(r["changed"], 1)
        self.assertEqual(r["changed_files"], ["searchapp/storage/index_repository.py"])
        s = self.ws.store
        self.assertIsNone(s.node("symbol:searchapp.storage.index_repository:IndexRepository.get"))
        self.assertIsNotNone(s.node("symbol:searchapp.storage.index_repository:IndexRepository.get_one"))
        self.assertEqual(scan(self.ws)["changed"], 0)

    def test_deleted_file_is_forgotten(self):
        (self.root / "web" / "client.ts").unlink()
        r = scan(self.ws)
        self.assertEqual(r["deleted"], 1)
        self.assertIsNone(self.ws.store.node("module:web/client"))

    def test_agent_knowledge_goes_stale_when_cited_file_changes(self):
        s = self.ws.store
        with s.transaction():
            out = knowledge.add_node(s, self.root, kind="finding", name="ranking is slow",
                                     citations=["searchapp/domain/ranking.py:20"], agent="a")
        self.assertEqual(s.node(out["id"])["status"], "verified")
        self.edit("searchapp/domain/ranking.py", "terms = query", "terms  = query")
        r = scan(self.ws)
        self.assertEqual(r["stale_inferences"], 1)
        self.assertEqual(s.node(out["id"])["status"], "stale")

    def test_failed_scan_keeps_last_valid_index(self):
        before = self.ws.store.counts()
        self.edit("searchapp/cli.py", "def main", "def main2")
        with mock.patch("promapkin.scanner.resolve", side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                scan(self.ws)
        self.assertEqual(self.ws.store.counts(), before)
        self.assertIsNotNone(self.ws.store.node("symbol:searchapp.cli:main"))
        self.assertEqual(self.ws.store.last_jobs(1)[0]["status"], "failed")

    def test_commit_changes_revision(self):
        self.edit("searchapp/cli.py", "def main", "def run_main")
        git(self.root, "commit", "-qam", "rename")
        r = scan(self.ws)
        self.assertNotEqual(r["revision"], self.scan_result["revision"])


class GraphTest(RepoTestCase):
    def test_resolve_forms(self):
        g = Graph(self.ws.store)
        self.assertEqual(g.resolve("IndexRepository"), "symbol:searchapp.storage.index_repository:IndexRepository")
        self.assertEqual(g.resolve("SearchService::search"),
                         "symbol:searchapp.services.search_service:SearchService.search")
        self.assertEqual(g.resolve("api:search"), "api:GET /search")

    def test_impact_reaches_endpoints_and_tests(self):
        r = Graph(self.ws.store).impact("IndexRepository.find_all")
        endpoints = {x["id"] for x in r["affected"].get("endpoints", [])}
        self.assertIn("api:GET /search", endpoints)
        tests = {t["id"] for t in r["tests_to_run"]}
        self.assertIn("symbol:tests.test_search_service:test_search_ranks_matching_documents_first", tests)
        for items in r["affected"].values():
            scores = [x["impact_score"] for x in items]
            self.assertEqual(scores, sorted(scores, reverse=True))

    def test_trace_prefers_trusted_paths_and_reports_evidence(self):
        r = Graph(self.ws.store).trace("api:GET /search", "IndexRepository.find_all")
        self.assertTrue(r["found"])
        best = r["paths"][0]
        self.assertEqual(best["basis"], "static")
        self.assertEqual(best["nodes"][-1], "symbol:searchapp.storage.index_repository:IndexRepository.find_all")
        self.assertTrue(all("evidence" in s for s in best["steps"]))

    def test_dataflow_finds_database_write(self):
        r = Graph(self.ws.store).dataflow("api:POST /documents")
        ops = {(f["operation"], f["sink"]) for f in r["flows"]}
        self.assertIn(("writes", "resource:database"), ops)

    def test_dispatch_through_strategy_is_inferred(self):
        r = Graph(self.ws.store).trace("SearchService.search", "EmbeddingAdapter.embed")
        self.assertTrue(r["found"])
        self.assertIn("dispatches", [s["type"] for s in r["paths"][0]["steps"]])
        self.assertEqual(r["paths"][0]["basis"], "inferred")

    def test_corroboration_raises_trust(self):
        self.assertGreater(ev.weight(ev.DOCUMENTATION, corroborations=1), ev.weight(ev.DOCUMENTATION))
        self.assertLess(ev.weight(ev.AST, "stale"), ev.weight(ev.AST))

    def test_dependency_report(self):
        r = Graph(self.ws.store).dependency_report("SearchService")
        self.assertEqual(r["module"], "module:searchapp.services.search_service")
        self.assertIn("searchapp.api.routes", [d["module"] for d in r["depended_on_by"]])
        self.assertEqual(r["cycles"], [])


if __name__ == "__main__":
    unittest.main()
