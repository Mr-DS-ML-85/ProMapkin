import json
import unittest

from tests.helpers import RepoTestCase
from promapkin import blueprint
from promapkin.agents import orchestrator
from promapkin.agents.contract import RESULT_SCHEMA, TaskContract, merge_result, validate
from promapkin.agents.providers import CommandProvider, ProviderError, make_provider
from promapkin.retrieval import Retriever


class BlueprintTest(RepoTestCase):
    def setUp(self):
        super().setUp()
        self.res = orchestrator.run(self.ws, goal="add pagination to search results")

    def test_all_sections_and_artifacts(self):
        self.assertEqual(self.res["_meta"]["sections"], orchestrator.SECTIONS)
        written = blueprint.write(self.ws, self.res)
        for fname in blueprint.FILES.values():
            self.assertIn(f".promapkin/blueprints/{fname}", written)
        data = json.loads((self.ws.blueprint_dir / "blueprint.json").read_text())
        self.assertEqual(set(data["sections"]), set(orchestrator.SECTIONS))
        md = (self.ws.blueprint_dir / "01-architecture.md").read_text()
        self.assertIn("```mermaid", md)
        self.assertIn("`[observed · ast", md)

    def test_architecture_layers_and_styles(self):
        arch = self.res["architecture"]
        layers = {l["module"]: l["layer"] for l in arch["layers"]}
        self.assertEqual(layers["searchapp.api.routes"], "interface")
        self.assertEqual(layers["searchapp.storage.index_repository"], "persistence")
        styles = " ".join(c["statement"] for c in arch["styles"])
        self.assertIn("Layered", styles)
        self.assertIn("fastapi", styles)
        self.assertTrue(arch["semantic_components"])

    def test_patterns(self):
        found = {p["pattern"]: p for p in self.res["patterns"]["patterns"]}
        self.assertIn("strategy", found)
        self.assertEqual(set(found["strategy"]["roles"]["concrete_strategies"]), {"BM25Ranker", "VectorRanker"})
        self.assertIn("repository", found)
        self.assertIn("adapter", found)
        self.assertTrue(all(p["intent"].startswith("unknown") for p in found.values()))
        roots = self.res["patterns"]["composition_roots"]
        self.assertTrue(any("build_service" in c["statement"] for c in roots))

    def test_decisions_separate_documented_from_inferred(self):
        d = self.res["decisions"]
        self.assertEqual(d["documented"][0]["status"], "Accepted")
        self.assertTrue(d["code_rationale"])
        repo = next(x for x in d["inferred"] if "IndexRepository" in x["statement"])
        self.assertIn("ADR 0001", repo["rationale"])
        strat = next(x for x in d["inferred"] if "Ranker" in x["statement"])
        self.assertTrue(strat["rationale"].startswith("unknown"))

    def test_verification_flags_false_doc_claim(self):
        v = self.res["verification"]
        bad = [c["claim"] for c in v["contradictions"]]
        self.assertEqual(bad, ["`SearchService` writes to `EmbeddingAdapter`"])
        self.assertEqual(v["summary"]["corroborated"], 2)

    def test_api_atlas(self):
        api = self.res["api"]
        eps = {e["endpoint"]: e for e in api["endpoints"]}
        self.assertEqual(set(eps), {"GET /search", "POST /documents"})
        self.assertEqual(eps["GET /search"]["auth_boundary"], "protected")
        self.assertIn("resource:database", eps["POST /documents"]["touches"])
        names = {p["package"] for p in api["packages"]}
        self.assertNotIn("os", names)

    def test_roadmap_is_dependency_ordered(self):
        r = self.res["roadmap"]
        titles = [s["title"] for s in r["steps"]]
        self.assertEqual(titles[0], "Confirm constraints and decisions")
        self.assertEqual(titles[-1], "Final validation")
        mods = r["affected_modules"]
        self.assertLess(mods.index("searchapp.services.search_service"), mods.index("searchapp.api.routes"))
        for st in r["steps"]:
            self.assertTrue(all(p < st["step"] for p in st["prerequisites"]))
            self.assertTrue(st["validation"])
        self.assertTrue(self.ws.store.nodes("roadmap_step"))

    def test_targeted_change_plans_subset(self):
        self.assertEqual(orchestrator.plan_for_changes(["docs/architecture.md"]), ["decisions", "verification"])
        self.assertIn("patterns", orchestrator.plan_for_changes(["searchapp/domain/ranking.py"]))
        self.assertEqual(orchestrator.plan_for_changes([]), [])


class AgentRuntimeTest(RepoTestCase):
    def test_contract_validation_and_merge(self):
        c = TaskContract(specialist="patterns", objective="x", scope=[], revision="r", facts={}).to_dict()
        self.assertTrue(validate({"summary": 1}, RESULT_SCHEMA))
        bad = merge_result(self.ws.store, self.root, c, {"summary": "s"}, agent="t")
        self.assertFalse(bad["accepted"])
        result = {"summary": "Ranking is pluggable.",
                  "claims": [{"claim": "Ranker is a strategy", "kind": "pattern",
                              "citations": ["symbol:searchapp.domain.ranking:Ranker"], "confidence": 0.8},
                             {"claim": "Made-up claim", "citations": ["nope.py:3"]}],
                  "relationships": [{"source": "symbol:searchapp.domain.ranking:VectorRanker",
                                     "target": "symbol:searchapp.adapters.embedding_client:EmbeddingAdapter",
                                     "type": "depends_on"},
                                    {"source": "symbol:ghost", "target": "symbol:ghost2", "type": "calls"}]}
        out = merge_result(self.ws.store, self.root, c, result, agent="t")
        self.assertTrue(out["accepted"])
        self.assertEqual(out["unsupported_claims"], 1)
        self.assertEqual(len(out["rejected"]), 1)

    def test_command_provider_round_trip(self):
        script = self.root.parent / "agent.py"
        script.write_text("import json,sys\nc=json.load(sys.stdin)\nprint(json.dumps({'summary':'ok '+c['specialist'],"
                          "'claims':[{'claim':'checked','citations':['searchapp/cli.py:1']}]}))\n")
        import sys
        prov = CommandProvider(f"{sys.executable} {script}")
        res = orchestrator.run(self.ws, sections=["risks"], provider=prov)
        self.assertEqual(res["risks"]["synthesis"]["summary"], "ok risks")
        self.assertEqual(res["architecture"]["synthesis"]["claims"][0]["status"], "verified")

    def test_remote_provider_requires_opt_in(self):
        self.ws.config.provider = "anthropic"
        with self.assertRaises(ProviderError):
            make_provider(self.ws)


class RetrievalTest(RepoTestCase):
    def test_hybrid_search_finds_by_meaning(self):
        hits = Retriever(self.ws).search("candidate selection", limit=5)
        self.assertEqual(hits[0]["id"], "symbol:searchapp.domain.ranking:select_candidates")
        hits = Retriever(self.ws).search("SearchService.search")
        self.assertEqual(hits[0]["id"], "symbol:searchapp.services.search_service:SearchService.search")
        self.assertIn("symbol", hits[0]["signals"])

    def test_context_respects_budget(self):
        pkg = Retriever(self.ws).build_context("optimize candidate selection for search", budget=600)
        self.assertLessEqual(pkg["estimated_tokens"], 600)
        self.assertFalse(pkg["budget_insufficient"])
        self.assertTrue(pkg["facts"])
        big = Retriever(self.ws).build_context("optimize candidate selection for search", budget=6000)
        self.assertGreaterEqual(len(big["facts"]) + len(big["sources"]), len(pkg["facts"]) + len(pkg["sources"]))

    def test_constraints_never_dropped(self):
        constraints = [f"constraint number {i} " + "x" * 200 for i in range(6)]
        pkg = Retriever(self.ws).build_context("change storage", budget=200, constraints=constraints,
                                               validation=["run tests"])
        self.assertTrue(pkg["budget_insufficient"])
        self.assertGreater(pkg["minimum_budget"], 200)
        self.assertEqual(len([c for c in pkg["constraints"] if c["source"] == "caller"]), 6)
        self.assertEqual(len(pkg["validation"]), 1)

    def test_adr_constraint_included_for_storage_task(self):
        pkg = Retriever(self.ws).build_context("change IndexRepository storage", budget=3000)
        self.assertTrue(any("SQLite" in c["text"] for c in pkg["constraints"]))

    def test_cache_is_revision_bound(self):
        r = Retriever(self.ws)
        self.assertEqual(r.build_context("search", budget=500)["cache"], "miss")
        self.assertEqual(r.build_context("search", budget=500)["cache"], "hit")
        from promapkin.scanner import scan
        self.edit("searchapp/cli.py", "def main", "def main_entry")
        scan(self.ws)
        self.assertEqual(Retriever(self.ws).build_context("search", budget=500)["cache"], "miss")


if __name__ == "__main__":
    unittest.main()
