import io
import json
import unittest

from tests.helpers import RepoTestCase, git
from promapkin.handoff import HandoffError, Handoffs, VersionConflict, to_yaml
from promapkin.mcp_server import TOOLS, Server, serve
from promapkin.scanner import scan

HANDOFF = {
    "task": "optimize vector search", "status": "investigation_complete",
    "findings": [{"claim": "candidate selection is a bottleneck", "sources": ["searchapp/domain/ranking.py:38"],
                  "evidence": "benchmark", "status": "verified"}],
    "constraints": ["preserve existing API", "maintain recall threshold"],
    "relevant_context": {"symbols": ["SearchService.search", "select_candidates"], "dependencies": ["IndexRepository"]},
    "decisions": [{"decision": "benchmark before changing algorithm", "rationale": "preserve quality"}],
    "completed": ["traced candidate selection"], "next_action": "implement and benchmark",
    "validation": ["run targeted tests", "compare latency", "verify recall"],
}


class HandoffTest(RepoTestCase):
    def setUp(self):
        super().setUp()
        self.h = Handoffs(self.ws)
        self.created = self.h.create(HANDOFF, agent="agent-a")
        self.hid = self.created["id"]

    def test_create_projects_into_graph(self):
        g = self.created["graph"]
        self.assertIn("symbol:searchapp.domain.ranking:select_candidates", g["linked_symbols"])
        self.assertEqual(g["unresolved_symbols"], [])
        s = self.ws.store
        self.assertEqual(s.node(f"handoff:{self.hid}")["kind"], "handoff")
        finding = s.node(g["findings"][0])
        self.assertEqual(finding["evidence"], "runtime")   # benchmark evidence
        cons = s.nodes("constraint")
        self.assertEqual(len(cons), 2)
        self.assertTrue(s.edges(src=cons[0]["id"], types=["constrains"]))
        self.assertTrue((self.ws.dir / "handoffs" / f"{self.hid}.yaml").exists())

    def test_versions_and_optimistic_concurrency(self):
        u = self.h.update(self.hid, {"completed": ["implemented heap"], "status": "in_progress"}, agent="agent-b",
                          expected_version=1)
        self.assertEqual(u["version"], 2)
        self.assertEqual(u["handoff"]["completed"], ["traced candidate selection", "implemented heap"])
        with self.assertRaises(VersionConflict):
            self.h.update(self.hid, {"next_action": "x"}, agent="agent-c", expected_version=1)
        self.assertEqual([v["version"] for v in self.h.history(self.hid)], [1, 2])

    def test_conflicting_findings_are_recorded_not_overwritten(self):
        u = self.h.update(self.hid, {"findings": [{"claim": "candidate selection is a bottleneck",
                                                   "status": "refuted"}]}, agent="agent-c")
        self.assertEqual(len(u["conflicts"]), 1)
        self.assertEqual(len(u["handoff"]["findings"]), 2)
        self.assertEqual(u["handoff"]["findings"][0]["status"], "verified")

    def test_branch_inherits_constraints(self):
        b = self.h.branch(self.hid, "benchmark recall", agent="agent-d")
        self.assertEqual(b["handoff"]["parent_id"], self.hid)
        self.assertIn("preserve existing API", b["handoff"]["constraints"])
        r = self.h.resume(b["id"])
        self.assertEqual(r["lineage"][0]["id"], self.hid)
        self.assertIn(b["id"], self.h.resume(self.hid)["branches"])

    def test_resume_builds_context_and_detects_staleness(self):
        r = self.h.resume(self.hid, budget=1500)
        self.assertFalse(r["freshness"]["stale"])
        texts = [c["text"] for c in r["context"]["constraints"]]
        self.assertIn("preserve existing API", texts)
        self.assertEqual(len(r["context"]["validation"]), 3)
        self.assertIn("implement and benchmark", r["context"]["next_actions"])
        # Another agent changes the code the finding cites, and commits.
        self.edit("searchapp/domain/ranking.py", "return documents[: limit * 4]", "return documents[: limit * 2]")
        git(self.root, "commit", "-qam", "tune candidates")
        scan(self.ws)
        r = self.h.resume(self.hid)
        self.assertTrue(r["freshness"]["revision_changed"])
        self.assertTrue(r["freshness"]["stale"])
        self.assertEqual(r["freshness"]["findings"][0]["current"], "needs_revalidation")

    def test_claim_marks_in_progress(self):
        r = self.h.resume(self.hid, agent="agent-b", claim=True)
        self.assertEqual(r["claimed_by"], "agent-b")
        self.assertEqual(self.h.get(self.hid)["status"], "in_progress")

    def test_validation(self):
        with self.assertRaises(HandoffError):
            self.h.create({"task": ""})
        with self.assertRaises(HandoffError):
            self.h.create({"task": "x", "status": "weird"})
        with self.assertRaises(HandoffError):
            self.h.update(self.hid, {"nonsense": 1})

    def test_yaml_export(self):
        y = to_yaml({"a": [1, {"b": "c: d"}], "e": None})
        self.assertIn('b: "c: d"', y)
        self.assertIn("e: null", y)


def rpc(server, method, params=None, id=1):
    return server.handle({"jsonrpc": "2.0", "id": id, "method": method, "params": params or {}})


def call(server, tool, **args):
    r = rpc(server, "tools/call", {"name": tool, "arguments": args})
    return r["result"]


class McpTest(RepoTestCase):
    def setUp(self):
        super().setUp()
        self.server = Server(self.ws)
        rpc(self.server, "initialize", {"protocolVersion": "2025-06-18", "clientInfo": {"name": "agent-x"}})

    def test_initialize_and_list(self):
        r = rpc(self.server, "initialize", {"protocolVersion": "1999-01-01"})["result"]
        self.assertEqual(r["protocolVersion"], "2025-06-18")
        tools = rpc(self.server, "tools/list")["result"]["tools"]
        names = {t["name"] for t in tools}
        for required in ("get_project_overview", "get_blueprint", "search_graph", "trace_dataflow",
                         "trace_dependencies", "analyze_impact", "get_design_decisions", "build_task_context",
                         "create_handoff", "resume_handoff", "verify_claims", "get_index_status",
                         "add_knowledge_node", "add_relationship", "create_map", "plan_implementation",
                         "get_design_patterns", "get_analysis_tasks", "submit_analysis", "classify"):
            self.assertIn(required, names)
        writes = {t["name"] for t in tools if not t["annotations"]["readOnlyHint"]}
        self.assertIn("create_handoff", writes)
        self.assertNotIn("search_graph", writes)
        self.assertEqual(len(names), len(TOOLS))

    def test_every_read_tool_schema_is_valid_object(self):
        for t in TOOLS:
            self.assertEqual(t.schema["type"], "object")
            self.assertFalse(t.schema.get("additionalProperties", True))

    def test_read_tools(self):
        ov = call(self.server, "get_project_overview")
        self.assertFalse(ov["isError"])
        self.assertTrue(ov["structuredContent"]["subsystems"])
        imp = call(self.server, "analyze_impact", target="IndexRepository.find_all")["structuredContent"]
        self.assertIn("endpoints", imp["affected"])
        df = call(self.server, "trace_dataflow", source="api:POST /documents")["structuredContent"]
        self.assertTrue(df["flows"])
        bp = call(self.server, "get_blueprint", section="decisions", format="markdown")["structuredContent"]
        self.assertIn("ADR 0001", bp["markdown"])
        q = call(self.server, "query_graph", kinds=["endpoint"], edge_types=["routes_to"])["structuredContent"]
        self.assertEqual(q["count"], 2)
        self.assertEqual(q["edge_count"], 2)
        src = call(self.server, "get_source", ref="select_candidates")["structuredContent"]
        self.assertIn("def select_candidates", src["code"])
        vc = call(self.server, "verify_claims", claims=[
            {"claim": "service reads repository", "source": "SearchService.search", "target": "IndexRepository.find_all"},
            {"claim": "bogus", "source": "SearchService", "target": "EmbeddingAdapter", "relationship": "writes"},
        ])["structuredContent"]["results"]
        self.assertEqual(vc[0]["verdict"], "supported")
        self.assertIn(vc[1]["verdict"], ("weakly_supported", "contradicted_or_unknown"))

    def test_write_tools_and_provenance(self):
        f = call(self.server, "record_finding", claim="search allocates per request",
                 citations=["searchapp/api/routes.py:24"])["structuredContent"]
        self.assertEqual(f["status"], "verified")
        node = self.ws.store.node(f["id"])
        self.assertEqual(node["attrs"]["agent"], "agent-x")
        unsupported = call(self.server, "record_finding", claim="no evidence")["structuredContent"]
        self.assertEqual(unsupported["status"], "unsupported")
        bad = call(self.server, "add_knowledge_node", kind="function", name="evil")
        self.assertTrue(bad["isError"])
        m = call(self.server, "create_map", title="Write path",
                 nodes=["api:POST /documents", "add_document", "SearchService.index_document",
                        "IndexRepository.add"])["structuredContent"]
        self.assertEqual(m["edges"], 3)
        self.assertIn("flowchart", call(self.server, "get_map", map="Write path")["structuredContent"]["mermaid"])
        a = call(self.server, "annotate_node", node="select_candidates", concept="vector candidate selection")
        self.assertEqual(a["structuredContent"]["concept"], "concept:vector_candidate_selection")

    def test_security_and_validation_errors(self):
        r = call(self.server, "get_source", ref="../outside.txt")
        self.assertTrue(r["isError"])
        r = call(self.server, "search_graph", query=5)
        self.assertTrue(r["isError"])
        r = call(self.server, "search_graph", query="x", surprise=True)
        self.assertTrue(r["isError"])
        self.assertEqual(rpc(self.server, "nope")["error"]["code"], -32601)
        self.assertIsNone(self.server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}))

    def test_handoff_between_agents(self):
        c = call(self.server, "create_handoff", handoff=HANDOFF)["structuredContent"]
        other = Server(self.ws)
        rpc(other, "initialize", {"clientInfo": {"name": "agent-y"}})
        r = call(other, "resume_handoff", handoff_id=c["id"], claim=True, budget=1200)["structuredContent"]
        self.assertEqual(r["claimed_by"], "agent-y")
        stale = call(self.server, "update_handoff", handoff_id=c["id"], patch={"next_action": "x"},
                     expected_version=1)
        self.assertTrue(stale["isError"])
        self.assertTrue(stale["structuredContent"]["conflict"])

    def test_external_specialist_loop(self):
        t = call(self.server, "get_analysis_tasks", sections=["patterns"])["structuredContent"]
        task = next(x for x in t["tasks"] if x["specialist"] == "patterns")
        result = {"summary": "Confirmed strategy.", "claims": [
            {"claim": "Ranker is a strategy interface", "kind": "pattern",
             "citations": ["symbol:searchapp.domain.ranking:Ranker"], "confidence": 0.9}]}
        sub = call(self.server, "submit_analysis", task_id=task["id"], result=result)["structuredContent"]
        self.assertTrue(sub["accepted"])
        self.assertEqual(self.ws.store.agent_task(task["id"])["status"], "completed")
        bp = json.loads((self.ws.blueprint_dir / "blueprint.json").read_text())
        self.assertEqual(bp["sections"]["patterns"]["synthesis"]["summary"], "Confirmed strategy.")

    def test_resources_and_prompts(self):
        call(self.server, "generate_blueprint", sections=["architecture"])
        res = rpc(self.server, "resources/list")["result"]["resources"]
        uris = {r["uri"] for r in res}
        self.assertIn("promapkin://blueprint/architecture", uris)
        text = rpc(self.server, "resources/read", {"uri": "promapkin://blueprint/architecture"})["result"]
        self.assertIn("Architecture map", text["contents"][0]["text"])
        p = rpc(self.server, "prompts/get", {"name": "investigate", "arguments": {"issue": "slow search"}})
        self.assertIn("slow search", p["result"]["messages"][0]["content"]["text"])

    def test_stdio_transport(self):
        lines = [json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}),
                 "not json",
                 json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                             "params": {"name": "get_index_status", "arguments": {}}})]
        out = io.StringIO()
        serve(self.ws, io.StringIO("\n".join(lines) + "\n"), out)
        replies = [json.loads(l) for l in out.getvalue().splitlines()]
        self.assertEqual(replies[0]["id"], 1)
        self.assertEqual(replies[1]["error"]["code"], -32700)
        self.assertTrue(replies[2]["result"]["structuredContent"]["index_fresh"])


if __name__ == "__main__":
    unittest.main()
