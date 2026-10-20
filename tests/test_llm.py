import contextlib
import importlib.util
import io
import json
import os
import stat
import tempfile
import unittest
from unittest import mock

from tests.helpers import RepoTestCase
from promapkin import credentials
from promapkin.llm.clients import strict_schema
from promapkin.llm.engine import redact

HAS_SDKS = importlib.util.find_spec("anthropic") is not None and importlib.util.find_spec("openai") is not None


class OfflineLLMTest(unittest.TestCase):
    def test_redaction(self):
        text = 'key = "sk-ant-abcdefghijklmnopqrstuvwxyz123456" and AKIAABCDEFGHIJKLMNOP and password="hunter22"'
        clean, n = redact(text)
        self.assertEqual(n, 3)
        self.assertNotIn("hunter22", clean)
        self.assertNotIn("AKIA", clean)

    def test_strict_schema(self):
        s = strict_schema({"type": "object", "properties": {"a": {"type": "string", "maxLength": 3},
                                                            "b": {"type": "object", "properties": {"c": {"type": "integer", "minimum": 0}}}}})
        self.assertEqual(s["required"], ["a", "b"])
        self.assertFalse(s["additionalProperties"])
        self.assertNotIn("maxLength", s["properties"]["a"])
        self.assertEqual(s["properties"]["b"]["required"], ["c"])

    def test_credentials_store(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, {"PROMAPKIN_CONFIG_HOME": d}, clear=False):
            os.environ.pop("OPENAI_API_KEY", None)
            path = credentials.set_key("openai", "sk-test-1234567890", "http://localhost:1234/v1")
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            self.assertEqual(credentials.resolve("openai"), ("sk-test-1234567890", str(path)))
            with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "from-env"}):
                self.assertEqual(credentials.resolve("openai")[0], "from-env")
            self.assertEqual(credentials.resolve("openai", "explicit")[0], "explicit")
            self.assertEqual(credentials.stored_base_url("openai"), "http://localhost:1234/v1")
            self.assertTrue(credentials.remove("openai"))
            self.assertEqual(credentials.resolve("openai")[1], "none")

    def test_missing_sdk_is_reported(self):
        from promapkin.cli import main
        with mock.patch("importlib.util.find_spec", return_value=None), \
                mock.patch.dict("sys.modules", {"openai": None}):
            err = io.StringIO()
            with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
                code = main(["--root", tempfile.mkdtemp(), "--llm", "openai", "--base-url", "http://x", "status"])
        self.assertEqual(code, 2)
        self.assertIn("LLM mode unavailable", err.getvalue())


@unittest.skipUnless(HAS_SDKS, "anthropic and openai SDKs not installed (pip install 'promapkin[llm]')")
class LLMModeTest(RepoTestCase):
    def setUp(self):
        super().setUp()
        from tests.fake_llm import FakeLLM
        self.fake = FakeLLM()
        self.env = mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test", "OPENAI_API_KEY": "sk-test",
                                                "PROMAPKIN_CONFIG_HOME": tempfile.mkdtemp()})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.fake.close()
        super().tearDown()

    def cli(self, *argv):
        from promapkin.cli import main
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(["--root", str(self.root), *argv])
        return code, out.getvalue(), err.getvalue()

    def test_anthropic_client_structured_and_text(self):
        from promapkin import llm
        engine = llm.make_engine(self.ws, provider="anthropic", base_url=self.fake.url)
        out = engine.classify_many("layer", ["app.storage.users repository", "app.api.routes"])
        self.assertEqual([o["label"] for o in out], ["persistence", "interface"])
        req = self.fake.requests[-1]
        self.assertEqual(req["body"]["model"], "claude-opus-5-5")
        self.assertEqual(req["body"]["output_config"]["format"]["type"], "json_schema")
        self.assertNotIn("fallbacks", req["body"])   # gateways (custom base_url) skip the first-party beta
        n = len(self.fake.requests)
        engine.classify_many("layer", ["app.storage.users repository", "app.api.routes"])
        self.assertEqual(len(self.fake.requests), n)   # cached: identical input never re-sent
        self.assertIn("Fake explanation", engine.explain("graph impact", {"x": 1}))
        self.assertTrue((self.ws.dir / "egress.log").exists())

    def test_openai_base_url_client(self):
        from promapkin import llm
        engine = llm.make_engine(self.ws, provider="openai", base_url=self.fake.url + "/v1", model="local-model")
        out = engine.expand_query("how are candidates picked")
        self.assertEqual(out["identifiers"], ["select_candidates"])
        req = self.fake.requests[-1]
        self.assertTrue(req["path"].endswith("/v1/chat/completions"))
        self.assertEqual(req["body"]["model"], "local-model")
        self.assertTrue(req["body"]["response_format"]["json_schema"]["strict"])
        self.assertIn("max_tokens", req["body"])   # compatible servers get max_tokens, not max_completion_tokens

    def test_llm_scan_replaces_regex_paths_and_grounds_facts(self):
        (self.root / "native").mkdir()
        (self.root / "native" / "lib.rs").write_text("fn parse_args() {\n    helper();\n}\nfn helper() {}\n")
        (self.root / "docs" / "notes.md").write_text("# Notes\n\n`SearchService` uses `IndexRepository` for reads.\n")
        with mock.patch("promapkin.toolchain.tree_sitter_available", return_value=False):
            code, out, err = self.cli("--llm", "anthropic", "--base-url", self.fake.url, "scan")
        self.assertEqual(code, 0, err)
        s = self.ws.store
        self.assertIsNotNone(s.node("symbol:native/lib:parse_args"))
        self.assertEqual(s.node("symbol:native/lib:parse_args")["evidence"], "llm_extraction")
        self.assertIsNone(s.node("symbol:native/lib:ghost_function"))       # ungrounded -> dropped
        calls = {(e["src"], e["dst"]) for e in s.edges(types=["calls"])}
        self.assertIn(("symbol:native/lib:parse_args", "symbol:native/lib:helper"), calls)
        layer = s.node("module:searchapp.storage.index_repository")["attrs"]["layer"]
        self.assertEqual(layer["tier"], "llm")
        claims = [e for e in s.edges(types=["depends_on"]) if e["attrs"].get("claim")]
        self.assertTrue(any("notes.md" in (e["file"] or "") for e in claims))
        self.assertIn("ungrounded fact(s) dropped", err)

    def test_query_with_explanation_and_rerank(self):
        self.cli("scan")
        code, out, err = self.cli("--llm", "openai", "--base-url", self.fake.url + "/v1", "graph", "query",
                                  "how are candidates picked")
        self.assertEqual(code, 0, err)
        self.assertIn("select_candidates", out)
        self.assertIn("Fake explanation from OpenAI-compatible server", out)

    def test_ask_runs_tool_loop(self):
        self.cli("scan")
        for provider, url in (("anthropic", self.fake.url), ("openai", self.fake.url + "/v1")):
            code, out, err = self.cli("--llm", provider, "--base-url", url, "--json", "ask", "what selects candidates?")
            self.assertEqual(code, 0, err)
            data = json.loads(out)
            self.assertEqual(data["steps"][0]["tool"], "search_graph")
            self.assertFalse(data["steps"][0]["error"])
            self.assertIn("Fake explanation", data["answer"])

    def test_blueprint_with_llm_specialists(self):
        self.cli("scan")
        code, out, err = self.cli("--llm", "anthropic", "--base-url", self.fake.url, "--no-explain", "blueprint",
                                  "generate", "--sections", "patterns", "verification")
        self.assertEqual(code, 0, err)
        bp = json.loads((self.ws.blueprint_dir / "blueprint.json").read_text())
        self.assertEqual(bp["sections"]["patterns"]["synthesis"]["summary"], "fake synthesis")
        self.assertEqual(bp["sections"]["patterns"]["classifier"]["mode"], "llm")

    def test_external_agent_provider_openai_with_base_url(self):
        self.cli("scan")
        self.ws.config.provider = "openai"
        self.ws.config.allow_remote = True
        self.ws.config.openai_base_url = self.fake.url + "/v1"
        self.ws.config.save(self.ws.config_path)
        code, out, err = self.cli("--json", "blueprint", "generate", "--sections", "risks")
        self.assertEqual(code, 0, err)
        self.assertEqual(json.loads(out)["provider"], "openai/gpt-5")

    def test_review_and_verify(self):
        self.cli("scan")
        self.edit("searchapp/domain/ranking.py", "limit * 4", "limit * 2")
        code, out, err = self.cli("--llm", "anthropic", "--base-url", self.fake.url, "--json", "review")
        self.assertEqual(code, 0, err)
        r = json.loads(out)
        self.assertIn("symbol:searchapp.domain.ranking:select_candidates", [c["id"] for c in r["changed_symbols"]])
        self.assertTrue(r["tests_to_run"])
        self.assertIn("llm_review", r)
        code, out, err = self.cli("--llm", "anthropic", "--base-url", self.fake.url, "--json", "verify")
        judged = json.loads(out)["documentation"]["llm_judged"]
        self.assertEqual(judged[0]["llm_verdict"]["verdict"], "contradicted")


if __name__ == "__main__":
    unittest.main()
