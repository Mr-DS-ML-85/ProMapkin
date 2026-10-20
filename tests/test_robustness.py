"""Regression tests for structured-output failures from OpenAI-compatible proxies and for index lock contention."""

import contextlib
import importlib.util
import io
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest import mock

from tests.helpers import RepoTestCase
from promapkin.extract import Facts
from promapkin.llm.clients import LLMClient, Turn, Usage, parse_structured, schema_errors
from promapkin.llm.engine import LLMEngine
from promapkin.llm.errors import LLMInvalidOutput, LLMUnavailable
from promapkin.store import Store, StoreBusy

HAS_OPENAI = importlib.util.find_spec("openai") is not None
DOC = LLMEngine.DOC_SCHEMA
VALID = {"claims": [], "decisions": [], "requirements": [], "code_mentions": ["Foo"]}


def empty_instance(schema):
    """The smallest value that satisfies a Promapkin schema."""
    kind = schema.get("type") or ("object" if "properties" in schema else "string")
    if "enum" in schema:
        return schema["enum"][0]
    if kind == "object":
        return {k: empty_instance(v) for k, v in schema.get("properties", {}).items()}
    return {"array": [], "string": "", "integer": 0, "number": 0, "boolean": False}[kind]


class ScriptedClient(LLMClient):
    """Replies with schema-valid empty data (or a scripted reply) and records whether the index's write
    transaction was open during each call."""
    provider = "openai"

    def __init__(self, ws=None, replies=None):
        super().__init__("scripted", Usage())
        self.ws, self.replies, self.calls = ws, list(replies or []), []

    def json(self, system, user, schema, *, max_tokens=16000, effort="medium", name="result"):
        locked = self.ws is not None and self.ws.store.conn.in_transaction
        self.calls.append({"task": name, "locked": locked, "system": system})
        self.usage.add(1, 1)
        if self.replies:
            reply = self.replies.pop(0)
            if isinstance(reply, Exception):
                raise reply
            return parse_structured(reply, schema)
        if name.startswith("classify:"):   # answer every item with the first label
            label = schema["properties"]["items"]["items"]["properties"]["label"]["enum"][0]
            return {"items": [{"index": it["index"], "label": label, "confidence": 0.9, "reason": "scripted"}
                              for it in json.loads(user)]}
        return empty_instance(schema)

    def chat(self, system, tools):
        client = self

        class _Chat:
            def user(self, text):
                pass

            def send(self, *, max_tokens=16000, effort="high"):
                client.calls.append({"task": "chat", "system": system})
                return Turn("the answer")

            def tool_results(self, results):
                pass
        return _Chat()


class ParseStructuredTest(unittest.TestCase):
    def kind(self, content, **kw):
        with self.assertRaises(LLMInvalidOutput) as cm:
            parse_structured(content, DOC, finish_reason="stop", **kw)
        return cm.exception.kind, str(cm.exception)

    def test_valid_reply(self):
        self.assertEqual(parse_structured(json.dumps(VALID), DOC), VALID)

    def test_failure_categories(self):
        body = json.dumps(VALID)
        self.assertEqual(self.kind("")[0], "empty")
        self.assertEqual(self.kind("  \n")[0], "empty")
        self.assertEqual(self.kind(body[:40])[0], "truncated")
        kind, msg = self.kind("```json\n" + body + "\n```")
        self.assertEqual(kind, "wrapped")
        self.assertIn("openai_structured_output", msg)        # actionable hint
        self.assertEqual(self.kind("Here is the analysis:\n" + body)[0], "wrapped")
        self.assertEqual(self.kind("<think>reading</think>\n" + body)[0], "wrapped")
        self.assertEqual(self.kind(body + "\nHope this helps!")[0], "wrapped")
        self.assertEqual(self.kind('{"claims": [,]}')[0], "malformed")
        self.assertEqual(self.kind("[1, 2]")[0], "not_object")
        kind, msg = self.kind('{"claims": [{"subject": "A"}]}')
        self.assertEqual(kind, "schema")
        self.assertIn("decisions is missing", msg)
        self.assertIn("finish_reason=stop", msg)

    def test_no_document_text_in_errors(self):
        secret_doc = "The private roadmap says we acquire ACME in May."
        _, msg = self.kind("Summary: " + secret_doc + json.dumps(VALID))
        self.assertNotIn("ACME", msg)

    def test_fence_only_when_allowed(self):
        body = json.dumps(VALID)
        self.assertEqual(parse_structured("```json\n" + body + "\n```", DOC, allow_fence=True), VALID)
        self.assertEqual(parse_structured("```\n" + body + "\n```", DOC, allow_fence=True), VALID)
        # One block only, nothing around it: prose plus a fence is still rejected.
        self.assertEqual(self.kind("Sure!\n```json\n" + body + "\n```", allow_fence=True)[0], "wrapped")

    def test_schema_errors(self):
        s = {"type": "object", "properties": {"n": {"type": "integer"}, "e": {"type": "string", "enum": ["a"]}}}
        self.assertEqual(schema_errors({"n": 1, "e": "a"}, s), [])
        self.assertTrue(schema_errors({"n": True, "e": "a"}, s))   # bool is not an integer
        self.assertTrue(schema_errors({"n": 1, "e": "b"}, s))
        self.assertTrue(schema_errors({"n": 1}, s))


@unittest.skipUnless(HAS_OPENAI, "openai SDK not installed (pip install 'promapkin[llm]')")
class OpenAIModesTest(unittest.TestCase):
    def run_mode(self, mode, content):
        from promapkin.llm.clients import OpenAIClient
        c = OpenAIClient("ds-web/deepseek-v4-pro", Usage(), base_url="http://localhost:20128/v1",
                         structured_output=mode)
        sent = {}

        def create(**kw):
            sent.update(kw)
            return NS(usage=None, choices=[NS(finish_reason="stop", message=NS(content=content, refusal=None))])
        c.client.chat.completions.create = create
        return c.json("Analyse.", "doc", DOC, name="enrich_document"), sent

    def test_json_schema_mode_is_strict(self):
        out, sent = self.run_mode("json_schema", json.dumps(VALID))
        self.assertEqual(out, VALID)
        self.assertTrue(sent["response_format"]["json_schema"]["strict"])
        with self.assertRaises(LLMInvalidOutput):
            self.run_mode("json_schema", "```json\n" + json.dumps(VALID) + "\n```")

    def test_json_object_and_prompt_modes_put_the_schema_in_the_prompt(self):
        out, sent = self.run_mode("json_object", json.dumps(VALID))
        self.assertEqual(sent["response_format"], {"type": "json_object"})
        self.assertIn('"code_mentions"', sent["messages"][0]["content"])
        out, sent = self.run_mode("prompt", "```json\n" + json.dumps(VALID) + "\n```")
        self.assertEqual(out, VALID)
        self.assertNotIn("response_format", sent)
        with self.assertRaises(LLMInvalidOutput) as cm:   # still validated against the schema
            self.run_mode("prompt", '{"claims": []}')
        self.assertEqual(cm.exception.kind, "schema")

    def test_unknown_mode_rejected(self):
        with self.assertRaises(LLMUnavailable):
            self.run_mode("loose", "{}")


class EngineRobustnessTest(RepoTestCase):
    scan_on_setup = False

    def engine(self, replies=None):
        client = ScriptedClient(self.ws, replies)
        return LLMEngine(self.ws, client), client

    def doc_facts(self):
        from promapkin.extract.docs_ext import extract
        text = (self.root / "docs" / "architecture.md").read_text()
        return text, extract("docs/architecture.md", text)

    def test_failed_enrichment_keeps_deterministic_facts(self):
        text, facts = self.doc_facts()
        before = (list(facts.nodes), list(facts.edges), list(facts.refs))
        self.assertTrue(any(r["type"] == "claims" for r in facts.refs))
        engine, _ = self.engine(["```json\n{}\n```"])
        with self.assertRaises(LLMInvalidOutput):
            engine.enrich_document("docs/architecture.md", text, facts)
        self.assertEqual((facts.nodes, facts.edges, facts.refs), before)
        self.assertEqual(engine.stats.fallbacks, 1)

    def test_failure_midway_rolls_back(self):
        text, facts = self.doc_facts()
        before = (list(facts.nodes), list(facts.edges), list(facts.refs))
        quote = next(line for line in text.splitlines() if len(line) > 20)
        reply = {"claims": [], "requirements": [], "code_mentions": [],
                 "decisions": [{"title": "T", "decision": "D", "rationale": "R", "status": "accepted",
                                "quote": quote}]}
        engine, _ = self.engine([json.dumps(reply)])
        # regex claims are removed before decisions are added: fail exactly there
        with mock.patch.object(Facts, "node", side_effect=RuntimeError("boom")), self.assertRaises(RuntimeError):
            engine.enrich_document("docs/architecture.md", text, facts)
        self.assertEqual((facts.nodes, facts.edges, facts.refs), before)

    def test_debug_log_is_opt_in_and_redacted(self):
        raw = 'Sure: {"claims": [], "token": "sk-ant-abcdefghijklmnopqrstuvwxyz123456"}'
        engine, _ = self.engine([raw])
        with self.assertRaises(LLMInvalidOutput):
            engine.call_json("enrich_document", "s", "u", DOC, subject="README.md")
        self.assertFalse((self.ws.dir / "llm-debug").exists())
        engine, _ = self.engine([raw])
        with mock.patch.dict(os.environ, {"PROMAPKIN_LLM_DEBUG": "1"}), self.assertRaises(LLMInvalidOutput) as cm:
            engine.call_json("enrich_document", "s", "u2", DOC, subject="README.md")
        self.assertIn("llm-debug", str(cm.exception))
        logged = json.loads(next((self.ws.dir / "llm-debug").iterdir()).read_text())
        self.assertEqual((logged["kind"], logged["subject"]), ("wrapped", "README.md"))
        self.assertNotIn("sk-ant-abcdef", logged["raw_reply"])

    def test_invalid_replies_are_never_cached_and_bad_cache_rows_are_ignored(self):
        engine, client = self.engine(["", json.dumps(VALID)])
        with self.assertRaises(LLMInvalidOutput):
            engine.call_json("enrich_document", "s", "u", DOC)
        self.assertEqual(engine.call_json("enrich_document", "s", "u", DOC), VALID)
        self.assertEqual(len(client.calls), 2)
        self.ws.store.conn.execute("UPDATE llm_cache SET response='{\"claims\": 1}'")
        engine.call_json("enrich_document", "s", "u", DOC)          # stale/invalid row: asked again
        self.assertEqual(len(client.calls), 3)

    def test_cache_only_never_calls_out(self):
        engine, client = self.engine()
        with engine.cache_only(), self.assertRaises(LLMUnavailable):
            engine.call_json("expand_query", "s", "u", DOC)
        self.assertEqual(client.calls, [])

    def test_ask_keeps_the_question_in_the_system_prompt(self):
        from promapkin.llm.agent import ask
        from promapkin.scanner import scan
        scan(self.ws)
        engine, client = self.engine()
        out = ask(self.ws, engine, "what are the fragmented parts of the code?")
        self.assertEqual(out["answer"], "the answer")
        self.assertIn("what are the fragmented parts of the code?", client.calls[0]["system"])


class NoNetworkUnderLockTest(RepoTestCase):
    scan_on_setup = False

    def test_llm_scan_and_blueprint_never_call_out_while_holding_the_write_lock(self):
        from promapkin.agents import orchestrator
        from promapkin.scanner import scan
        client = ScriptedClient(self.ws)
        self.ws.llm = LLMEngine(self.ws, client)
        with mock.patch("promapkin.toolchain.tree_sitter_available", return_value=False):
            (self.root / "native").mkdir()
            (self.root / "native" / "lib.rs").write_text("fn parse() {\n    helper();\n}\nfn helper() {}\n")
            r = scan(self.ws)
        tasks = {c["task"] for c in client.calls}
        self.assertIn("enrich_document", tasks)
        self.assertIn("extract_code", tasks)
        self.assertTrue(any(t.startswith("classify:") for t in tasks))
        self.assertEqual([c["task"] for c in client.calls if c["locked"]], [])
        self.assertEqual(r["classified"]["classifier"], f"llm:{self.ws.llm.label}")
        layer = self.ws.store.node("module:searchapp.storage.index_repository")["attrs"]["layer"]
        self.assertEqual(layer["tier"], "llm")

        client.calls.clear()
        orchestrator.run(self.ws, sections=["patterns", "roadmap"], goal="add caching")
        self.assertIn("classify:design_pattern", {c["task"] for c in client.calls})
        self.assertIn("refine_roadmap", {c["task"] for c in client.calls})
        self.assertEqual([c["task"] for c in client.calls if c["locked"]], [])


HOLD = """
import sys, time
from promapkin.store import Store
s = Store(sys.argv[1])
with s.transaction():
    s.set_meta("holder", "partial")
    print("locked", flush=True)
    time.sleep(float(sys.argv[2]))
    if sys.argv[3] == "fail":
        raise SystemExit(1)
"""


class LockContentionTest(RepoTestCase):
    def hold(self, seconds, outcome="ok"):
        env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])}
        p = subprocess.Popen([sys.executable, "-c", HOLD, str(self.ws.db_path), str(seconds), outcome],
                             stdout=subprocess.PIPE, text=True, env=env)
        self.addCleanup(p.stdout.close)
        self.assertEqual(p.stdout.readline().strip(), "locked")
        return p

    def test_busy_writer_gives_a_clear_error_and_leaves_the_index_consistent(self):
        p = self.hold(3)
        store = Store(self.ws.db_path, busy_timeout=0.3)
        with self.assertRaises(StoreBusy) as cm, store.transaction():
            store.set_meta("never", "written")
        self.assertIn("another promapkin process", str(cm.exception))
        self.assertEqual(store._depth, 0)
        p.wait()
        with store.transaction():     # usable again once the other writer is done
            store.set_meta("after", "1")
        self.assertIsNone(store.get_meta("never"))
        self.assertEqual(store.conn.execute("PRAGMA integrity_check").fetchone()[0], "ok")
        store.close()

    def test_waits_for_a_short_writer_and_sees_rollbacks(self):
        p = self.hold(0.5, "fail")
        store = Store(self.ws.db_path, busy_timeout=10)
        with store.transaction():
            store.set_meta("second", "1")
        p.wait()
        self.assertEqual(p.returncode, 1)
        self.assertIsNone(store.get_meta("holder"))      # the failed writer rolled back
        self.assertEqual(store.get_meta("second"), "1")
        store.close()

    def test_error_inside_transaction_rolls_back(self):
        s = self.ws.store
        with self.assertRaises(ValueError), s.transaction():
            s.set_meta("x", "1")
            raise ValueError
        self.assertIsNone(s.get_meta("x"))
        self.assertFalse(s.conn.in_transaction)

    def test_cli_reports_busy_index(self):
        from promapkin.cli import main
        self.ws.config.db_busy_timeout = 0.3
        self.ws.config.save(self.ws.config_path)
        p = self.hold(3)
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            code = main(["--root", str(self.root), "blueprint", "generate", "--sections", "risks"])
        p.wait()
        self.assertEqual(code, 3)
        self.assertIn("index busy", err.getvalue())
        self.assertNotIn("Traceback", err.getvalue())


if __name__ == "__main__":
    unittest.main()
