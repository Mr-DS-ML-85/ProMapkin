"""Coverage for the v0.2 work: incremental resolution, multi-language parsers, Rust crate paths,
TypeScript bridge, runtime tracing, dynamic wiring, graph insights and the CLI additions."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from tests.helpers import RepoTestCase, git
from promapkin import toolchain
from promapkin.config import Workspace
from promapkin.extract import generic_lexical, python_ext
from promapkin.graph import Graph
from promapkin.scanner import scan
from promapkin.store import Store


def resolved_edges(ws: Workspace) -> set[tuple]:
    return {(e["src"], e["dst"], e["type"], e["evidence"]) for e in ws.store.edges(
        where="json_extract(attrs, '$.resolved') = 1 AND json_extract(attrs, '$.disambiguation') IS NULL")}


class IncrementalResolutionTest(RepoTestCase):
    def test_incremental_equals_full(self):
        self.edit("searchapp/storage/index_repository.py", "def find_all(self)", "def load_all(self)")
        self.edit("searchapp/services/search_service.py", "self.repository.find_all()", "self.repository.load_all()")
        (self.root / "searchapp" / "extra.py").write_text(
            "from searchapp.domain.ranking import select_candidates\n\ndef use():\n    return select_candidates([], 1)\n")
        r = scan(self.ws)
        self.assertTrue(r["resolution"].startswith("incremental"))
        incremental = resolved_edges(self.ws)
        r = scan(self.ws, full=True)
        self.assertEqual(r["resolution"], "full")
        self.assertEqual(incremental, resolved_edges(self.ws))
        self.assertIn(("symbol:searchapp.services.search_service:SearchService.search",
                       "symbol:searchapp.storage.index_repository:IndexRepository.load_all", "calls",
                       "static_resolved"), incremental)

    def test_timings_and_optional_pass_failures_do_not_lose_the_index(self):
        from unittest import mock
        self.edit("searchapp/cli.py", "def main", "def main2")
        with mock.patch("promapkin.scanner.graph_ml_pass", side_effect=RuntimeError("boom")):
            r = scan(self.ws)
        self.assertTrue(any("graph ML pass skipped" in w for w in r["warnings"]))
        self.assertIsNotNone(self.ws.store.node("symbol:searchapp.cli:main2"))
        self.assertIn("resolve", r["timings"])


class GraphInsightsTest(RepoTestCase):
    def test_trace_falls_back_to_structural(self):
        g = Graph(self.ws.store)
        r = g.trace("SearchService", "IndexRepository")
        self.assertTrue(r["found"])
        self.assertIn(r["mode"], ("flow", "structural"))
        self.assertTrue(r["paths"][0]["summary"].startswith("SearchService"))
        self.assertEqual(g.trace("SearchService", "IndexRepository", mode="related")["mode"], "related")

    def test_hubs_communities_surprises_html(self):
        from promapkin.service import Service
        svc = Service(self.ws)
        hubs = svc.hubs(5)["hubs"]
        self.assertTrue(hubs)
        self.assertTrue(all(h["fan_in"] >= hubs[-1]["fan_in"] for h in hubs))
        self.assertGreater(svc.communities()["count"], 0)
        sp = svc.surprises()
        self.assertIn("observed evidence only", sp["policy"])
        for c in sp["connections"]:
            self.assertIn(c["evidence"], ("ast", "static_resolved", "lexical", "runtime", "manifest"))
        out = svc.export_html()
        html = (self.root / out["path"]).read_text()
        self.assertIn("const D=", html)
        self.assertGreater(out["nodes"], 5)

    def test_resolve_prefers_production_top_level(self):
        (self.root / "tests" / "test_dup.py").write_text("class SearchService:\n    pass\n")
        scan(self.ws)
        self.assertEqual(Graph(self.ws.store).resolve("SearchService"),
                         "symbol:searchapp.services.search_service:SearchService")


class MultiLanguageTest(unittest.TestCase):
    RUST = textwrap.dedent("""\
        use crate::builder::Arg;
        pub struct Command { name: String }
        pub trait Render { fn render(&self) -> String; }
        impl Command {
            pub fn new(name: &str) -> Self { Command { name: name.into() } }
            pub fn arg(self, a: Arg) -> Self { self.push(a); self }
        }
        impl Render for Command { fn render(&self) -> String { format_name(&self.name) } }
        fn format_name(n: &str) -> String { n.to_string() }
        """)

    def test_lexical_fallback_rust_go_java(self):
        f = generic_lexical.extract("src/command.rs", self.RUST, "rust")
        names = {n["name"] for n in f.nodes}
        self.assertTrue({"Command", "new", "arg", "format_name"} <= names)
        go = "package store\nimport \"github.com/x/app/db\"\ntype Repo struct{}\nfunc (r *Repo) Find() { db.Query() }\n"
        f = generic_lexical.extract("internal/store/repo.go", go, "go")
        self.assertIn("symbol:internal/store/repo:Repo.Find", {n["id"] for n in f.nodes})
        self.assertIn("github.com/x/app/db", {r["target"] for r in f.refs if r["type"] == "imports"})
        java = "package a;\nimport a.db.Db;\npublic class Repo {\n  public void find() { db.query(); }\n}\n"
        f = generic_lexical.extract("src/a/Repo.java", java, "java")
        self.assertIn("symbol:src/a/Repo:Repo.find", {n["id"] for n in f.nodes})

    @unittest.skipUnless(toolchain.tree_sitter_available(), "tree-sitter not installed (pip install promapkin[parsers])")
    def test_tree_sitter_rust_impls_and_go_receivers(self):
        from promapkin.extract import treesitter_ext
        f = treesitter_ext.extract("src/command.rs", self.RUST, "rust")
        ids = {n["id"] for n in f.nodes}
        self.assertIn("symbol:src/command:Command.new", ids)
        self.assertIn("symbol:src/command:Command.render", ids)
        self.assertIn(("symbol:src/command:Command", "Render", "implements"),
                      {(r["src"], r["target"], r["type"]) for r in f.refs})
        go = "package store\ntype Repo struct{}\nfunc (r *Repo) Find(id int) { r.helper() }\n"
        f = treesitter_ext.extract("store/repo.go", go, "go")
        self.assertIn("symbol:store/repo:Repo.Find", {n["id"] for n in f.nodes})


class RustWorkspaceTest(unittest.TestCase):
    def test_crate_paths_and_type_member_calls(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            (tmp / "core" / "src" / "builder").mkdir(parents=True)
            (tmp / "app" / "src").mkdir(parents=True)
            (tmp / "core" / "Cargo.toml").write_text('[package]\nname = "my-core"\nversion = "0.1.0"\n')
            (tmp / "app" / "Cargo.toml").write_text('[package]\nname = "app"\nversion = "0.1.0"\n')
            (tmp / "core" / "src" / "lib.rs").write_text("pub mod builder;\npub use builder::command::Command;\n")
            (tmp / "core" / "src" / "builder" / "mod.rs").write_text("pub mod command;\n")
            (tmp / "core" / "src" / "builder" / "command.rs").write_text(
                "pub struct Command { n: u8 }\nimpl Command {\n    pub fn new() -> Command { Command { n: 0 } }\n}\n")
            (tmp / "app" / "src" / "main.rs").write_text(
                "use my_core::builder::command::Command;\nfn main() {\n    let c = Command::new();\n}\n")
            git(tmp, "init", "-q")
            git(tmp, "add", "-A")
            git(tmp, "commit", "-qm", "x")
            ws = Workspace(tmp)
            ws.init()
            scan(ws)
            imports = {(e["src"], e["dst"]) for e in ws.store.edges(types=["imports"])}
            self.assertIn(("module:app/src/main", "module:core/src/builder/command"), imports)
            calls = {(e["src"], e["dst"]) for e in ws.store.edges(types=["calls"])}
            self.assertIn(("symbol:app/src/main:main", "symbol:core/src/builder/command:Command.new"), calls)
            ws.close()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


@unittest.skipUnless(toolchain.typescript_available(), "TypeScript toolchain not installed (promapkin parsers install)")
class TypeScriptBridgeTest(unittest.TestCase):
    def test_type_resolved_calls(self):
        from promapkin.extract import ts_ext
        tmp = Path(tempfile.mkdtemp())
        try:
            (tmp / "src").mkdir()
            (tmp / "src" / "repo.ts").write_text("export interface Repo { findAll(): string[] }\n"
                                                 "export class SqlRepo implements Repo { findAll() { return []; } }\n")
            (tmp / "src" / "svc.ts").write_text("import { Repo } from './repo';\n"
                                                "export class Svc { constructor(private repo: Repo) {}\n"
                                                "  list() { return this.repo.findAll(); } }\n")
            out = ts_ext.extract_batch(tmp, ["src/repo.ts", "src/svc.ts"])
            refs = {(r["src"], r["target"]) for r in out["src/svc.ts"].refs if r["type"] == "calls"}
            self.assertIn(("symbol:src/svc:Svc.list", "ID:symbol:src/repo:Repo.findAll"), refs)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_object_literal_container_leaves_no_dangling_edges(self):
        from promapkin.extract import ts_ext
        tmp = Path(tempfile.mkdtemp())
        try:
            (tmp / "background.js").write_text(
                "const api = {\n  attach() { return 1; },\n  health() { return 2; },\n};\n"
                "export default api;\n")
            out = ts_ext.extract_batch(tmp, ["background.js"])
            f = out["background.js"]
            # the scanner creates the `file:` node before persisting facts
            ids = {n["id"] for n in f.nodes} | {"file:background.js"}
            for e in f.edges:
                self.assertIn(e["src"], ids, f"dangling edge src: {e}")
                self.assertIn(e["dst"], ids, f"dangling edge dst: {e}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class ContainerGuardTest(unittest.TestCase):
    """`_to_facts` must never edge to a container that has no declaration (e.g. `const api = {…}`),
    and a scan must drop edges whose endpoints vanished."""

    def test_undeclared_container_falls_back_to_module(self):
        from promapkin.extract.ts_ext import _to_facts
        rec = {"module": "bg", "lines": 10,
               "decls": [
                   {"kind": "class", "qual": "Widget", "name": "Widget", "line_start": 1, "line_end": 2},
                   {"kind": "method", "qual": "api.attach", "name": "attach", "line_start": 3, "line_end": 4},
                   {"kind": "method", "qual": "api.health", "name": "health", "line_start": 5, "line_end": 6},
                   {"kind": "method", "qual": "Widget.render", "name": "render", "line_start": 7, "line_end": 8},
               ],
               "imports": [], "calls": [], "inherits": [], "routes": [], "env": [], "events": []}
        f = _to_facts("bg/background.js", rec)
        # the scanner creates the `file:` node before persisting facts
        ids = {n["id"] for n in f.nodes} | {"file:bg/background.js"}
        for e in f.edges:
            self.assertIn(e["src"], ids, f"dangling edge src: {e}")
            self.assertIn(e["dst"], ids, f"dangling edge dst: {e}")
        edges = {(e["src"], e["dst"], e["type"]) for e in f.edges}
        self.assertIn(("module:bg", "symbol:bg:api.attach", "defines"), edges)
        self.assertIn(("symbol:bg:Widget", "symbol:bg:Widget.render", "contains"), edges)

    def test_scan_prunes_edges_to_missing_nodes(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            (tmp / "app.py").write_text("def main():\n    return 1\n")
            git(tmp, "init", "-q")
            git(tmp, "add", "-A")
            git(tmp, "commit", "-qm", "x")
            ws = Workspace(tmp)
            ws.init()
            scan(ws)
            with ws.store.transaction():
                ws.store.add_edge("symbol:app:ghost", "symbol:app:main", "contains",
                                  evidence="ast", owner_file="app.py")
            count = "SELECT COUNT(*) FROM edges WHERE src NOT IN (SELECT id FROM nodes) " \
                    "OR dst NOT IN (SELECT id FROM nodes)"
            self.assertGreater(ws.store.conn.execute(count).fetchone()[0], 0)  # precondition
            scan(ws)  # incremental scan with no file changes
            self.assertEqual(ws.store.conn.execute(count).fetchone()[0], 0)
            ws.close()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class DynamicWiringAndRuntimeTest(RepoTestCase):
    def test_static_wiring_detection(self):
        src = textwrap.dedent("""\
            import importlib
            import pytest
            from fastapi import Depends

            @pytest.fixture
            def db():
                return 1

            def auth():
                return True

            def handler(user=Depends(auth)):
                bus.emit("user.created", user)

            def on_created(u):
                pass

            bus.on("user.created", on_created)
            plugin = getattr(registry, "load_plugin")
            mod = importlib.import_module("searchapp.plugins")
            """)
        f = python_ext.extract("app/wiring.py", src)
        refs = {(r["type"], r["target"], r["context"].get("wiring")) for r in f.refs}
        self.assertIn(("invokes", "auth", "dependency injection (Depends)"), refs)
        self.assertIn(("invokes", "on_created", "event handler"), refs)
        self.assertIn(("calls", "registry.load_plugin", "reflection (getattr)"), refs)
        self.assertIn(("imports", "searchapp.plugins", "dynamic import"), refs)
        edges = {(e["src"], e["dst"], e["type"]) for e in f.edges}
        self.assertIn(("symbol:app.wiring:handler", "event:user.created", "produces"), edges)
        t = python_ext.extract("tests/test_x.py", "def test_a(db):\n    assert db\n")
        self.assertIn(("invokes", "db"), {(r["type"], r["target"]) for r in t.refs if r["context"].get("fixture")})

    def test_runtime_trace_finds_dynamic_links(self):
        from promapkin import runtime
        (self.root / "searchapp" / "plugins.py").write_text(
            "import importlib\n\n\ndef load(name):\n    mod = importlib.import_module('searchapp.domain.ranking')\n"
            "    return getattr(mod, name)()\n")
        (self.root / "tests" / "test_rt_flow.py").write_text(
            "from searchapp.plugins import load\nfrom searchapp.services.search_service import SearchService\n\n"
            "class Repo:\n    def find_all(self):\n        return [{'id': 'a', 'text': 'x x'}]\n\n"
            "def test_flow():\n    SearchService(Repo(), load('BM25Ranker')).search('x')\n\ntest_flow()\n")
        scan(self.ws)
        r = runtime.run(self.ws, [sys.executable, "-c", "import runpy, sys; sys.path.insert(0, '.'); "
                                  "runpy.run_path('tests/test_rt_flow.py', run_name='__main__')"])
        self.assertEqual(r["exit_code"], 0)
        s = self.ws.store
        rt = {(e["src"], e["dst"]) for e in s.edges(types=["calls"], where="evidence='runtime'")}
        # Dynamic dispatch: statically SearchService.search only reaches the abstract Ranker.score.
        self.assertIn(("symbol:searchapp.services.search_service:SearchService.search",
                       "symbol:searchapp.domain.ranking:BM25Ranker.score"), rt)
        self.assertGreater(r["dynamic_link_count"], 0)
        verif = {e["dst"] for e in s.edges(src="symbol:tests.test_rt_flow:test_flow", types=["verifies"])}
        self.assertIn("symbol:searchapp.domain.ranking:BM25Ranker.score", verif)
        # Changing traced code marks runtime evidence stale until re-traced.
        self.edit("searchapp/domain/ranking.py", "terms = query", "terms  = query")
        scan(self.ws)
        stale = s.edges(where="evidence='runtime' AND status='stale'")
        self.assertTrue(stale)


class MigrationAndCliTest(unittest.TestCase):
    def test_fts_migration_rebuilds_rowid_index(self):
        tmp = Path(tempfile.mkdtemp()) / "index.db"
        st = Store(tmp)
        with st.transaction():
            st.upsert_node("symbol:a:Thing", "class", "Thing", evidence="ast", text="does things")
        st.set_meta("fts_rowid", "0")
        st.close()
        st = Store(tmp)
        self.assertEqual(st.get_meta("fts_rowid"), "1")
        self.assertEqual(st.search("Thing")[0][0], "symbol:a:Thing")
        with st.transaction():
            st.delete_nodes("id=?", ("symbol:a:Thing",))
        self.assertEqual(st.search("Thing"), [])
        st.close()

    def test_doctor_and_auth_status(self):
        out = subprocess.run([sys.executable, "-m", "promapkin", "--json", "--root", tempfile.mkdtemp(), "doctor"],
                             capture_output=True, text=True, env={**os.environ, "PROMAPKIN_CONFIG_HOME":
                                                                  tempfile.mkdtemp()})
        checks = {c["check"]: c for c in json.loads(out.stdout)["checks"]}
        self.assertTrue(checks["python"]["ok"])
        self.assertIn("anthropic LLM", checks)
        out = subprocess.run([sys.executable, "-m", "promapkin", "auth", "status"], capture_output=True, text=True,
                             env={**os.environ, "PROMAPKIN_CONFIG_HOME": tempfile.mkdtemp(),
                                  "ANTHROPIC_API_KEY": "sk-ant-abcdefghijklmnop"})
        self.assertIn("anthropic  configured", out.stdout)
        self.assertNotIn("abcdefghijklmnop", out.stdout)


if __name__ == "__main__":
    unittest.main()
