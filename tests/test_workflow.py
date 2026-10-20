"""Workflow features: parallel scans, git hooks, watch mode, PR dashboard, source ingestion, export/merge,
wiki, call flow, hyperedges and duplicate-logic detection."""

import json
import os
import re
import threading
import time
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest import mock

from tests.helpers import RepoTestCase, git, make_repo

ROOT = Path(__file__).resolve().parents[1]


def graph_signature(store):
    nodes = {(n["id"], n["kind"]) for n in store.nodes()}
    edges = {(e["src"], e["dst"], e["type"], e["evidence"]) for e in store.edges()}
    return nodes, edges


class ParallelScanTest(RepoTestCase):
    scan_on_setup = False

    def test_parallel_extraction_matches_sequential(self):
        from promapkin import scanner
        from promapkin.config import Workspace
        scanner.scan(self.ws)
        sequential = graph_signature(self.ws.store)
        other = make_repo()
        ws2 = Workspace(other)
        ws2.init()
        with mock.patch.object(scanner, "PARALLEL_MIN_FILES", 1), mock.patch.dict(os.environ,
                                                                                  {"PROMAPKIN_WORKERS": "2"}):
            r = scanner.scan(ws2, full=True)
        self.assertTrue(any("processes" in k for k in r["timings"]), r["timings"])
        nodes, edges = graph_signature(ws2.store)
        self.assertEqual(nodes, sequential[0])
        self.assertEqual(edges, sequential[1])
        ws2.close()


class HooksAndWatchTest(RepoTestCase):
    def test_hooks_install_keep_existing_and_reindex_after_commit(self):
        from promapkin import autoupdate
        hooks = autoupdate.hooks_dir(self.root)
        hooks.mkdir(parents=True, exist_ok=True)
        (hooks / "post-commit").write_text("#!/bin/sh\necho mine\n")
        r = autoupdate.install_hooks(self.root)
        self.assertEqual({h["hook"] for h in r["hooks"]}, set(autoupdate.HOOKS))
        text = (hooks / "post-commit").read_text()
        self.assertIn("echo mine", text)
        self.assertEqual(autoupdate.install_hooks(self.root)["hooks"][0]["status"], "unchanged")
        self.assertEqual(autoupdate.hook_status(self.root)["installed"], list(autoupdate.HOOKS))
        before = self.ws.store.get_meta("indexed_revision")
        self.edit("searchapp/domain/ranking.py", "limit * 4", "limit * 5")
        env = {"PYTHONPATH": str(ROOT)}
        with mock.patch.dict(os.environ, env):
            git(self.root, "commit", "-qam", "tweak")
        deadline = time.time() + 60
        from promapkin.store import Store
        while time.time() < deadline:
            s = Store(self.ws.db_path)
            rev = s.get_meta("indexed_revision")
            s.close()
            if rev != before and rev and "+dirty" not in rev:
                break
            time.sleep(0.5)
        self.assertNotEqual(rev, before, (self.ws.dir / "hook.log").read_text() if (self.ws.dir / "hook.log").exists()
                            else "no hook log")
        autoupdate.uninstall_hooks(self.root)
        self.assertEqual((hooks / "post-commit").read_text().strip(), "#!/bin/sh\necho mine")
        self.assertFalse((hooks / "post-merge").exists())

    def test_watch_reindexes_after_changes_settle(self):
        from promapkin import autoupdate
        events = []
        done = threading.Event()

        def changer():
            time.sleep(0.4)
            (self.root / "searchapp" / "new_module.py").write_text("def brand_new():\n    return 1\n")
        threading.Thread(target=changer).start()
        n = autoupdate.watch(self.ws, interval=0.2, debounce=0.3, on_event=events.append, max_rounds=25,
                             stop=lambda: done.is_set() or any(e["event"] == "rebuilt" for e in events))
        self.assertEqual(n, 1, events)
        self.assertIsNotNone(self.ws.store.node("symbol:searchapp.new_module:brand_new"))


class PullRequestTest(RepoTestCase):
    def branch(self, name, edits, message):
        git(self.root, "checkout", "-q", "-b", name)
        for path, old, new in edits:
            self.edit(path, old, new)
        git(self.root, "commit", "-qam", message)
        git(self.root, "checkout", "-q", "-")

    def test_local_branches_conflicts_triage_and_worktrees(self):
        from promapkin import prs
        base = git_out(self.root, "rev-parse", "--abbrev-ref", "HEAD")
        self.branch("feat-limit", [("searchapp/domain/ranking.py", "limit * 4", "limit * 8")], "Widen pool")
        self.branch("feat-score", [("searchapp/domain/ranking.py", "limit * 4", "limit * 3")], "Tune scoring")
        self.branch("docs", [("README.md", "#", "##")], "Docs")
        wt = self.root.parent / "wt"
        git(self.root, "worktree", "add", "-q", str(wt), "feat-score")
        r = prs.dashboard(self.ws, "local", base=base, do_triage=True)
        by = {p["id"]: p for p in r["prs"]}
        self.assertEqual(set(by), {"feat-limit", "feat-score", "docs"})
        self.assertIn("symbol:searchapp.domain.ranking:select_candidates", by["feat-limit"]["graph"]["changed_symbols"])
        self.assertTrue(by["feat-limit"]["graph"]["tests_to_run"])
        self.assertEqual(Path(by["feat-score"]["worktree"]).resolve(), wt.resolve())
        high = [c for c in r["conflicts"] if c["level"] == "high"]
        self.assertEqual({(c["a"], c["b"]) for c in high} | {(c["b"], c["a"]) for c in high} >=
                         {("feat-limit", "feat-score")}, True)
        self.assertEqual(r["triage"][0]["id"], "docs")
        one = prs.dashboard(self.ws, "local", number="feat-limit", base=base)
        self.assertIn("detail", one)

    def test_github_source_with_recorded_api(self):
        from promapkin import prs
        patch_rank = "@@ -10,3 +10,3 @@ def select_candidates\n-    x\n+    y\n"
        api = {
            "repos/o/r/pulls?state=open&per_page=30": [
                {"number": 7, "title": "Faster ranking", "user": {"login": "ana"}, "draft": False,
                 "head": {"ref": "fast", "sha": "aaa"}, "base": {"ref": "main"}, "requested_reviewers": [],
                 "html_url": "https://github.com/o/r/pull/7"},
                {"number": 9, "title": "Ranking cleanup", "user": {"login": "bo"}, "draft": False,
                 "head": {"ref": "clean", "sha": "bbb"}, "base": {"ref": "main"},
                 "requested_reviewers": [{"login": "cy"}]}],
            "repos/o/r/pulls/7": {"number": 7, "mergeable": True, "mergeable_state": "clean"},
            "repos/o/r/pulls/9": {"number": 9, "mergeable": False, "mergeable_state": "dirty"},
            "repos/o/r/pulls/7/files?per_page=100&page=1": [
                {"filename": "searchapp/domain/ranking.py", "status": "modified", "additions": 1, "deletions": 1,
                 "patch": patch_rank}],
            "repos/o/r/pulls/9/files?per_page=100&page=1": [
                {"filename": "searchapp/domain/ranking.py", "status": "modified", "additions": 4, "deletions": 4,
                 "patch": patch_rank}],
            "repos/o/r/pulls/7/reviews?per_page=100": [{"user": {"login": "cy"}, "state": "APPROVED"}],
            "repos/o/r/pulls/9/reviews?per_page=100": [],
            "repos/o/r/commits/aaa/check-runs?per_page=100": {"check_runs": [
                {"name": "test", "status": "completed", "conclusion": "success"}]},
            "repos/o/r/commits/aaa/status": {"statuses": []},
            "repos/o/r/commits/bbb/check-runs?per_page=100": {"check_runs": [
                {"name": "test (3.10)", "status": "completed", "conclusion": "failure"}]},
            "repos/o/r/commits/bbb/status": {"statuses": []},
        }
        gh = prs.GitHub("github.com", "o", "r", get=lambda path: api[path])
        r = prs.dashboard(self.ws, "github", gh=gh, do_triage=True)
        by = {p["number"]: p for p in r["prs"]}
        self.assertEqual(by[7]["ci"]["state"], "passing")
        self.assertEqual(by[9]["ci"]["failing"], ["test (3.10)"])
        self.assertEqual(by[7]["review"]["decision"], "approved")
        self.assertEqual(by[9]["review"]["decision"], "review_requested")
        self.assertEqual(r["conflicts"][0]["level"], "high")
        self.assertEqual(r["triage"][0]["id"], "#7")
        self.assertIn("CI failing", " ".join(r["triage"][-1]["why"]))

    def test_remote_parsing_and_hunks(self):
        from promapkin import prs
        self.assertEqual(prs.changed_lines("@@ -1,2 +3,4 @@\n@@ -9 +20 @@"), [(3, 6), (20, 20)])
        git(self.root, "remote", "add", "origin", "git@github.com:acme/search.git")
        self.assertEqual(prs.github_remote(self.root), ("github.com", "acme", "search"))


def git_out(root, *args):
    import subprocess
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True).stdout.strip()


class SourcesTest(RepoTestCase):
    def test_arxiv_web_youtube_and_files(self):
        from promapkin import sources
        from promapkin.scanner import scan
        from tests.test_assets import make_pdf
        atom = ("<feed><entry><title>Attention Is All You Need</title><summary>The SearchService ranks documents "
                "with attention.</summary><published>2017-06-12T00:00:00Z</published><author><name>A. Vaswani"
                "</name></author><category term=\"cs.CL\"/></entry></feed>")
        page = ("<html><head><title>Ranking notes</title></head><body><nav>menu</nav><main><h1>Ranking notes</h1>"
                "<p>Our <code>IndexRepository</code> stores documents.</p>" + "<p>filler text.</p>" * 40 +
                "<pre>repo.find_all()</pre></main><footer>f</footer></body></html>")
        responses = {"https://export.arxiv.org/api/query?id_list=1706.03762": (atom.encode(), "application/atom+xml"),
                     "https://arxiv.org/pdf/1706.03762": (make_pdf(["The IndexRepository persists documents."]),
                                                          "application/pdf"),
                     "https://example.com/notes": (page.encode(), "text/html; charset=utf-8")}
        fetch = lambda url: responses[url]  # noqa: E731
        r = sources.add(self.ws, "https://arxiv.org/abs/1706.03762v5", fetch=fetch)
        self.assertEqual(r["kind"], "arxiv")
        self.assertEqual(r["title"], "Attention Is All You Need")
        self.assertEqual(len(r["files"]), 2)
        sources.add(self.ws, "https://example.com/notes", fetch=fetch)
        transcript = {"title": "Search talk", "channel": "Dev", "segments": [(0.0, "Today we look at the"),
                                                                            (3.0, "SearchService and its ranking.")],
                      "transcription": "captions"}
        sources.add(self.ws, "https://youtu.be/dQw4w9WgXcQ", transcribe=lambda url, d: transcript)
        notes = self.root.parent / "design.md"
        notes.write_text("# Design\n\nThe `Ranker` scores results.\n")
        sources.add(self.ws, str(notes))
        self.assertEqual(len(sources.list_sources(self.ws)), 4)
        scan(self.ws)
        docs = {n["file"] for n in self.ws.store.nodes("document")}
        self.assertTrue(any(f.endswith("arxiv-1706-03762.md") for f in docs))
        self.assertTrue(any(f.endswith(".pdf") for f in docs))
        web = next(f for f in docs if "web-example-com" in f)
        self.assertIn("```", (self.root / web).read_text())
        self.assertNotIn("menu", (self.root / web).read_text())
        mentions = {e["dst"] for e in self.ws.store.edges(types=["documents"])
                    if e["src"].startswith("doc:.promapkin/sources/")}
        self.assertIn("symbol:searchapp.services.search_service:SearchService", mentions)
        self.assertIn("symbol:searchapp.storage.index_repository:IndexRepository", mentions)
        removed = sources.remove(self.ws, "Search talk")
        self.assertEqual(len(removed["removed"]), 1)

    def test_url_guard_and_vtt(self):
        from promapkin import sources
        for bad in ("file:///etc/passwd", "http://127.0.0.1/admin", "ftp://example.com/x"):
            with self.assertRaises(sources.SourceError):
                sources.check_url(bad)
        vtt = ("WEBVTT\n\n00:00:01.000 --> 00:00:03.000\nhello there\n\n00:00:03.000 --> 00:00:05.000\nhello there\n"
               "general kenobi\n\n00:01:05.500 --> 00:01:07.000\n<c>new</c> topic\n")
        self.assertEqual(sources.parse_vtt(vtt), [(1.0, "hello there"), (3.0, "general kenobi"), (65.0, "new topic")])
        self.assertEqual(sources.classify("arXiv:2101.00001")[0], "arxiv")
        self.assertEqual(sources.classify("https://www.youtube.com/watch?v=abcdefgh&t=3")[1], "abcdefgh")


class ExportMergeWikiTest(RepoTestCase):
    def test_export_json_and_graphml(self):
        from promapkin import exports
        r = exports.write_export(self.ws, self.ws.dir / "x" / "g.json", "json")
        data = json.loads(Path(r["path"]).read_text())
        self.assertEqual(data["schema_version"], 1)
        self.assertTrue(any(e["type"] == "calls" for e in data["edges"]))
        g = exports.write_export(self.ws, self.ws.dir / "x" / "g.graphml", "graphml")
        root = ET.parse(g["path"]).getroot()
        self.assertTrue(root.tag.endswith("graphml"))

    def test_merge_links_repositories(self):
        from promapkin import exports
        from promapkin.config import Workspace
        from promapkin.scanner import scan
        api = make_repo()
        (api / "openapi.yaml").write_text("openapi: 3.0.0\ninfo: {title: t, version: '1'}\npaths:\n  /search:\n"
                                          "    get: {summary: s}\n")
        git(api, "add", "-A")
        git(api, "commit", "-qm", "spec")
        ws_api = Workspace(api)
        ws_api.init()
        scan(ws_api)
        (self.root.parent / "spec-only").mkdir()
        exports.write_export(ws_api, self.root.parent / "spec-only" / "api.json")
        out = self.root.parent / "merged"
        r = exports.merge([self.root, self.root.parent / "spec-only" / "api.json"], out)
        self.assertEqual(len(r["repositories"]), 2)
        self.assertGreater(r["cross_repo_links"].get("specifies", 0), 0)
        self.assertGreater(r["cross_repo_links"].get("same_as", 0), 0)
        merged = Workspace(out)
        hits = [h["id"] for h in __import__("promapkin.service", fromlist=["Service"]).Service(merged)
                .search("IndexRepository", limit=5)["results"]]
        self.assertTrue(any(h.startswith("repo::") or "::symbol:" in h for h in hits), hits)
        merged.close()
        ws_api.close()

    def test_wiki_pages_and_links_resolve(self):
        from promapkin import exports
        r = exports.wiki(self.ws)
        out = Path(r["path"])
        self.assertTrue((out / "index.md").exists())
        self.assertTrue(any((out / "modules").iterdir()))
        broken = []
        for page in out.rglob("*.md"):
            for target in re.findall(r"\]\(([^)#]+)\)", page.read_text()):
                if not (page.parent / target).resolve().exists():
                    broken.append((page.name, target))
        self.assertEqual(broken, [])
        mod = (out / "modules" / "searchapp.storage.index_repository.md").read_text()
        self.assertIn("IndexRepository", mod)

    def test_callflow(self):
        from promapkin import exports
        flows = exports.callflow(self.ws, "api:GET /search", depth=4)
        self.assertIn("flowchart LR", flows[0]["mermaid"])
        self.assertIn("search_endpoint", flows[0]["mermaid"])
        r = exports.callflow_html(self.ws, self.ws.blueprint_dir / "callflow.html", flows)
        self.assertIn("mermaid", Path(r["path"]).read_text())
        self.assertGreater(len(exports.callflow(self.ws)), 0)


class HyperedgeTest(RepoTestCase):
    def test_co_change_flow_and_html(self):
        from promapkin import insights
        from promapkin.scanner import scan
        for i in range(3):
            self.edit("searchapp/domain/ranking.py", f"limit * {4 + i}", f"limit * {5 + i}")
            p = self.root / "searchapp" / "services" / "search_service.py"
            p.write_text(p.read_text() + f"# change {i}\n")
            git(self.root, "commit", "-qam", f"co-change {i}")
        scan(self.ws)
        hs = insights.hyperedges(self.ws.store, self.root)
        co = [h for h in hs if h["kind"] == "co_change"]
        self.assertTrue(co)
        self.assertEqual(set(co[0]["members"]), {"module:searchapp.domain.ranking",
                                                 "module:searchapp.services.search_service"})
        self.assertGreaterEqual(co[0]["support"], 3)
        self.assertTrue(any(h["kind"] == "flow" for h in hs))
        from promapkin.service import Service
        r = Service(self.ws).export_html()
        html = (self.root / r["path"]).read_text()
        self.assertIn('"hyper":[', html)

    def test_semantic_duplicates(self):
        from promapkin import insights
        s = self.ws.store
        with s.transaction():
            s.add_edge("symbol:searchapp.domain.ranking:select_candidates",
                       "symbol:searchapp.adapters.embedding_client:EmbeddingAdapter.embed", "similar_to",
                       evidence="ml_classifier", confidence=0.91)
        rows = insights.semantic_surprises(s)
        self.assertEqual(rows[0]["similarity"], 0.91)
        self.assertNotEqual(rows[0]["communities"][0], rows[0]["communities"][1])


class McpAndTomlTest(unittest.TestCase):
    def test_new_tools_registered(self):
        from promapkin.mcp_server import TOOLS
        names = {t.name for t in TOOLS}
        for t in ("get_pull_requests", "add_source", "list_sources", "get_hyperedges", "find_duplicate_logic",
                  "render_callflow", "export_graph", "export_wiki"):
            self.assertIn(t, names)

    def test_tomlish(self):
        from promapkin.extract import tomlish
        d = tomlish.loads('[package]\nname = "app" # c\nversion = "0.1"\n\n[dependencies]\nserde = { version = "1",'
                          ' features = ["derive"] }\nmy-core = { path = "../core" }\n\n[workspace]\nmembers = [\n'
                          '  "a",\n  "b",\n]\n[[bin]]\nname = "x"\n')
        self.assertEqual(d["package"]["name"], "app")
        self.assertEqual(d["dependencies"]["serde"]["features"], ["derive"])
        self.assertEqual(d["workspace"]["members"], ["a", "b"])
        self.assertEqual(d["bin"][0]["name"], "x")


if __name__ == "__main__":
    unittest.main()
