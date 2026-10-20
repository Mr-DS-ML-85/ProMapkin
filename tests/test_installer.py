import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests.helpers import RepoTestCase
from promapkin import installer

ROOT = Path(__file__).resolve().parents[1]


class InstallerTest(RepoTestCase):
    def setUp(self):
        super().setUp()
        self.home = Path(tempfile.mkdtemp(prefix="promapkin-home-"))
        (self.root / ".mcp.json").write_text(json.dumps({"mcpServers": {"other": {"command": "other-server"}}}))
        (self.root / "CLAUDE.md").write_text("# Team notes\n\nUse tabs.\n")
        (self.root / ".claude").mkdir()
        (self.root / ".claude" / "settings.json").write_text(json.dumps(
            {"permissions": {"allow": ["Bash(ls)"]},
             "hooks": {"SessionStart": [{"matcher": "startup", "hooks": [{"type": "command", "command": "echo hi"}]}]}}))
        (self.root / ".cursor").mkdir()
        (self.root / ".cursor" / "mcp.json").write_text("{ // a comment makes this invalid JSON\n}")
        (self.home / ".codex").mkdir()
        (self.home / ".codex" / "config.toml").write_text('model = "gpt-5"\n')
        self.originals = {p: (self.root / p).read_text() for p in (".mcp.json", "CLAUDE.md", ".claude/settings.json",
                                                                   ".cursor/mcp.json")}

    def install(self, *platforms, **kw):
        return installer.install(self.root, list(platforms), home=self.home, **kw)

    def test_every_platform_installs_and_uninstalls_cleanly(self):
        r = self.install(*installer.PLATFORMS)
        self.assertEqual(len(r["platforms"]), len(installer.PLATFORMS))
        self.assertTrue(any("not valid JSON" in n for n in r["notes"]))       # never rewrites a file it can't parse
        self.assertEqual((self.root / ".cursor" / "mcp.json").read_text(), self.originals[".cursor/mcp.json"])

        mcp = json.loads((self.root / ".mcp.json").read_text())["mcpServers"]
        self.assertEqual(set(mcp), {"other", "promapkin"})
        self.assertEqual(mcp["promapkin"]["type"], "stdio")
        self.assertEqual(mcp["promapkin"]["args"][-2:], ["mcp", "serve"])
        self.assertIn("Use tabs.", (self.root / "CLAUDE.md").read_text())
        self.assertIn("build_task_context", (self.root / "CLAUDE.md").read_text())
        settings = json.loads((self.root / ".claude" / "settings.json").read_text())
        self.assertEqual(settings["permissions"], {"allow": ["Bash(ls)"]})
        self.assertEqual(len(settings["hooks"]["SessionStart"]), 2)            # the user's hook is kept
        self.assertIn("brief --sync", json.dumps(settings["hooks"]["SessionStart"]))
        self.assertEqual(settings["hooks"]["PostToolUse"][0]["matcher"], "Edit|MultiEdit|Write")
        self.assertTrue((self.root / ".claude" / "skills" / "promapkin" / "SKILL.md").exists())
        self.assertIn("$ARGUMENTS", (self.root / ".claude" / "commands" / "promapkin-impact.md").read_text())

        self.assertEqual(json.loads((self.root / ".vscode" / "mcp.json").read_text())["servers"]["promapkin"]["type"],
                         "stdio")
        oc = json.loads((self.root / "opencode.json").read_text())
        self.assertEqual(oc["mcp"]["promapkin"]["type"], "local")
        self.assertIsInstance(oc["mcp"]["promapkin"]["command"], list)
        self.assertIn("context_servers", json.loads((self.root / ".zed" / "settings.json").read_text()))
        self.assertIn("alwaysApply: true", (self.root / ".cursor" / "rules" / "promapkin.mdc").read_text())
        self.assertIn("inclusion: always", (self.root / ".kiro" / "steering" / "promapkin.md").read_text())
        self.assertIn("{{args}}", (self.root / ".gemini" / "commands" / "promapkin" / "impact.toml").read_text())
        codex = (self.home / ".codex" / "config.toml").read_text()
        self.assertIn('model = "gpt-5"', codex)
        self.assertIn("[mcp_servers.promapkin]", codex)
        self.assertIn("promapkin", (self.home / ".codeium" / "windsurf" / "mcp_config.json").read_text())
        self.assertIn("build_task_context", (self.root / "AGENTS.md").read_text())

        again = self.install(*installer.PLATFORMS)
        self.assertEqual(again["changes"], [], "a second install changes nothing")

        installer.uninstall(self.root, home=self.home)
        for rel, text in self.originals.items():
            got = (self.root / rel).read_text()
            if rel.endswith(".json") and rel != ".cursor/mcp.json":
                self.assertEqual(json.loads(got), json.loads(text), rel)
            else:
                self.assertEqual(got.strip(), text.strip(), rel)
        self.assertEqual((self.home / ".codex" / "config.toml").read_text().strip(), 'model = "gpt-5"')
        for gone in ("AGENTS.md", "GEMINI.md", ".vscode", "opencode.json", ".kiro", ".claude/skills", ".claude/commands",
                     ".roo", ".zed", ".continue"):
            self.assertFalse((self.root / gone).exists(), gone)
        self.assertFalse((self.home / ".codeium" / "windsurf" / "mcp_config.json").exists())

    def test_user_edited_owned_files_are_left_alone(self):
        self.install("cursor")
        rule = self.root / ".cursor" / "rules" / "promapkin.mdc"
        rule.write_text("my own rules\n")
        r = self.install("cursor")
        self.assertTrue(any("kept .cursor/rules/promapkin.mdc" in n for n in r["notes"]))
        self.assertEqual(rule.read_text(), "my own rules\n")
        installer.uninstall(self.root, ["cursor"], home=self.home)
        self.assertEqual(rule.read_text(), "my own rules\n")

    def test_dry_run_writes_nothing(self):
        before = sorted(p.relative_to(self.root) for p in self.root.rglob("*") if ".git" not in p.parts)
        r = self.install("claude-code", "vscode", dry_run=True)
        self.assertTrue(r["changes"])
        after = sorted(p.relative_to(self.root) for p in self.root.rglob("*") if ".git" not in p.parts)
        self.assertEqual(before, after)

    def test_configured_server_and_hook_actually_run(self):
        self.install("claude-code")
        entry = json.loads((self.root / ".mcp.json").read_text())["mcpServers"]["promapkin"]
        args = [a.replace("${CLAUDE_PROJECT_DIR:-.}", str(self.root)) for a in entry["args"]]
        env = {**os.environ, "PYTHONPATH": str(ROOT)}
        msgs = [{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18",
                                                                              "capabilities": {}, "clientInfo":
                                                                              {"name": "t", "version": "1"}}},
                {"jsonrpc": "2.0", "method": "notifications/initialized"},
                {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}]
        out = subprocess.run([entry["command"], *args], input="\n".join(json.dumps(m) for m in msgs) + "\n",
                             capture_output=True, text=True, env=env, timeout=60).stdout
        replies = [json.loads(line) for line in out.splitlines() if line.strip()]
        tools = next(r for r in replies if r.get("id") == 2)["result"]["tools"]
        self.assertIn("build_task_context", {t["name"] for t in tools})

        groups = json.loads((self.root / ".claude" / "settings.json").read_text())["hooks"]["SessionStart"]
        cmd = next(g["hooks"][0]["command"] for g in groups if "brief" in g["hooks"][0]["command"])
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120,
                             env={**env, "CLAUDE_PROJECT_DIR": str(self.root)})
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertIn("Promapkin index of", res.stdout)
        self.assertIn("build_task_context", res.stdout)

    def test_cli_list_and_unknown_platform(self):
        from promapkin.cli import main
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(main(["--root", str(self.root), "install", "--list"]), 0)
        self.assertIn("claude-code", out.getvalue())
        self.assertIn("*", out.getvalue())                     # .claude/ exists here: detected
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--root", str(self.root), "install", "-p", "notepad"]), 2)
        self.assertIn("unknown platform", err.getvalue())


class BriefTest(RepoTestCase):
    def test_brief_without_index_and_with_open_handoff(self):
        from promapkin.brief import brief
        from promapkin.config import Workspace
        empty = Path(tempfile.mkdtemp())
        self.assertIn("no index yet", brief(Workspace(empty)))
        from promapkin.service import Service
        Service(self.ws).create_handoff({"task": "Add rate limiting", "next_action": "wire limiter into routes",
                                         "constraints": ["no new dependencies"]})
        text = brief(self.ws)
        self.assertIn("Open handoff", text)
        self.assertIn("no new dependencies", text)
        self.assertLessEqual(len(brief(self.ws, max_chars=300)), 300)


if __name__ == "__main__":
    unittest.main()
