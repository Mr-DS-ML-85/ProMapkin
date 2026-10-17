"""Test helpers: an isolated git copy of the sample repository with ML downloads disabled."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

FIXTURE = Path(__file__).parent / "fixtures" / "sample_repo"

os.environ["PROMAPKIN_ML_MODE"] = "never"          # tests never download models
os.environ.setdefault("PROMAPKIN_MODEL_DIR", tempfile.mkdtemp(prefix="promapkin-models-"))


def git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), "-c", "user.name=test", "-c", "user.email=test@example.com", *args],
                   check=True, capture_output=True)


def make_repo() -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="promapkin-test-"))
    root = tmp / "repo"
    shutil.copytree(FIXTURE, root)
    git(root, "init", "-q")
    git(root, "add", "-A")
    git(root, "commit", "-qm", "initial")
    return root


class RepoTestCase(unittest.TestCase):
    scan_on_setup = True

    def setUp(self) -> None:
        from promapkin.config import Workspace
        from promapkin.scanner import scan

        self.root = make_repo()
        self.ws = Workspace(self.root)
        self.ws.init()
        if self.scan_on_setup:
            self.scan_result = scan(self.ws)

    def tearDown(self) -> None:
        self.ws.close()
        shutil.rmtree(self.root.parent, ignore_errors=True)

    def edit(self, rel: str, old: str, new: str) -> None:
        p = self.root / rel
        text = p.read_text()
        assert old in text, f"{old!r} not in {rel}"
        p.write_text(text.replace(old, new))
