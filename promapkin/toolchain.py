"""Optional parser toolchains: the TypeScript compiler (via Node) and tree-sitter grammars.

Both are optional. When present they replace the lexical extractors with real parsers;
when absent Promapkin falls back (lexical extraction, or LLM extraction in --llm mode).
Installing them is explicit (`promapkin parsers install`) and never touches the analyzed
repository: packages go to Promapkin's own cache directory.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path


def cache_dir() -> Path:
    base = os.environ.get("PROMAPKIN_TOOLCHAIN_DIR") or os.path.join(
        os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")), "promapkin", "toolchain")
    return Path(base)


def node_binary() -> str | None:
    return os.environ.get("PROMAPKIN_NODE") or shutil.which("node")


def typescript_module(repo_root: Path | None = None) -> str | None:
    """Path to the `typescript` package: $PROMAPKIN_TYPESCRIPT, the repo's own, or Promapkin's cache."""
    candidates = []
    if os.environ.get("PROMAPKIN_TYPESCRIPT"):
        candidates.append(Path(os.environ["PROMAPKIN_TYPESCRIPT"]))
    if repo_root is not None:
        candidates.append(repo_root / "node_modules" / "typescript")
    candidates.append(cache_dir() / "node" / "node_modules" / "typescript")
    for c in candidates:
        if (c / "lib" / "typescript.js").is_file() and (c / "package.json").is_file():
            return str(c)
    return None


def typescript_available(repo_root: Path | None = None) -> bool:
    return bool(node_binary() and typescript_module(repo_root))


def tree_sitter_available() -> bool:
    return importlib.util.find_spec("tree_sitter_language_pack") is not None and \
        importlib.util.find_spec("tree_sitter") is not None


def status(repo_root: Path | None = None) -> dict:
    ts_mod = typescript_module(repo_root)
    version = None
    if ts_mod:
        try:
            import json
            version = json.loads((Path(ts_mod) / "package.json").read_text()).get("version")
        except (OSError, ValueError):
            pass
    return {
        "node": node_binary(), "typescript": ts_mod, "typescript_version": version,
        "javascript_typescript": "typescript-compiler" if ts_mod and node_binary() else "lexical",
        "tree_sitter": tree_sitter_available(),
        "other_languages": "tree-sitter" if tree_sitter_available() else "lexical",
        "cache_dir": str(cache_dir()),
    }


def install_typescript(version: str = "5") -> tuple[bool, str]:
    npm = shutil.which("npm")
    if not npm or not node_binary():
        return False, "Node.js and npm are required to install the TypeScript parser"
    target = cache_dir() / "node"
    target.mkdir(parents=True, exist_ok=True)
    if not (target / "package.json").exists():
        (target / "package.json").write_text('{"name": "promapkin-toolchain", "private": true}\n')
    proc = subprocess.run([npm, "install", "--no-audit", "--no-fund", "--silent", f"typescript@{version}"],
                          cwd=target, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        return False, f"npm install failed: {proc.stderr.strip()[-400:]}"
    return True, f"typescript installed in {target}"


def install_tree_sitter() -> tuple[bool, str]:
    proc = subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "tree-sitter",
                           "tree-sitter-language-pack"], capture_output=True, text=True, check=False)
    importlib.invalidate_caches()
    if proc.returncode != 0 or not tree_sitter_available():
        return False, f"pip install failed: {proc.stderr.strip()[-400:]}"
    return True, "tree-sitter grammars installed"
