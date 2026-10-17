"""Structural extraction.

Extractors are pure functions over file text: they never import or execute
repository code. Each returns a `Facts` bundle of nodes, edges and unresolved
references that the scanner persists and the resolver links across files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any, Callable

LANGUAGES = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".md": "markdown",
    ".markdown": "markdown",
    ".rst": "markdown",
    ".txt": "text",
}
MANIFESTS = {
    "requirements.txt", "requirements-dev.txt", "pyproject.toml", "setup.cfg", "package.json",
    "go.mod", "Cargo.toml", "Pipfile", "Dockerfile", "docker-compose.yml", "docker-compose.yaml",
    ".env.example", "Makefile",
}


@dataclass
class Facts:
    path: str
    language: str
    nodes: list[dict[str, Any]] = field(default_factory=list)
    edges: list[dict[str, Any]] = field(default_factory=list)
    refs: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def node(self, id: str, kind: str, name: str, evidence: str, **kw: Any) -> str:
        self.nodes.append({"id": id, "kind": kind, "name": name, "evidence": evidence, **kw})
        return id

    def edge(self, src: str, dst: str, type: str, evidence: str, **kw: Any) -> None:
        self.edges.append({"src": src, "dst": dst, "type": type, "evidence": evidence, **kw})

    def ref(self, src: str, target: str, type: str, line: int | None, **context: Any) -> None:
        self.refs.append({"src": src, "target": target, "type": type, "line": line, "context": context})


def language_of(path: str) -> str | None:
    p = PurePosixPath(path)
    if p.name in MANIFESTS or p.name.startswith("requirements") and p.suffix == ".txt":
        return "manifest"
    if p.suffix == ".txt":
        return None
    return LANGUAGES.get(p.suffix.lower())


def is_test_path(path: str) -> bool:
    p = PurePosixPath(path)
    name = p.name
    return (
        name.startswith("test_") or name.endswith("_test.py") or ".test." in name or ".spec." in name
        or any(part in ("tests", "test", "__tests__", "spec") for part in p.parts[:-1])
    )


def extractor_for(language: str) -> Callable[[str, str], Facts] | None:
    if language == "python":
        from promapkin.extract.python_ext import extract
        return extract
    if language in ("javascript", "typescript"):
        from promapkin.extract.js_ext import extract
        return extract
    if language == "markdown":
        from promapkin.extract.docs_ext import extract
        return extract
    if language == "manifest":
        from promapkin.extract.manifest_ext import extract
        return extract
    return None
