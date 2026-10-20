"""Structural extraction.

Extractors are pure functions over file text: they never import or execute
repository code. Each returns a `Facts` bundle of nodes, edges and unresolved
references that the scanner persists and the resolver links across files.
"""

from __future__ import annotations

import re
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
    ".mts": "typescript",
    ".cts": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".cs": "csharp",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".rb": "ruby",
    ".php": "php",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
    ".swift": "swift",
    ".scala": "scala",
    ".sc": "scala",
    ".dart": "dart",
    ".lua": "lua",
    ".ex": "elixir",
    ".exs": "elixir",
    ".erl": "erlang",
    ".hrl": "erlang",
    ".hs": "haskell",
    ".lhs": "haskell",
    ".ml": "ocaml",
    ".mli": "ocaml",
    ".fs": "fsharp",
    ".fsi": "fsharp",
    ".fsx": "fsharp",
    ".clj": "clojure",
    ".cljs": "clojure",
    ".cljc": "clojure",
    ".r": "r",
    ".jl": "julia",
    ".pl": "perl",
    ".pm": "perl",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "bash",
    ".ps1": "powershell",
    ".psm1": "powershell",
    ".zig": "zig",
    ".nim": "nim",
    ".m": "objc",          # or MATLAB: decided by content in language_of
    ".mm": "objc",
    ".groovy": "groovy",
    ".gradle": "groovy",
    ".elm": "elm",
    ".sol": "solidity",
    ".cr": "crystal",
    ".d": "d",
    ".f": "fortran",
    ".f90": "fortran",
    ".f95": "fortran",
    ".f03": "fortran",
    ".f08": "fortran",
    ".pas": "pascal",
    ".pp": "pascal",
    ".dpr": "pascal",
    ".v": "verilog",       # or V: decided by content in language_of
    ".sv": "verilog",
    ".svh": "verilog",
    ".vhd": "vhdl",
    ".vhdl": "vhdl",
    ".gleam": "gleam",
    ".odin": "odin",
    ".lisp": "commonlisp",
    ".lsp": "commonlisp",
    ".cl": "commonlisp",
    ".scm": "scheme",
    ".ss": "scheme",
    ".rkt": "racket",
    ".cu": "cuda",
    ".cuh": "cuda",
    ".glsl": "glsl",
    ".vert": "glsl",
    ".frag": "glsl",
    ".hlsl": "hlsl",
    ".cls": "apex",
    ".trigger": "apex",
    ".vue": "vue",
    ".svelte": "svelte",
    ".tf": "terraform",
    ".tfvars": "terraform",
    ".sql": "sql",
    ".prisma": "prisma",
    ".graphql": "graphql",
    ".graphqls": "graphql",
    ".gql": "graphql",
    ".proto": "proto",
    ".yml": "yaml",        # Kubernetes, OpenAPI, Compose or CI by content/path; other YAML is skipped
    ".yaml": "yaml",
    ".json": "json",       # OpenAPI by content; other JSON is skipped
    ".pdf": "pdf",
    ".docx": "office",
    ".pptx": "office",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".gif": "image",
    ".webp": "image",
    ".bmp": "image",
    ".svg": "image",
    ".mp4": "video",
    ".mov": "video",
    ".m4v": "video",
    ".webm": "video",
    ".mkv": "video",
    ".avi": "video",
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


def language_of(path: str, head: str | None = None) -> str | None:
    """Language by file name; `head` (the start of the file) settles shared extensions (.m, .v, .yaml, .json).
    Without `head`, sniffable files report their placeholder (see SNIFF) so the caller knows to read them."""
    from promapkin.extract.infra_ext import infra_language
    p = PurePosixPath(path)
    named = infra_language(path, "")
    if named in ("dockerfile", "compose", "terraform"):
        return named
    if p.name in MANIFESTS or p.name.startswith("requirements") and p.suffix == ".txt":
        return "manifest"
    if p.suffix == ".txt":
        return None
    lang = LANGUAGES.get(p.suffix.lower()) or ("dockerfile" if p.name.lower().startswith("dockerfile") else None)
    if head is not None and lang in SNIFF:
        return SNIFF[lang](path, head)
    return lang


def _sniff_infra(path: str, head: str) -> str | None:
    from promapkin.extract.infra_ext import infra_language
    return infra_language(path, head)


def _sniff_m(path: str, head: str) -> str:
    return "objc" if re.search(r"^\s*(#import|@interface|@implementation|@protocol)\b", head, re.M) else "matlab"


def _sniff_v(path: str, head: str) -> str:
    return "verilog" if re.search(r"^\s*(module\s+\w+\s*[#(;]|`timescale|`include)", head, re.M) and \
        "endmodule" in head or "`timescale" in head else "v"


SNIFF = {"objc": _sniff_m, "verilog": _sniff_v, "yaml": _sniff_infra, "json": _sniff_infra}


TEST_SUFFIXES = ("_test.py", "_test.go", "_test.rs", "_test.rb", "_spec.rb", "Test.java", "Tests.java", "Test.kt",
                 "Tests.kt", "Tests.cs", "Test.cs", "Test.php", "Tests.swift", "Spec.scala", "Test.scala", "_test.exs",
                 "_test.dart", "_spec.lua", "_test.lua", "Spec.hs", "_test.jl", ".t", "_test.zig", "_test.nim",
                 "Test.groovy", "Spec.groovy", "_test.gleam", "_SUITE.erl", "_tests.erl", ".bats", ".Tests.ps1")
TREE_SITTER_CORE = {"go", "rust", "java", "csharp", "kotlin", "ruby", "php", "c", "cpp", "swift", "scala", "cuda",
                    "glsl", "hlsl", "apex"}
TREE_SITTER_GENERIC = {"dart", "lua", "elixir", "erlang", "haskell", "ocaml", "fsharp", "clojure", "commonlisp",
                       "scheme", "racket", "r", "julia", "perl", "bash", "powershell", "zig", "nim", "objc", "elm",
                       "solidity", "crystal", "d", "fortran", "pascal", "matlab", "verilog", "vhdl", "gleam", "odin",
                       "v"}
LEXICAL_ONLY = {"groovy"}            # no usable grammar: lexical extraction only
EMBEDDED = {"vue", "svelte"}         # single-file components: their <script> blocks go through the JS/TS extractor
TREE_SITTER_LANGUAGES = TREE_SITTER_CORE | TREE_SITTER_GENERIC
INFRA_LANGUAGES = {"dockerfile", "compose", "kubernetes", "terraform", "ci", "openapi"}
SCHEMA_LANGUAGES = {"sql", "prisma", "graphql", "proto"}
ASSET_LANGUAGES = {"pdf", "office", "image", "video"}   # binary: read as bytes by the scanner
NON_CODE = {"markdown", "manifest", "text"} | INFRA_LANGUAGES | SCHEMA_LANGUAGES | ASSET_LANGUAGES
GENERATED_MARKERS = ("code generated", "do not edit", "@generated", "autogenerated", "auto-generated")


def is_test_path(path: str) -> bool:
    p = PurePosixPath(path)
    name = p.name
    return (
        name.startswith("test_") or name.endswith(TEST_SUFFIXES) or ".test." in name or ".spec." in name
        or any(part in ("tests", "test", "__tests__", "spec", "testing") for part in p.parts[:-1])
    )


def is_generated(path: str, text: str) -> bool:
    """Generated code (protobuf stubs, codegen output) is indexed but marked, so readers know not to edit it."""
    name = PurePosixPath(path).name
    if re.search(r"(_pb2(_grpc)?\.py|\.pb\.go|\.generated\.|\.g\.(cs|dart)|_generated\.)", name):
        return True
    head = text[:600].lower()
    return any(m in head for m in GENERATED_MARKERS)


def extractor_for(language: str) -> Callable[[str, str], Facts] | None:
    if language == "python":
        from promapkin.extract.python_ext import extract
        return extract
    if language in ("javascript", "typescript"):
        from promapkin.extract.js_ext import extract
        return extract
    if language in TREE_SITTER_LANGUAGES or language in LEXICAL_ONLY:
        from promapkin import toolchain
        from promapkin.extract.generic_lexical import extract as lexical
        if language in LEXICAL_ONLY or not toolchain.tree_sitter_available():
            return lambda path, text: lexical(path, text, language)
        if language in TREE_SITTER_CORE:
            from promapkin.extract.treesitter_ext import extract as ts_extract
        else:
            from promapkin.extract.treesitter_generic import extract as ts_extract

        def run(path: str, text: str) -> Facts:
            try:
                return ts_extract(path, text, language)
            except Exception as exc:  # noqa: BLE001 - grammar missing or failed to load: stay lexical
                facts = lexical(path, text, language)
                facts.errors.append(f"tree-sitter grammar for {language} unavailable ({str(exc)[:120]}); "
                                    "lexical fallback")
                return facts
        return run
    if language in INFRA_LANGUAGES:
        from promapkin.extract.infra_ext import extract as infra
        return lambda path, text: infra(path, text, language)
    if language in SCHEMA_LANGUAGES:
        from promapkin.extract.schema_ext import extract as schema
        return lambda path, text: schema(path, text, language)
    if language in EMBEDDED:
        from promapkin.extract.embedded import extract as embedded
        return embedded
    if language == "markdown":
        from promapkin.extract.docs_ext import extract
        return extract
    if language == "manifest":
        from promapkin.extract.manifest_ext import extract
        return extract
    return None
