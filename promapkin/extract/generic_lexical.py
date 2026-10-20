"""Lexical fallback for languages without an installed parser (used when tree-sitter is missing
and --llm mode is off). Facts carry `lexical` evidence and are weighted accordingly."""

from __future__ import annotations

import re

from promapkin import evidence as ev
from promapkin.extract import Facts, is_test_path
from promapkin.extract.treesitter_ext import TEST_FUNC, _norm_call, module_name

PATTERNS: dict[str, dict[str, str]] = {
    "go": {"class": r"^type\s+(?P<name>[A-Z_a-z]\w*)\s+(struct|interface)\b",
           "func": r"^func\s+(?:\((?:\w+\s+)?\*?(?P<cls>\w+)\)\s+)?(?P<name>\w+)\s*\(",
           "import": r"^\s*(?:import\s+)?(?:\w+\s+)?\"(?P<target>[^\"]+)\""},
    "rust": {"class": r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:struct|enum|trait)\s+(?P<name>\w+)",
             "func": r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:const\s+)?(?:async\s+)?(?:unsafe\s+)?fn\s+(?P<name>\w+)",
             "import": r"^\s*(?:pub\s+)?use\s+(?P<target>[\w:]+)",
             "impl": r"^\s*impl(?:<[^>]*>)?\s+(?:(?P<trait>[\w:]+)(?:<[^>]*>)?\s+for\s+)?(?P<name>\w+)"},
    "java": {"class": r"^\s*(?:(?:public|private|protected|abstract|final|static)\s+)*(?:class|interface|enum|record)\s+(?P<name>\w+)",
             "func": r"^\s*(?:(?:public|private|protected|static|final|abstract|synchronized)\s+)+[\w<>\[\],\s]+?\s+(?P<name>\w+)\s*\(",
             "import": r"^\s*import\s+(?:static\s+)?(?P<target>[\w.]+)"},
    "csharp": {"class": r"^\s*(?:(?:public|private|internal|protected|abstract|sealed|static|partial)\s+)*(?:class|interface|struct|record)\s+(?P<name>\w+)",
               "func": r"^\s*(?:(?:public|private|internal|protected|static|virtual|override|async)\s+)+[\w<>\[\],\s]+?\s+(?P<name>\w+)\s*\(",
               "import": r"^\s*using\s+(?P<target>[\w.]+)\s*;"},
    "kotlin": {"class": r"^\s*(?:(?:open|data|abstract|sealed|private|internal)\s+)*(?:class|interface|object)\s+(?P<name>\w+)",
               "func": r"^\s*(?:(?:private|public|internal|override|suspend|open)\s+)*fun\s+(?:<[^>]*>\s*)?(?:\w+\.)?(?P<name>\w+)\s*\(",
               "import": r"^\s*import\s+(?P<target>[\w.]+)"},
    "ruby": {"class": r"^\s*(?:class|module)\s+(?P<name>[A-Z]\w*)",
             "func": r"^\s*def\s+(?:self\.)?(?P<name>\w+[?!]?)",
             "import": r"^\s*require(?:_relative)?\s+['\"](?P<target>[^'\"]+)"},
    "php": {"class": r"^\s*(?:abstract\s+|final\s+)?(?:class|interface|trait)\s+(?P<name>\w+)",
            "func": r"^\s*(?:(?:public|private|protected|static|abstract|final)\s+)*function\s+(?P<name>\w+)\s*\(",
            "import": r"^\s*use\s+(?P<target>[\w\\]+)"},
    "c": {"class": r"^\s*(?:typedef\s+)?struct\s+(?P<name>\w+)\s*\{",
          "func": r"^[\w\s\*]+?\b(?P<name>\w+)\s*\([^;]*\)\s*\{",
          "import": r"^\s*#include\s+[\"<](?P<target>[^\">]+)"},
    "cpp": {"class": r"^\s*(?:class|struct)\s+(?P<name>\w+)[^;]*\{",
            "func": r"^[\w\s\*&:<>]+?\b(?:(?P<cls>\w+)::)?(?P<name>~?\w+)\s*\([^;]*\)\s*(?:const\s*)?\{",
            "import": r"^\s*#include\s+[\"<](?P<target>[^\">]+)"},
    "swift": {"class": r"^\s*(?:(?:public|private|final|open)\s+)*(?:class|struct|protocol|enum)\s+(?P<name>\w+)",
              "func": r"^\s*(?:(?:public|private|static|override|open)\s+)*func\s+(?P<name>\w+)",
              "import": r"^\s*import\s+(?P<target>\w+)"},
    "scala": {"class": r"^\s*(?:(?:case|abstract|sealed|final)\s+)*(?:class|object|trait)\s+(?P<name>\w+)",
              "func": r"^\s*(?:(?:override|private|protected)\s+)*def\s+(?P<name>\w+)",
              "import": r"^\s*import\s+(?P<target>[\w.]+)"},
}
CALL_RE = re.compile(r"(?<![\w$.])((?:[A-Za-z_$][\w$]*(?:\.|::|->))*[A-Za-z_$][\w$]*)\s*\(")
DEF_KEYWORDS = {"if", "for", "while", "switch", "catch", "return", "match", "when", "else", "elif"}
KEYWORDS = {"if", "for", "while", "switch", "catch", "return", "func", "fn", "def", "function", "new", "sizeof",
            "match", "when", "elif", "else", "and", "or", "not", "super", "this", "self", "typeof", "using"}


def extract(path: str, text: str, language: str) -> Facts:
    f = Facts(path, language)
    pats = {k: re.compile(v, re.M) for k, v in PATTERNS[language].items()}
    mod = module_name(path)
    mod_id = f"module:{mod}"
    is_test = is_test_path(path)
    lines = text.splitlines()
    f.node(mod_id, "module", mod, ev.LEXICAL, qualname=mod, file=path, line_start=1, line_end=len(lines) or 1,
           attrs={"test": is_test, "loc": len(lines), "parser": "lexical", "imports": {}})
    f.edge(f"file:{path}", mod_id, "defines", ev.LEXICAL)

    def line_of(pos: int) -> int:
        return text.count("\n", 0, pos) + 1

    for m in pats["import"].finditer(text):
        f.ref(mod_id, m.group("target").replace("::", ".").replace("\\", "."), "imports", line_of(m.start()),
              module=mod)
    defs: list[tuple[int, str, str, str | None]] = []   # (line, kind, name, class)
    for m in pats["class"].finditer(text):
        defs.append((line_of(m.start()), "class", m.group("name"), None))
    for m in pats["func"].finditer(text):
        name = m.group("name")
        if name in DEF_KEYWORDS:
            continue
        defs.append((line_of(m.start()), "function", name, m.groupdict().get("cls")))
    impl_spans: list[tuple[int, int, str]] = []
    if "impl" in pats:   # Rust: methods belong to the type named by the enclosing `impl` block
        for m in pats["impl"].finditer(text):
            start = line_of(m.start())
            end = _block_end_line(text, m.end(), start)
            impl_spans.append((start, end, m.group("name")))
            if m.group("trait"):
                f.ref(f"symbol:{mod}:{m.group('name')}", m.group("trait").replace("::", "."), "implements", start,
                      module=mod)
    defs.sort()
    current_class = None
    spans = []
    for i, (line, kind, name, cls) in enumerate(defs):
        end = defs[i + 1][0] - 1 if i + 1 < len(defs) else len(lines)
        if kind == "class":
            current_class = name
            sid = f"symbol:{mod}:{name}"
            f.node(sid, "class", name, ev.LEXICAL, qualname=f"{mod}.{name}", file=path, line_start=line,
                   line_end=end, attrs={"symbol_type": "class", "methods": []})
            f.edge(mod_id, sid, "defines", ev.LEXICAL, line=line)
            continue
        impl = next((n for a, b, n in reversed(impl_spans) if a < line <= b), None)
        owner = cls or impl or (current_class if _indented(lines[line - 1]) and not impl_spans else None)
        qual = f"{owner}.{name}" if owner else name
        sid = f"symbol:{mod}:{qual}"
        kind_ = "test" if is_test and TEST_FUNC.search(name) else ("method" if owner else "function")
        f.node(sid, kind_, name, ev.LEXICAL, qualname=f"{mod}.{qual}", file=path, line_start=line, line_end=end,
               attrs={"symbol_type": kind_, "class": owner})
        declared = owner is not None and f"symbol:{mod}:{owner}" in {n["id"] for n in f.nodes if n["kind"] == "class"}
        f.edge(f"symbol:{mod}:{owner}" if declared else mod_id, sid, "contains" if declared else "defines",
               ev.LEXICAL, line=line)
        spans.append((line, end, sid, owner))
    for i, line_text in enumerate(lines, start=1):
        span = next((s for s in reversed(spans) if s[0] < i <= s[1]), None)
        if not span:
            continue
        for m in CALL_RE.finditer(line_text):
            target = _norm_call(m.group(1))
            if target and target.split(".")[0] not in KEYWORDS:
                f.ref(span[2], target, "calls", i, module=mod, cls=span[3])
    return f


def _block_end_line(text: str, pos: int, start_line: int) -> int:
    """Line of the brace closing the block that opens at or after `pos` (comments/strings not masked)."""
    i = text.find("{", pos)
    if i < 0:
        return start_line
    depth = 0
    for j in range(i, len(text)):
        c = text[j]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text.count("\n", 0, j) + 1
    return text.count("\n") + 1


def _indented(line: str) -> bool:
    return line[:1] in (" ", "\t")
