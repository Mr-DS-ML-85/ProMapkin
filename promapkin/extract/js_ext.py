"""JavaScript / TypeScript extraction.

Lexical (not a full parser): comments and strings are masked, then declarations,
imports, routes and calls are recognised with brace matching. Facts carry the
`lexical` evidence class so traversal weights them below AST facts.
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath

from promapkin import evidence as ev
from promapkin.extract import Facts, is_test_path
from promapkin.extract.resources import AUTH_HINT, classify_call

KEYWORDS = {
    "if", "for", "while", "switch", "catch", "return", "function", "typeof", "new", "await", "super", "import",
    "require", "constructor", "else", "do", "try", "with", "yield", "delete", "void", "in", "of", "case",
    "describe", "it", "test", "expect",
}

IMPORT_RE = re.compile(r"""import\s+(?:type\s+)?(?:(?P<what>[\w*{}\s,$]+?)\s+from\s+)?['"](?P<src>[^'"]+)['"]""")
REQUIRE_RE = re.compile(r"""(?:const|let|var)\s+(?P<what>[\w{}\s,$:]+?)\s*=\s*require\(\s*['"](?P<src>[^'"]+)['"]\s*\)""")
CLASS_RE = re.compile(r"(?:export\s+)?(?:default\s+)?(?:abstract\s+)?class\s+(?P<name>[A-Za-z_$][\w$]*)"
                      r"(?:<[^>{]*>)?(?:\s+extends\s+(?P<base>[\w$.]+))?(?:\s+implements\s+(?P<impl>[\w$.,\s]+))?\s*\{")
FUNC_RE = re.compile(r"(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s*\*?\s*(?P<name>[A-Za-z_$][\w$]*)\s*"
                     r"(?:<[^>]*>)?\s*\((?P<params>[^)]*)\)[^{;]*\{")
ARROW_RE = re.compile(r"(?:export\s+)?(?:const|let|var)\s+(?P<name>[A-Za-z_$][\w$]*)\s*(?::[^=]+)?=\s*(?:async\s+)?"
                      r"(?:function\s*\*?\s*[\w$]*\s*)?\((?P<params>[^)]*)\)\s*(?::[^=]+)?(?:=>)?\s*\{")
METHOD_RE = re.compile(r"^\s*(?:public\s+|private\s+|protected\s+|static\s+|async\s+|readonly\s+|override\s+)*"
                       r"(?P<name>[A-Za-z_$][\w$]*)\s*(?:<[^>]*>)?\s*\((?P<params>[^)]*)\)\s*(?::[^{]+)?\{", re.M)
CALL_RE = re.compile(r"(?<![\w$.])(?P<expr>(?:this\.)?[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*)\s*\(")
ROUTE_RE = re.compile(r"(?P<router>[A-Za-z_$][\w$]*)\.(?P<verb>get|post|put|delete|patch|all|use)\(\s*"
                      r"['\"`](?P<route>/[^'\"`]*)['\"`]\s*,\s*(?P<rest>[^;]*?)\)", re.S)
ENV_RE = re.compile(r"process\.env\.(?P<key>[A-Z_][A-Z0-9_]*)|process\.env\[['\"](?P<key2>[^'\"]+)['\"]\]")


def _mask(text: str) -> str:
    """Blank out comments and string contents (keeping offsets) so braces inside them don't count."""
    out = list(text)
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if text.startswith("//", i):
            j = text.find("\n", i)
            j = n if j < 0 else j
            for k in range(i, j):
                out[k] = " "
            i = j
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            for k in range(i, j):
                if out[k] != "\n":
                    out[k] = " "
            i = j
        elif c in "'\"`":
            j = i + 1
            while j < n and text[j] != c:
                j += 2 if text[j] == "\\" else 1
            for k in range(i + 1, min(j, n)):
                if out[k] != "\n":
                    out[k] = " "
            i = j + 1
        else:
            i += 1
    return "".join(out)


def _block_end(masked: str, open_brace: int) -> int:
    depth = 0
    for i in range(open_brace, len(masked)):
        if masked[i] == "{":
            depth += 1
        elif masked[i] == "}":
            depth -= 1
            if depth == 0:
                return i
    return len(masked) - 1


def module_name(path: str) -> str:
    p = PurePosixPath(path).with_suffix("")
    if p.name == "index":
        p = p.parent
    return p.as_posix()


def extract(path: str, text: str) -> Facts:
    facts = Facts(path, "typescript" if path.endswith((".ts", ".tsx")) else "javascript")
    mod = module_name(path)
    mod_id = f"module:{mod}"
    is_test = is_test_path(path)
    masked = _mask(text)
    line_at = _line_index(text)

    imports: dict[str, str] = {}
    for rx in (IMPORT_RE, REQUIRE_RE):
        for m in rx.finditer(text):
            if masked[m.start()] == " ":   # inside a comment
                continue
            src = m.group("src")
            target = _resolve_import(path, src)
            what = (m.group("what") or "").replace("{", " ").replace("}", " ")
            for name in re.findall(r"[\w$]+(?:\s+as\s+[\w$]+)?|[\w$]+\s*:\s*[\w$]+", what):
                parts = re.split(r"\s+as\s+|\s*:\s*", name)
                local = parts[-1].strip()
                if local not in ("type", "from"):
                    imports[local] = f"{target}.{parts[0].strip()}" if "{" in (m.group("what") or "") else target
            facts.ref(mod_id, target, "imports", line_at(m.start()), module=mod)

    docblock = re.match(r"\s*/\*\*(.*?)\*/", text, re.S)
    facts.node(mod_id, "module", mod, ev.LEXICAL, qualname=mod, file=path, line_start=1,
               line_end=text.count("\n") + 1, text=_clean_doc(docblock.group(1)) if docblock else None,
               attrs={"imports": imports, "test": is_test, "loc": text.count("\n") + 1,
                      "entrypoint": PurePosixPath(path).stem in ("index", "main", "server", "app", "cli")})
    facts.edge(f"file:{path}", mod_id, "defines", ev.LEXICAL)

    spans: list[tuple[int, int, str]] = []   # (start, end, symbol id) for attributing calls

    for m in CLASS_RE.finditer(masked):
        name = m.group("name")
        start = m.end() - 1
        end = _block_end(masked, start)
        sid = f"symbol:{mod}:{name}"
        body = masked[start + 1:end]
        methods = [mm.group("name") for mm in METHOD_RE.finditer(body) if mm.group("name") not in KEYWORDS]
        facts.node(sid, "class", name, ev.LEXICAL, qualname=f"{mod}.{name}", file=path,
                   line_start=line_at(m.start()), line_end=line_at(end), text=_doc_before(text, m.start()),
                   attrs={"bases": [b for b in [m.group("base")] if b], "methods": methods, "symbol_type": "class",
                          "abstract": "abstract class" in masked[max(0, m.start() - 20):m.end()] or
                          "interface" in masked[max(0, m.start() - 12):m.start()],
                          "implements": [s.strip() for s in (m.group("impl") or "").split(",") if s.strip()]})
        facts.edge(mod_id, sid, "defines", ev.LEXICAL, line=line_at(m.start()))
        if m.group("base"):
            facts.ref(sid, m.group("base"), "inherits", line_at(m.start()), module=mod)
        for impl in (m.group("impl") or "").split(","):
            if impl.strip():
                facts.ref(sid, impl.strip(), "implements", line_at(m.start()), module=mod)
        for mm in METHOD_RE.finditer(body):
            mname = mm.group("name")
            if mname in KEYWORDS and mname != "constructor":
                continue
            mstart = start + 1 + mm.end() - 1
            mend = _block_end(masked, mstart)
            msid = f"symbol:{mod}:{name}.{mname}"
            facts.node(msid, "test" if is_test else "method", mname, ev.LEXICAL, qualname=f"{mod}.{name}.{mname}",
                       file=path, line_start=line_at(start + 1 + mm.start()), line_end=line_at(mend),
                       attrs={"params": _params(mm.group("params")), "class": name, "symbol_type": "method",
                              "public": not mname.startswith(("_", "#"))})
            facts.edge(sid, msid, "contains", ev.LEXICAL)
            spans.append((mstart, mend, msid))
            if mname == "constructor":
                for p in _params(mm.group("params")):
                    if p["type"] and p["type"][:1].isupper():
                        facts.ref(msid, p["type"], "references", line_at(mstart), module=mod, injected=True)

    for rx in (FUNC_RE, ARROW_RE):
        for m in rx.finditer(masked):
            name = m.group("name")
            if name in KEYWORDS:
                continue
            start = m.end() - 1
            end = _block_end(masked, start)
            if any(s <= m.start() <= e for s, e, _ in spans):
                continue   # nested inside an already-recorded method/function
            sid = f"symbol:{mod}:{name}"
            facts.node(sid, "test" if is_test else "function", name, ev.LEXICAL, qualname=f"{mod}.{name}",
                       file=path, line_start=line_at(m.start()), line_end=line_at(end),
                       text=_doc_before(text, m.start()),
                       attrs={"params": _params(m.group("params")), "symbol_type": "function",
                              "public": "export" in masked[m.start():m.start() + 8],
                              "async": "async" in masked[m.start():m.end()]})
            facts.edge(mod_id, sid, "defines", ev.LEXICAL, line=line_at(m.start()))
            spans.append((start, end, sid))

    def owner_at(pos: int) -> str:
        best = None
        for s, e, sid in spans:
            if s <= pos <= e and (best is None or s >= best[0]):
                best = (s, sid)
        return best[1] if best else mod_id

    for m in ROUTE_RE.finditer(text):
        if masked[m.start()] == " " or m.group("verb") == "use":
            continue
        verb = "ANY" if m.group("verb") == "all" else m.group("verb").upper()
        route = m.group("route")
        api_id = f"api:{verb} {route}"
        handlers = [h for h in re.findall(r"[A-Za-z_$][\w$.]*", m.group("rest")) if h not in KEYWORDS
                    and h not in ("req", "res", "next", "async", "await")]
        auth = [h for h in handlers if AUTH_HINT.search(h)]
        facts.node(api_id, "endpoint", f"{verb} {route}", ev.LEXICAL, file=path, line_start=line_at(m.start()),
                   attrs={"method": verb, "route": route, "router": m.group("router"), "auth": auth})
        facts.edge(mod_id, api_id, "exposes", ev.LEXICAL, line=line_at(m.start()))
        for h in handlers:
            if h not in auth:
                facts.ref(api_id, h, "routes_to", line_at(m.start()), module=mod)

    for m in CALL_RE.finditer(masked):
        expr = m.group("expr")
        head = expr.split(".")[0]
        if head in KEYWORDS or expr in KEYWORDS:
            continue
        before = masked[max(0, m.start() - 10):m.start()]
        if re.search(r"(function\s*\*?|class|const|let|var)\s*$", before):
            continue   # a declaration, not a call
        if re.match(r"\s*(:[^{]*)?\{", masked[_block_end_paren(masked, m.end() - 1) + 1:][:80]) and \
                not re.search(r"[=(,]\s*$", masked[max(0, m.start() - 3):m.start()]):
            continue   # method definition "name(...) {"
        owner = owner_at(m.start())
        line = line_at(m.start())
        call = expr.replace("this.", "self.", 1) if expr.startswith("this.") else expr
        res = classify_call(call, imports.get(head))
        if res:
            resource, verb = res
            facts.node(f"resource:{resource}", "resource", resource, ev.HEURISTIC, attrs={"category": resource})
            facts.edge(owner, f"resource:{resource}", verb, ev.HEURISTIC, line=line, confidence=0.65,
                       attrs={"call": expr, "classifier": "heuristic"})
        elif "." in call and head in imports and not imports[head].startswith((".", "/")) and \
                not imports[head].startswith(mod.split("/")[0] + "/"):
            facts.ref(owner, call, "external_call", line, module=mod, library=imports[head])
        facts.ref(owner, call, "calls", line, module=mod)

    for m in ENV_RE.finditer(text):
        key = m.group("key") or m.group("key2")
        facts.node(f"config:{key}", "config", key, ev.LEXICAL, attrs={"source": "environment"})
        facts.edge(owner_at(m.start()), f"config:{key}", "reads", ev.LEXICAL, line=line_at(m.start()))
    return facts


def _block_end_paren(masked: str, open_paren: int) -> int:
    depth = 0
    for i in range(open_paren, min(len(masked), open_paren + 2000)):
        if masked[i] == "(":
            depth += 1
        elif masked[i] == ")":
            depth -= 1
            if depth == 0:
                return i
    return open_paren


def _resolve_import(path: str, src: str) -> str:
    if src.startswith("."):
        base = PurePosixPath(path).parent
        parts: list[str] = list(base.parts)
        for seg in src.split("/"):
            if seg == "..":
                parts = parts[:-1]
            elif seg not in (".", ""):
                parts.append(seg)
        target = PurePosixPath(*parts) if parts else PurePosixPath(".")
        target = target.with_suffix("") if target.suffix in (".js", ".ts", ".jsx", ".tsx", ".mjs") else target
        return "./" + target.as_posix()
    return src


def _params(raw: str) -> list[dict]:
    out = []
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        piece = re.sub(r"^(public|private|protected|readonly)\s+", "", piece)
        name, _, typ = piece.partition(":")
        name = name.split("=")[0].strip().rstrip("?")
        out.append({"name": name, "type": typ.split("=")[0].strip() or None})
    return out


def _line_index(text: str):
    starts = [0] + [i + 1 for i, c in enumerate(text) if c == "\n"]
    import bisect

    def line_at(pos: int) -> int:
        return bisect.bisect_right(starts, pos)
    return line_at


def _doc_before(text: str, pos: int) -> str | None:
    m = re.search(r"/\*\*(.*?)\*/\s*$", text[max(0, pos - 1500):pos], re.S)
    return _clean_doc(m.group(1)) if m else None


def _clean_doc(raw: str) -> str:
    return " ".join(line.strip().lstrip("*").strip() for line in raw.splitlines()).strip()
