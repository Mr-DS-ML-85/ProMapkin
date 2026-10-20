"""Multi-language extraction with tree-sitter grammars (Go, Rust, Java, C#, Kotlin, Ruby, PHP, C,
C++, Swift, Scala). Real parse trees, so facts carry `ast` evidence; cross-file calls are
resolved later by the scanner's resolver (same-module/class first, then imports, then names).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any

from promapkin import evidence as ev
from promapkin.extract import Facts, is_test_path
from promapkin.extract.resources import classify_call

NAME_TYPES = {"identifier", "type_identifier", "simple_identifier", "constant", "name", "field_identifier",
              "property_identifier", "namespace_identifier"}


@dataclass
class Spec:
    grammar: str
    classes: set[str]
    funcs: set[str]
    imports: set[str]
    calls: set[str]
    news: set[str] = field(default_factory=set)
    inherit: set[str] = field(default_factory=set)
    impl: set[str] = field(default_factory=set)


SPECS: dict[str, Spec] = {
    "go": Spec("go", {"type_spec"}, {"function_declaration", "method_declaration", "method_elem"},
               {"import_spec"}, {"call_expression"}, {"composite_literal"}),
    "rust": Spec("rust", {"struct_item", "enum_item", "trait_item", "union_item"},
                 {"function_item", "function_signature_item"}, {"use_declaration"}, {"call_expression"},
                 {"struct_expression"}, impl={"impl_item"}),
    "java": Spec("java", {"class_declaration", "interface_declaration", "enum_declaration", "record_declaration"},
                 {"method_declaration", "constructor_declaration"}, {"import_declaration"}, {"method_invocation"},
                 {"object_creation_expression"}, {"superclass", "super_interfaces", "extends_interfaces"}),
    "csharp": Spec("csharp", {"class_declaration", "interface_declaration", "struct_declaration",
                              "record_declaration", "enum_declaration"},
                   {"method_declaration", "constructor_declaration"}, {"using_directive"}, {"invocation_expression"},
                   {"object_creation_expression"}, {"base_list"}),
    "kotlin": Spec("kotlin", {"class_declaration", "object_declaration"}, {"function_declaration"},
                   {"import_header"}, {"call_expression"}, set(), {"delegation_specifier"}),
    "ruby": Spec("ruby", {"class", "module"}, {"method", "singleton_method"}, set(), {"call"}, set(), {"superclass"}),
    "php": Spec("php", {"class_declaration", "interface_declaration", "trait_declaration", "enum_declaration"},
                {"method_declaration", "function_definition"}, {"namespace_use_clause"},
                {"function_call_expression", "member_call_expression", "scoped_call_expression"},
                {"object_creation_expression"}, {"base_clause", "class_interface_clause"}),
    "c": Spec("c", {"struct_specifier"}, {"function_definition"}, {"preproc_include"}, {"call_expression"}),
    "cpp": Spec("cpp", {"class_specifier", "struct_specifier"}, {"function_definition"}, {"preproc_include"},
                {"call_expression"}, {"new_expression"}, {"base_class_clause"}),
    "swift": Spec("swift", {"class_declaration", "protocol_declaration"}, {"function_declaration", "init_declaration"},
                  {"import_declaration"}, {"call_expression"}, set(), {"inheritance_specifier"}),
    "scala": Spec("scala", {"class_definition", "object_definition", "trait_definition"},
                  {"function_definition", "function_declaration"}, {"import_declaration"}, {"call_expression"},
                  {"instance_expression"}, {"extends_clause"}),
}
SPECS["cuda"] = Spec("cuda", SPECS["cpp"].classes, SPECS["cpp"].funcs, SPECS["cpp"].imports, SPECS["cpp"].calls,
                     SPECS["cpp"].news, SPECS["cpp"].inherit)
SPECS["glsl"] = Spec("glsl", {"struct_specifier"}, {"function_definition"}, {"preproc_include"}, {"call_expression"})
SPECS["hlsl"] = Spec("hlsl", {"struct_specifier"}, {"function_definition"}, {"preproc_include"}, {"call_expression"})
SPECS["apex"] = Spec("apex", SPECS["java"].classes | {"trigger_declaration"}, SPECS["java"].funcs, set(),
                     SPECS["java"].calls, SPECS["java"].news, {"superclass", "interfaces"})
TEST_FUNC = re.compile(r"^(test|Test)|_test$|Test$")
IDENT_CHAIN = re.compile(r"[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*")
_parsers: dict[str, Any] = {}


def module_name(path: str) -> str:
    return PurePosixPath(path).with_suffix("").as_posix()


def _parser(grammar: str) -> Any:
    if grammar not in _parsers:
        from tree_sitter_language_pack import get_parser
        _parsers[grammar] = get_parser(grammar)
    return _parsers[grammar]


def _text(n: Any) -> str:
    return n.text.decode("utf-8", errors="replace") if n is not None else ""


def _name(n: Any) -> str | None:
    named = n.child_by_field_name("name")
    if named is not None:
        return _text(named)
    decl = n.child_by_field_name("declarator")
    while decl is not None:   # C/C++ declarator chains
        if decl.type in ("identifier", "field_identifier", "qualified_identifier", "destructor_name",
                         "operator_name"):
            return _text(decl)
        decl = decl.child_by_field_name("declarator") or next(
            (c for c in decl.named_children if c.type in ("identifier", "field_identifier", "qualified_identifier")),
            None)
    for c in n.named_children:
        if c.type in NAME_TYPES:
            return _text(c)
    return None


def _norm_call(text: str) -> str:
    t = text.replace("::", ".").replace("->", ".").replace("?.", ".").replace("!!.", ".")
    t = re.sub(r"<[^<>]*>", "", t)
    t = re.sub(r"^\$this\.", "self.", t)
    t = re.sub(r"^this\.", "self.", t)
    t = t.lstrip("&*$")
    m = IDENT_CHAIN.match(t)
    return m.group(0) if m else ""


def _call_target(n: Any, lang: str) -> str:
    if lang == "java" and n.type == "method_invocation":
        obj, name = n.child_by_field_name("object"), n.child_by_field_name("name")
        return _norm_call(f"{_text(obj)}.{_text(name)}" if obj is not None else _text(name))
    if lang == "ruby":
        recv, meth = n.child_by_field_name("receiver"), n.child_by_field_name("method")
        return _norm_call(f"{_text(recv)}.{_text(meth)}" if recv is not None else _text(meth))
    if lang == "php" and n.type in ("member_call_expression", "scoped_call_expression"):
        obj = n.child_by_field_name("object") or n.child_by_field_name("scope")
        return _norm_call(f"{_text(obj)}.{_text(n.child_by_field_name('name'))}")
    fn = n.child_by_field_name("function") or n.child_by_field_name("name")
    if fn is None and n.named_children:
        fn = n.named_children[0]
    return _norm_call(_text(fn))


def extract(path: str, text: str, language: str) -> Facts:
    spec = SPECS[language]
    facts = Facts(path, language)
    mod = module_name(path)
    mod_id = f"module:{mod}"
    is_test = is_test_path(path)
    parser = _parser(spec.grammar)   # a grammar that cannot load propagates: the caller falls back to lexical
    try:
        tree = parser.parse(text.encode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        facts.errors.append(f"tree-sitter parse failed: {exc}")
        return facts
    lines = text.count("\n") + 1
    facts.node(mod_id, "module", mod, ev.AST, qualname=mod, file=path, line_start=1, line_end=lines,
               text=_leading_comment(text), attrs={"test": is_test, "loc": lines, "parser": f"tree-sitter:{language}",
                                                   "imports": {}, "entrypoint": _is_entry(language, text, path)})
    facts.edge(f"file:{path}", mod_id, "defines", ev.AST)
    _Walker(facts, spec, language, mod, path, is_test).walk(tree.root_node, [])
    if tree.root_node.has_error:
        facts.errors.append("syntax errors present; partial extraction")
    return facts


def _leading_comment(text: str) -> str | None:
    m = re.match(r"\s*((?:(?://|#)[^\n]*\n)+|/\*.*?\*/)", text, re.S)
    if not m:
        return None
    body = re.sub(r"^\s*(//+|#+|/\*+|\*+/?)", "", m.group(1), flags=re.M)
    return " ".join(body.split())[:500] or None


def _is_entry(language: str, text: str, path: str) -> bool:
    return bool(
        (language in ("cuda", "glsl", "hlsl") and re.search(r"\bvoid\s+main\s*\(|\bint\s+main\s*\(", text))
        or (language == "go" and re.search(r"^package main\b", text, re.M) and "func main(" in text)
        or (language == "rust" and PurePosixPath(path).name == "main.rs")
        or (language in ("java", "kotlin", "scala", "csharp") and re.search(r"\b(static\s+void\s+[Mm]ain|fun main|def main)\b", text))
        or (language in ("c", "cpp") and re.search(r"\bint\s+main\s*\(", text))
    )


class _Walker:
    def __init__(self, facts: Facts, spec: Spec, lang: str, mod: str, path: str, is_test: bool):
        self.f, self.spec, self.lang, self.mod, self.path, self.is_test = facts, spec, lang, mod, path, is_test
        self.mod_id = f"module:{mod}"

    def sid(self, qual: str) -> str:
        return f"symbol:{self.mod}:{qual}"

    def walk(self, node: Any, scope: list[tuple[str, str]]) -> None:
        t = node.type
        if t in self.spec.imports or (self.lang == "ruby" and t == "call" and
                                      _text(node.child_by_field_name("method")) in ("require", "require_relative")):
            self._import(node)
            return
        if t in self.spec.impl:
            self._impl(node, scope)
            return
        if t in self.spec.classes and self._has_body(node):
            name = _name(node)
            if name:
                qual = ".".join([q for q, _ in scope] + [name])
                sid = self.sid(qual)
                methods = self._member_names(node)
                abstract = t in ("interface_declaration", "trait_item", "trait_definition", "protocol_declaration") \
                    or "abstract" in _text(node)[:60]
                self.f.node(sid, "test" if self.is_test and TEST_FUNC.search(name) else "class", name, ev.AST,
                            qualname=f"{self.mod}.{qual}", file=self.path, line_start=node.start_point[0] + 1,
                            line_end=node.end_point[0] + 1,
                            attrs={"symbol_type": "class", "methods": methods, "abstract": abstract,
                                   "bases": self._bases(node), "kind": t})
                self.f.edge(self.sid(scope[-1][0]) if scope else self.mod_id, sid,
                            "contains" if scope else "defines", ev.AST, line=node.start_point[0] + 1)
                for base in self._bases(node):
                    self.f.ref(sid, base, "inherits", node.start_point[0] + 1, module=self.mod)
                for c in node.children:
                    self.walk(c, scope + [(qual, "class")])
                return
        if t in self.spec.funcs:
            self._func(node, scope)
            return
        if t in self.spec.calls or t in self.spec.news:
            self._call(node, scope)
        for c in node.children:
            self.walk(c, scope)

    def _has_body(self, node: Any) -> bool:
        if self.lang in ("c", "cpp", "cuda", "glsl", "hlsl") and node.type == "struct_specifier":
            return any(c.type == "field_declaration_list" for c in node.children)
        return True

    def _member_names(self, node: Any) -> list[str]:
        names = []
        stack = list(node.children)
        while stack:
            n = stack.pop()
            if n.type in self.spec.funcs:
                nm = _name(n)
                if nm:
                    names.append(nm.split("::")[-1])
                continue
            if n.type in self.spec.classes:
                continue
            stack.extend(n.children)
        return sorted(set(names))

    def _bases(self, node: Any) -> list[str]:
        out = []
        for c in node.children:
            if c.type in self.spec.inherit:
                for m in re.finditer(r"[A-Z][\w.]*", _text(c)):
                    base = m.group(0).split("(")[0]
                    if base not in out and base not in ("Unit",):
                        out.append(base)
        return out

    def _impl(self, node: Any, scope: list) -> None:   # Rust: impl [Trait for] Type { fns }
        typ = node.child_by_field_name("type")
        trait = node.child_by_field_name("trait")
        tname = re.sub(r"<.*", "", _text(typ)) if typ is not None else None
        if tname and trait is not None:
            self.f.ref(self.sid(tname), re.sub(r"<.*", "", _text(trait)), "implements", node.start_point[0] + 1,
                       module=self.mod)
        for c in node.children:
            self.walk(c, scope + ([(tname, "class")] if tname else []))

    def _func(self, node: Any, scope: list) -> None:
        name = _name(node)
        if not name:
            return
        owner_class = scope[-1][0] if scope and scope[-1][1] == "class" else None
        if self.lang == "go" and node.type == "method_declaration":
            recv = node.child_by_field_name("receiver")
            m = re.search(r"([A-Z_a-z]\w*)\s*\)?\s*$", _text(recv).replace("*", " ").rstrip(")")) if recv else None
            if m:
                owner_class = m.group(1)
        if "::" in name:   # C++ out-of-line definition: Class::method
            owner_class, name = name.rsplit("::", 1)
            owner_class = owner_class.replace("::", ".")
        qual = f"{owner_class}.{name}" if owner_class else name
        if scope and scope[-1][1] == "function":
            qual = f"{scope[-1][0]}.{name}"
        sid = self.sid(qual)
        kind = "test" if self.is_test and TEST_FUNC.search(name) else ("method" if owner_class else "function")
        params = self._params(node)
        self.f.node(sid, kind, name, ev.AST, qualname=f"{self.mod}.{qual}", file=self.path,
                    line_start=node.start_point[0] + 1, line_end=node.end_point[0] + 1,
                    text=_doc_before(node), attrs={"symbol_type": kind, "params": params, "class": owner_class,
                                                    "public": not name.startswith("_") and name[:1] != "#",
                                                    "complexity": 1 + _text(node).count("if ") +
                                                    _text(node).count("for ") + _text(node).count("while ")})
        parent = self.sid(owner_class) if owner_class and any(q == owner_class for q, _ in scope) else self.mod_id
        self.f.edge(parent, sid, "contains" if parent != self.mod_id else "defines", ev.AST,
                    line=node.start_point[0] + 1)
        for p in params:
            if p.get("type") and p["type"][:1].isupper():
                self.f.ref(sid, re.sub(r"[<\[(].*", "", p["type"]).lstrip("*&"), "references",
                           node.start_point[0] + 1, module=self.mod, cls=owner_class,
                           injected=name in ("constructor", "__construct", "init", "New") or
                           (owner_class is not None and name == owner_class))
        for c in node.children:
            self.walk(c, scope + [(qual, "function")])

    def _params(self, node: Any) -> list[dict[str, Any]]:
        plist = node.child_by_field_name("parameters")
        if plist is None:
            plist = next((c for c in node.children if "parameter" in c.type), None)
        out = []
        for p in (plist.named_children if plist is not None else []):
            nm = p.child_by_field_name("name") or p.child_by_field_name("pattern")
            ty = p.child_by_field_name("type")
            if nm is None:
                idents = [c for c in p.named_children if c.type in NAME_TYPES]
                nm = idents[-1] if idents else None
                ty = ty or (idents[0] if len(idents) > 1 else None)
            if nm is not None:
                out.append({"name": _text(nm), "type": _text(ty) if ty is not None else None})
        return out[:20]

    def _call(self, node: Any, scope: list) -> None:
        owner = self.sid(scope[-1][0]) if scope and scope[-1][1] == "function" else self.mod_id
        cls = next((q for q, k in reversed(scope) if k == "class"), None)
        if node.type in self.spec.news:
            typ = node.child_by_field_name("type") or next(iter(node.named_children), None)
            target = _norm_call(_text(typ))
        else:
            target = _call_target(node, self.lang)
        if not target:
            return
        line = node.start_point[0] + 1
        res = classify_call(target)
        if res:
            resource, verb = res
            self.f.node(f"resource:{resource}", "resource", resource, ev.HEURISTIC, attrs={"category": resource})
            self.f.edge(owner, f"resource:{resource}", verb, ev.HEURISTIC, line=line, confidence=0.65,
                        attrs={"call": target, "classifier": "heuristic"})
        self.f.ref(owner, target, "calls", line, module=self.mod, cls=cls)

    def _import(self, node: Any) -> None:
        raw = _text(node)
        m = re.search(r"[\"'<]([^\"'>]+)[\"'>]", raw)
        if m:
            target = m.group(1)
        else:
            target = re.sub(r"^\s*(import|using|use|static)\s+", "", raw).strip().rstrip(";").split(" as ")[0]
        target = re.sub(r"\.(h|hpp|hh|rb|php)$", "", target).replace("::", ".").replace("\\", ".").strip("{}* .")
        if target:
            self.f.ref(self.mod_id, target, "imports", node.start_point[0] + 1, module=self.mod)


def _doc_before(node: Any) -> str | None:
    prev = node.prev_named_sibling
    if prev is not None and "comment" in prev.type and prev.end_point[0] >= node.start_point[0] - 1:
        body = re.sub(r"^\s*(///?|#+|/\*+|\*+/?|--)", "", _text(prev), flags=re.M)
        return " ".join(body.split())[:400] or None
    return None
