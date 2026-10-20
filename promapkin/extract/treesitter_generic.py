"""Tree-sitter extraction for the long tail of languages (Dart, Lua, Elixir, Erlang, Haskell, OCaml, F#,
Clojure, Lisps, R, Julia, Perl, Bash, PowerShell, Zig, Nim, Objective-C, Elm, Solidity, Crystal, D,
Fortran, Pascal, MATLAB, Verilog, VHDL, Gleam, Odin, V).

Each language is a declarative spec: which node types declare containers (classes, modules, records),
functions, imports and calls, and how to read a name from each. Grammars differ wildly, so names are
read by small getters (a field, or the first identifier-like descendant). Languages whose definitions are
ordinary calls (Elixir `def`, Lisp `defun`/`define`) use a form reader instead. Calls the grammar does not
expose as call nodes are read lexically from the function body. Facts carry `ast` evidence for
declarations and imports; lexically read calls carry `lexical` evidence on their references.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

from promapkin import evidence as ev
from promapkin.extract import Facts, is_test_path
from promapkin.extract.resources import classify_call
from promapkin.extract import treesitter_ext as _core
from promapkin.extract.treesitter_ext import TEST_FUNC, _doc_before, _leading_comment, _norm_call, _text, module_name

Getter = Callable[[Any], "str | None"]

IDENTS = ("identifier", "type_identifier", "simple_identifier", "constant", "name", "simple_name", "bareword",
          "upper_case_identifier", "lower_case_identifier", "value_name", "module_name", "type_constructor",
          "constructor", "atom", "ident", "IDENTIFIER", "function_name", "sym_name", "symbol", "module_identifier",
          "upper_case_qid", "constructor_name", "module_id", "type_name", "variable", "package")
CALL_RE = re.compile(r"(?<![\w$.:])((?:[A-Za-z_$][\w$]*(?:\.|::|->|:))*[A-Za-z_$][\w$]*)\s*\(")
HYPHEN_NAMES = {"powershell", "bash", "clojure", "commonlisp", "scheme", "racket"}
NOT_CALLS = {"if", "for", "while", "switch", "catch", "return", "function", "fn", "def", "proc", "func", "new",
             "sizeof", "match", "when", "elif", "else", "and", "or", "not", "super", "this", "self", "typeof", "case",
             "do", "end", "then", "local", "let", "in", "of", "begin", "unless", "until", "elsif", "foreach"}


# ------------------------------------------------------------------ getters
def fld(*names: str) -> Getter:
    def get(n: Any) -> str | None:
        for name in names:
            c = n.child_by_field_name(name)
            if c is not None:
                return _text(c)
        return None
    return get


def desc(*types: str, depth: int = 3) -> Getter:
    """Text of the first descendant (breadth-first, bounded depth) whose type is one of `types`."""
    want = set(types or IDENTS)

    def get(n: Any) -> str | None:
        level = list(n.named_children)
        for _ in range(depth):
            nxt = []
            for c in level:
                if c.type in want:
                    return _text(c)
                nxt.extend(c.named_children)
            level = nxt
        return None
    return get


def either(*getters: Getter) -> Getter:
    def get(n: Any) -> str | None:
        for g in getters:
            v = g(n)
            if v:
                return v
        return None
    return get


NAME = either(fld("name"), desc())


def import_target(n: Any) -> str | None:
    """Import targets: a quoted path when present, else the text after the import keyword."""
    raw = _text(n)
    m = re.search(r"[\"'<]([^\"'>\n]+)[\"'>]", raw)
    if m:
        return m.group(1)
    raw = re.sub(r"^\s*-?(Import-Module|import|using|use|uses|open|require|include|library|source|alias|from|"
                 r"cimport)\b[\s(]*", "", raw, flags=re.I)
    m = re.match(r"[\w.:/\\@-]+", raw.strip())
    return m.group(0) if m else None


def call_target(n: Any) -> str | None:
    fn = (n.child_by_field_name("function") or n.child_by_field_name("name") or n.child_by_field_name("target")
          or n.child_by_field_name("method") or n.child_by_field_name("expr") or n.child_by_field_name("callee"))
    recv = n.child_by_field_name("receiver") or n.child_by_field_name("invocant")
    if fn is None:
        named = n.named_children
        fn = named[0] if named else None
    text = _text(fn)
    if recv is not None:
        text = f"{_text(recv)}.{text}"
    return _norm_call(text.replace("::", ".").replace(":", ".").replace("$", "."))


@dataclass
class LangSpec:
    grammar: str
    classes: dict[str, Getter] = field(default_factory=dict)
    funcs: dict[str, Getter] = field(default_factory=dict)
    imports: dict[str, Getter] = field(default_factory=dict)
    calls: dict[str, Getter] = field(default_factory=dict)
    body_sibling: set[str] = field(default_factory=set)       # function bodies that follow the signature node
    forms: Callable[[Any], "tuple[str, str] | None"] | None = None
    lexical_calls: bool = False
    entry: str | None = None                                   # regex marking an entry point


def _elixir_form(n: Any) -> tuple[str, str] | None:
    if n.type != "call":
        return None
    head = _text(n.child_by_field_name("target"))
    args = next((c for c in n.named_children if c.type == "arguments"), None)
    first = args.named_children[0] if args is not None and args.named_children else None
    if first is None:
        return None
    if head in ("defmodule", "defprotocol", "defimpl"):
        return "class", _text(first)
    if head in ("def", "defp", "defmacro", "defmacrop", "defguard", "defdelegate"):
        name = _text(first.child_by_field_name("target")) if first.type == "call" else _text(first)
        return "func", name.split(" ")[0]
    if head in ("alias", "import", "require", "use"):
        return "import", _text(first)
    return None


LISP_FUNC = {"defn", "defn-", "defmacro", "defun", "defmethod", "define", "define-syntax", "defgeneric",
             "define-public", "lambda*"}
LISP_CLASS = {"defrecord", "deftype", "defprotocol", "defclass", "defstruct", "struct", "define-record-type",
              "defpackage", "ns", "define-module", "module"}
LISP_IMPORT = {"require", "import", "use", "use-modules", ":require", ":import"}


def _lisp_form(n: Any) -> tuple[str, str] | None:
    if n.type not in ("list_lit", "list"):
        return None
    items = [c for c in n.named_children if c.type not in ("comment", "meta_lit")]
    if n.type == "list_lit" and items and items[0].type == "defun":   # Common Lisp grammar
        header = items[0].child_by_field_name("header") or next(
            (c for c in items[0].named_children if c.type == "defun_header"), None)
        name = header.child_by_field_name("function_name") if header is not None else None
        return ("func", _text(name)) if name is not None else None
    if len(items) < 2:
        return None
    head = _text(items[0])
    second = items[1]
    if second.type in ("list", "list_lit") and second.named_children:   # (define (name args) ...)
        name = _text(second.named_children[0])
    else:
        name = _text(second)
    if head in LISP_FUNC:
        return "func", name
    if head in LISP_CLASS:
        return "class", name
    if head in LISP_IMPORT:
        return "import", name.strip("'[]():")
    return None


def _lisp_call(n: Any) -> str | None:
    parent = n.parent
    if parent is not None and parent.type == "defun_header":
        return None   # Common Lisp lambda list
    if parent is not None and _lisp_form(parent) and parent.named_children[1:2] and \
            parent.named_children[1].id == n.id:
        return None   # the (name args) head of a definition, not a call
    items = [c for c in n.named_children if c.type not in ("comment",)]
    if not items or items[0].type not in ("sym_lit", "symbol"):
        return None
    return _norm_call(_text(items[0]).replace("/", "."))


def _julia_name(n: Any) -> str | None:
    sig = next((c for c in n.named_children if c.type == "signature"), None)
    return desc("identifier", depth=2)(sig) if sig is not None else None


def _c_name(n: Any) -> str | None:
    from promapkin.extract.treesitter_ext import _name
    return _name(n)


def _descendants(n: Any, depth: int) -> list[Any]:
    out, level = [], list(n.named_children)
    for _ in range(depth):
        out.extend(level)
        level = [c for x in level for c in x.named_children]
    return out


C_LIKE_CALLS = {"call_expression": call_target}
SPECS: dict[str, LangSpec] = {
    "dart": LangSpec("dart", classes={t: fld("name") for t in ("class_definition", "mixin_declaration",
                                                               "extension_declaration", "enum_declaration")},
                     funcs={"function_signature": fld("name"), "constructor_signature": fld("name"),
                            "getter_signature": fld("name"), "setter_signature": fld("name")},
                     imports={"import_or_export": import_target}, body_sibling={"function_body"},
                     lexical_calls=True, entry=r"\bvoid\s+main\s*\("),
    "lua": LangSpec("lua", funcs={"function_declaration": fld("name")},
                    imports={"function_call": lambda n: import_target(n.child_by_field_name("arguments") or n)
                             if _text(n.child_by_field_name("name")) in ("require", "dofile") else None},
                    calls={"function_call": call_target}),
    "elixir": LangSpec("elixir", forms=_elixir_form, calls={"call": call_target}),
    "erlang": LangSpec("erlang", classes={"module_attribute": fld("name")},
                       funcs={"function_clause": fld("name")},
                       imports={"import_attribute": fld("module"), "pp_include": import_target,
                                "pp_include_lib": import_target},
                       calls={"call": lambda n: None if n.parent is not None and n.parent.type == "remote"
                              else _norm_call(_text(n.child_by_field_name("expr"))),
                              "remote": lambda n: _norm_call(_text(n).split("(")[0].replace(":", "."))}),
    "haskell": LangSpec("haskell", classes={"data_type": fld("name"), "newtype": fld("name"), "class": fld("name"),
                                            "type_synomym": fld("name"), "type_synonym": fld("name")},
                        funcs={"function": fld("name"), "bind": fld("name")},
                        imports={"import": fld("module")},
                        calls={"apply": lambda n: _norm_call(_text(n.child_by_field_name("function")))},
                        entry=r"^main\s*="),
    "ocaml": LangSpec("ocaml", classes={"module_binding": desc("module_name", depth=1),
                                        "type_binding": fld("name"), "class_binding": fld("name")},
                      funcs={"let_binding": either(fld("pattern"), desc("value_name", depth=1))},
                      imports={"open_module": fld("module")},
                      calls={"application_expression": lambda n: _norm_call(_text(n.child_by_field_name("function")))}),
    "fsharp": LangSpec("fsharp", classes={"type_definition": desc("type_name", "identifier", depth=3),
                                          "module_defn": desc("identifier", depth=2)},
                       funcs={"method_or_prop_defn": lambda n: _text(n.child_by_field_name("name")),
                              "function_or_value_defn": desc("identifier", depth=2)},
                       imports={"import_decl": import_target},
                       calls={"application_expression": lambda n: _norm_call(_text(n.named_children[0]))
                              if n.named_children else None}),
    "clojure": LangSpec("clojure", forms=_lisp_form, calls={"list_lit": _lisp_call}, entry=r"\(defn\s+-main\b"),
    "commonlisp": LangSpec("commonlisp", forms=_lisp_form, calls={"list_lit": _lisp_call}),
    "scheme": LangSpec("scheme", forms=_lisp_form, calls={"list": _lisp_call}),
    "racket": LangSpec("racket", forms=_lisp_form, calls={"list": _lisp_call}),
    "r": LangSpec("r", funcs={"binary_operator": lambda n: _text(n.child_by_field_name("lhs")) if (
        (r := n.child_by_field_name("rhs")) is not None and r.type == "function_definition") else None},
        imports={"call": lambda n: _text(n.child_by_field_name("arguments")).strip("()'\" ")
                 if _text(n.child_by_field_name("function")) in ("library", "require", "source") else None},
        calls={"call": call_target}),
    "julia": LangSpec("julia", classes={"module_definition": fld("name"), "struct_definition": desc("identifier", depth=2),
                                        "abstract_definition": desc("identifier", depth=2)},
                      funcs={"function_definition": lambda n: _julia_name(n),
                             "macro_definition": desc("identifier", depth=2)},
                      imports={"using_statement": import_target, "import_statement": import_target},
                      calls={"call_expression": call_target}),
    "perl": LangSpec("perl", classes={"package_statement": fld("name")},
                     funcs={"subroutine_declaration_statement": fld("name")},
                     imports={"use_statement": fld("module"), "require_expression": import_target},
                     calls={"function_call_expression": call_target, "method_call_expression": call_target,
                            "ambiguous_function_call_expression": call_target}),
    "bash": LangSpec("bash", funcs={"function_definition": fld("name")},
                     imports={"command": lambda n: import_target(n.child_by_field_name("argument") or n)
                              if _text(n.child_by_field_name("name")) in ("source", ".") else None},
                     calls={"command": lambda n: _text(n.child_by_field_name("name"))
                            if _text(n.child_by_field_name("name")) not in ("source", ".") else None}),
    "powershell": LangSpec("powershell", classes={"class_statement": desc("simple_name", depth=1)},
                           funcs={"function_statement": desc("function_name", depth=1),
                                  "class_method_definition": desc("simple_name", depth=1)},
                           imports={"command": lambda n: import_target(n)
                                    if _text(n.child_by_field_name("command_name")).lower() == "import-module"
                                    else None},
                           calls={"command": lambda n: _text(n.child_by_field_name("command_name"))}),
    "zig": LangSpec("zig", classes={"VarDecl": lambda n: _text(n.named_children[0]) if any(
        c.type == "ContainerDecl" for c in _descendants(n, 3)) and n.named_children else None},
        funcs={"FnProto": fld("function")}, body_sibling={"Block"}, lexical_calls=True,
        imports={"VarDecl": lambda n: (m.group(1) if (m := re.search(r'@import\("([^"]+)"\)', _text(n))) else None)},
        entry=r"pub\s+fn\s+main\s*\("),
    "nim": LangSpec("nim", classes={"typeDef": desc("ident", depth=2)},
                    funcs={"routine": desc("ident", depth=2)},
                    imports={"importStmt": import_target, "fromStmt": import_target, "includeStmt": import_target},
                    lexical_calls=True),
    "objc": LangSpec("objc", classes={"class_interface": desc("identifier", depth=1),
                                      "class_implementation": desc("identifier", depth=1),
                                      "protocol_declaration": desc("identifier", depth=1)},
                     funcs={"method_definition": desc("identifier", depth=1), "function_definition": _c_name},
                     imports={"preproc_include": import_target},
                     calls={"message_expression": lambda n: _norm_call(
                         f"{_text(n.child_by_field_name('receiver'))}.{_text(n.child_by_field_name('method'))}"),
                         "call_expression": call_target}),
    "elm": LangSpec("elm", classes={"type_declaration": fld("name"), "type_alias_declaration": fld("name")},
                    funcs={"value_declaration": lambda n: desc("lower_case_identifier", depth=2)(n)},
                    imports={"import_clause": fld("moduleName")},
                    calls={"function_call_expr": lambda n: _norm_call(_text(n.named_children[0]))
                           if n.named_children else None}),
    "solidity": LangSpec("solidity", classes={t: fld("name") for t in ("contract_declaration", "interface_declaration",
                                                                       "library_declaration", "struct_declaration")},
                         funcs={"function_definition": fld("name"), "modifier_definition": fld("name"),
                                "constructor_definition": lambda n: "constructor"},
                         imports={"import_directive": fld("source")}, calls={"call_expression": call_target}),
    "crystal": LangSpec("crystal", classes={t: fld("name") for t in ("class_def", "module_def", "struct_def")},
                        funcs={"method_def": fld("name"), "fun_def": fld("name")},
                        imports={"require": import_target}, calls={"call": call_target}),
    "d": LangSpec("d", classes={t: desc("identifier", depth=1) for t in ("class_declaration", "struct_declaration",
                                                                         "interface_declaration")},
                  funcs={"function_declaration": desc("identifier", depth=1)},
                  imports={"import_declaration": lambda n: desc("module_fqn", depth=3)(n)},
                  calls={"call_expression": call_target}, entry=r"\bvoid\s+main\s*\("),
    "fortran": LangSpec("fortran", classes={"module": desc("name", depth=2), "program": desc("name", depth=2)},
                        funcs={"subroutine": lambda n: _text(n.named_children[0].child_by_field_name("name"))
                               if n.named_children else None,
                               "function": lambda n: _text(n.named_children[0].child_by_field_name("name"))
                               if n.named_children else None},
                        imports={"use_statement": lambda n: desc("module_name", "name", depth=2)(n)},
                        calls={"subroutine_call": lambda n: _norm_call(re.sub(r"^\s*call\s+", "", _text(n), flags=re.I)
                                                                       .split("(")[0].strip()),
                               "call_expression": call_target}),
    "pascal": LangSpec("pascal", classes={"declType": fld("name")},
                       funcs={"defProc": lambda n: _text((n.child_by_field_name("header") or n)
                                                        .child_by_field_name("name"))},
                       imports={"declUses": lambda n: ",".join(_text(c) for c in n.named_children
                                                               if c.type == "moduleName")},
                       calls={"exprCall": lambda n: _norm_call(_text(n).split("(")[0])}),
    "matlab": LangSpec("matlab", classes={"class_definition": fld("name")},
                       funcs={"function_definition": fld("name")},
                       calls={"function_call": lambda n: _norm_call(_text(n.child_by_field_name("name")))}),
    "verilog": LangSpec("verilog", classes={"module_declaration": desc("simple_identifier", depth=2),
                                            "interface_declaration": desc("simple_identifier", depth=2)},
                        funcs={"function_declaration": desc("simple_identifier", depth=3),
                               "task_declaration": desc("simple_identifier", depth=3)},
                        imports={"include_compiler_directive": import_target},
                        calls={"module_instantiation": lambda n: desc("simple_identifier", depth=1)(n)}),
    "vhdl": LangSpec("vhdl", classes={"entity_declaration": fld("name"), "architecture_body": fld("name"),
                                      "package_declaration": fld("name")},
                     funcs={"function_body": fld("designator", "name"), "procedure_body": fld("designator", "name")},
                     imports={"use_clause": import_target},
                     calls={"component_instantiation_statement": lambda n: desc("simple_name", "identifier",
                                                                                depth=3)(n)}),
    "gleam": LangSpec("gleam", classes={"type_definition": desc("type_identifier", depth=2)},
                      funcs={"function": fld("name")}, imports={"import": fld("module")},
                      calls={"function_call": call_target}),
    "odin": LangSpec("odin", classes={"struct_declaration": desc("identifier", depth=1)},
                     funcs={"procedure_declaration": desc("identifier", depth=1)},
                     imports={"import_declaration": import_target}, calls={"call_expression": call_target}),
    "v": LangSpec("v", classes={"struct_declaration": fld("name"), "interface_declaration": fld("name")},
                  funcs={"function_declaration": fld("name")}, imports={"import_declaration": fld("path")},
                  calls={"call_expression": call_target}, entry=r"\bfn\s+main\s*\("),
}


BASE_TYPES = {"superclass", "base_class", "inheritance_specifier", "base_clause", "extends_clause", "interfaces",
              "base_list", "class_heritage", "typeref", "superclasses"}


def extract(path: str, text: str, language: str) -> Facts:
    spec = SPECS[language]
    facts = Facts(path, language)
    mod = module_name(path)
    mod_id = f"module:{mod}"
    is_test = is_test_path(path)
    tree = _core._parser(spec.grammar).parse(text.encode("utf-8"))   # load errors propagate: caller falls back
    lines = text.count("\n") + 1
    entry = bool(spec.entry and re.search(spec.entry, text, re.M))
    facts.node(mod_id, "module", mod, ev.AST, qualname=mod, file=path, line_start=1, line_end=lines,
               text=_leading_comment(text), attrs={"test": is_test, "loc": lines, "parser": f"tree-sitter:{language}",
                                                   "imports": {}, "entrypoint": entry})
    facts.edge(f"file:{path}", mod_id, "defines", ev.AST)
    w = _Walker(facts, spec, language, mod, path, is_test, text.splitlines())
    w.walk(tree.root_node, [])
    if spec.lexical_calls:
        w.lexical_calls()
    if tree.root_node.has_error:
        facts.errors.append("syntax errors present; partial extraction")
    return facts


class _Walker:
    def __init__(self, facts: Facts, spec: LangSpec, lang: str, mod: str, path: str, is_test: bool,
                 lines: list[str]):
        self.f, self.spec, self.lang, self.mod, self.path, self.is_test = facts, spec, lang, mod, path, is_test
        self.mod_id = f"module:{mod}"
        self.lines = lines
        self.funcs: list[tuple[str, int, int]] = []    # (symbol id, first line, last line) for lexical calls
        self.skip: set[int] = set()

    def sid(self, qual: str) -> str:
        return f"symbol:{self.mod}:{qual}"

    def walk(self, node: Any, scope: list[tuple[str, str]]) -> None:
        if node.id in self.skip:
            return
        t = node.type
        if self.spec.forms is not None:
            form = self.spec.forms(node)
            if form and form[1]:
                kind, name = form[0], _clean(form[1]) if form[0] != "import" else form[1]
                if kind == "import":
                    self._import_ref(name, node)
                    return
                if kind == "class":
                    self._class(node, name, scope)
                    return
                self._func(node, name, scope)
                return
        if t in self.spec.imports:
            target = self.spec.imports[t](node)
            if target:
                for part in target.split(","):
                    self._import_ref(part.strip(), node)
                return
        if t in self.spec.classes:
            name = self.spec.classes[t](node)
            if name:
                self._class(node, _clean(name), scope)
                return
        if t in self.spec.funcs:
            name = self.spec.funcs[t](node)
            if name:
                self._func(node, _clean(name), scope)
                return
        if t in self.spec.calls:
            target = self.spec.calls[t](node)
            if target:
                self._call(target, node, scope)
        for c in node.children:
            self.walk(c, scope)

    def _class(self, node: Any, name: str, scope: list[tuple[str, str]]) -> None:
        qual = ".".join([q for q, _ in scope if _ == "class"] + [name]) if scope else name
        sid = self.sid(qual)
        bases = self._bases(node)
        self.f.node(sid, "class", name.rsplit(".", 1)[-1], ev.AST, qualname=f"{self.mod}.{qual}", file=self.path,
                    line_start=node.start_point[0] + 1, line_end=node.end_point[0] + 1, text=_doc_before(node),
                    attrs={"symbol_type": "class", "methods": [], "abstract": node.type in (
                        "interface_declaration", "class", "protocol_declaration"), "bases": bases, "kind": node.type})
        parent = self.sid(scope[-1][0]) if scope and scope[-1][1] == "class" else self.mod_id
        self.f.edge(parent, sid, "contains" if parent != self.mod_id else "defines", ev.AST,
                    line=node.start_point[0] + 1)
        for base in bases:
            self.f.ref(sid, base, "inherits", node.start_point[0] + 1, module=self.mod)
        for c in node.children:
            self.walk(c, scope + [(qual, "class")])

    def _func(self, node: Any, name: str, scope: list[tuple[str, str]]) -> None:
        owner = next((q for q, k in reversed(scope) if k == "class"), None)
        if "." in name:   # Type.method / Module.fn definitions (Lua, Pascal, Go-style receivers)
            prefix, name = name.rsplit(".", 1)
            if prefix not in ("this", "self", "M", "_M") and not owner:
                owner = prefix
        if not name or not re.match(r"[\w$!?<>=+*/-]", name):
            return
        end = node.end_point[0]
        body = node.next_named_sibling
        bodies = [body] if body is not None and body.type in self.spec.body_sibling else []
        if bodies:
            end = body.end_point[0]
            self.skip.add(body.id)
        in_func = scope and scope[-1][1] == "function"
        qual = f"{scope[-1][0]}.{name}" if in_func else (f"{owner}.{name}" if owner else name)
        sid = self.sid(qual)
        kind = "test" if self.is_test and TEST_FUNC.search(name) else ("method" if owner else "function")
        existing = next((n for n in self.f.nodes if n["id"] == sid), None)
        if existing is None:   # multi-clause definitions (Erlang, Haskell, Elixir) share one symbol
            self.f.node(sid, kind, name, ev.AST, qualname=f"{self.mod}.{qual}", file=self.path,
                        line_start=node.start_point[0] + 1, line_end=end + 1, text=_doc_before(node),
                        attrs={"symbol_type": kind, "params": [], "class": owner,
                               "public": not name.startswith("_"), "complexity": 1})
            parent = self.sid(owner) if owner and any(q == owner for q, _ in scope) else self.mod_id
            self.f.edge(parent, sid, "contains" if parent != self.mod_id else "defines", ev.AST,
                        line=node.start_point[0] + 1)
            if owner:
                cls = next((n for n in self.f.nodes if n["id"] == self.sid(owner)), None)
                if cls is not None and name not in cls["attrs"]["methods"]:
                    cls["attrs"]["methods"].append(name)
        else:
            existing["line_end"] = max(existing.get("line_end") or 0, end + 1)
        self.funcs.append((sid, node.start_point[0] + 1, end + 1))
        for c in node.children + bodies:
            self.walk(c, scope + [(qual, "function")])

    def _bases(self, node: Any) -> list[str]:
        out: list[str] = []
        for c in _descendants(node, 2):
            if c.type in BASE_TYPES:
                for m in re.finditer(r"[A-Z][\w.]*", _text(c)):
                    if m.group(0) not in out:
                        out.append(m.group(0))
        return out[:8]

    def _import_ref(self, target: str, node: Any) -> None:
        target = target.strip().strip("'\"`").split()[0] if target.strip() else ""
        target = re.sub(r"\.(h|hpp|dart|sol|lua|rb|ex|jl|pl|pm|sh|ps1|psm1|nim|zig|v|odin|gleam)$", "", target)
        target = target.replace("::", ".").replace("\\", ".").strip("{}*;:[]() ")
        if target and len(target) < 200:
            self.f.ref(self.mod_id, target, "imports", node.start_point[0] + 1, module=self.mod)

    def _call(self, target: str, node: Any, scope: list[tuple[str, str]], evidence: str | None = None) -> None:
        keep = self.lang in HYPHEN_NAMES and re.fullmatch(r"[A-Za-z_][\w.*+!?<>-]*", target or "")
        target = target if keep else _norm_call(target)
        if not target or target in NOT_CALLS:
            return
        in_func = bool(scope) and scope[-1][1] == "function"
        if self.lang in HYPHEN_NAMES - {"powershell", "bash"} and not in_func:
            return   # Lisp forms at the top level are declarations and data, not calls
        if in_func and node is not None and target.rsplit(".", 1)[-1] == scope[-1][0].rsplit(".", 1)[-1] and \
                any(sid == self.sid(scope[-1][0]) and start == node.start_point[0] + 1
                    for sid, start, _ in self.funcs):
            return   # the definition's own head (Elixir `def save(x)`, Julia signatures)
        owner = self.sid(scope[-1][0]) if in_func else self.mod_id
        cls = next((q for q, k in reversed(scope) if k == "class"), None)
        line = node.start_point[0] + 1 if node is not None else None
        res = classify_call(target)
        if res:
            resource, verb = res
            self.f.node(f"resource:{resource}", "resource", resource, ev.HEURISTIC, attrs={"category": resource})
            self.f.edge(owner, f"resource:{resource}", verb, ev.HEURISTIC, line=line, confidence=0.65,
                        attrs={"call": target, "classifier": "heuristic"})
        ctx: dict[str, Any] = {"module": self.mod, "cls": cls}
        if evidence:
            ctx["evidence"] = evidence
        self.f.ref(owner, target, "calls", line, **ctx)

    def lexical_calls(self) -> None:
        """Calls for grammars without a call node: read from each function's own lines (innermost wins)."""
        claimed: set[int] = set()
        for sid, start, end in sorted(self.funcs, key=lambda f: f[2] - f[1]):
            for ln in range(start, min(end, len(self.lines)) + 1):
                if ln in claimed:
                    continue
                claimed.add(ln)
                code = self.lines[ln - 1].split("//")[0]
                own = sid.rsplit(":", 1)[-1].rsplit(".", 1)[-1]
                for m in CALL_RE.finditer(code):
                    target = _norm_call(m.group(1).replace("::", ".").replace(":", "."))
                    if not target or target.split(".")[-1] in NOT_CALLS or (ln == start and target == own):
                        continue
                    if True:
                        self.f.ref(sid, target, "calls", ln, module=self.mod, evidence=ev.LEXICAL)


def _clean(name: str) -> str:
    name = name.strip().strip("'\"`:")
    name = re.sub(r"\s+", "", name.split("(")[0].split("\n")[0])
    return name.replace("::", ".")
