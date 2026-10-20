"""Python extraction using the standard-library AST (parse only, never import)."""

from __future__ import annotations

import ast
import io
import re
import tokenize
from pathlib import PurePosixPath

from promapkin import evidence as ev
from promapkin.extract import Facts, is_test_path
from promapkin.extract.resources import AUTH_HINT, classify_call

EVENT_EMIT = {"emit", "publish", "trigger", "dispatch_event", "fire"}
EVENT_ON = {"on", "once", "add_listener", "add_event_listener", "subscribe", "add_handler"}
REGISTRATION_CALLS = {"register", "add_done_callback", "submit", "add_job", "callLater", "call_soon", "call_later",
                      "add_task", "register_callback", "add_command", "add_url_rule", "connect_handler"}
CALLBACK_KWARGS = {"target", "callback", "handler", "func", "view_func", "on_success", "on_error", "on_failure",
                   "listener", "hook"}
DOTTED_PATH = re.compile(r"^[a-z_][a-z0-9_]*(\.[a-z_][a-z0-9_]*)+\.(?:[A-Z][A-Za-z0-9]*|[a-z_][a-z0-9_]*)$")
ROUTE_METHODS = {"get", "post", "put", "delete", "patch", "head", "options", "route", "api_route", "websocket"}
NOTE_RE = re.compile(
    r"\b(we (chose|decided|use|prefer|avoid)|because|decision|trade-?off|rationale|deliberately|intentionally|"
    r"workaround|hack|todo|fixme|xxx|note:|warning:|security)\b",
    re.IGNORECASE,
)


def module_name(path: str) -> str:
    p = PurePosixPath(path)
    parts = list(p.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    if parts and parts[0] in ("src", "lib") and len(parts) > 1:
        parts = parts[1:]
    return ".".join(parts) or p.stem


def dotted(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = dotted(node.value)
        return f"{base}.{node.attr}" if base else None
    if isinstance(node, ast.Call):
        base = dotted(node.func)
        return f"{base}()" if base else None
    if isinstance(node, ast.Subscript):
        return dotted(node.value)
    return None


def _annotation_names(node: ast.AST | None) -> list[str]:
    if node is None:
        return []
    names = []
    for sub in ast.walk(node):
        if isinstance(sub, (ast.Name, ast.Attribute)):
            d = dotted(sub)
            if d and d[0].isupper() or (d and "." in d and d.rsplit(".", 1)[-1][:1].isupper()):
                names.append(d)
        elif isinstance(sub, ast.Constant) and isinstance(sub.value, str) and sub.value[:1].isupper():
            names.append(sub.value.strip("'\""))
    return names


def _complexity(fn: ast.AST) -> int:
    return 1 + sum(isinstance(n, (ast.If, ast.For, ast.While, ast.Try, ast.With, ast.BoolOp, ast.IfExp,
                                  ast.ExceptHandler, ast.comprehension, ast.Match if hasattr(ast, "Match") else ast.If))
                   for n in ast.walk(fn))


def extract(path: str, text: str) -> Facts:
    facts = Facts(path, "python")
    mod = module_name(path)
    is_test = is_test_path(path)
    file_id = f"file:{path}"
    mod_id = f"module:{mod}"
    try:
        tree = ast.parse(text, filename=path)
    except (SyntaxError, ValueError) as exc:
        facts.errors.append(f"parse error: {exc}")
        facts.node(mod_id, "module", mod, ev.AST, file=path, attrs={"parse_error": str(exc)})
        facts.edge(file_id, mod_id, "defines", ev.AST)
        return facts

    imports: dict[str, str] = {}   # local alias -> fully qualified dotted target
    import_records: list[dict] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                imports[a.asname or a.name.split(".")[0]] = a.name if a.asname else a.name.split(".")[0]
                import_records.append({"module": a.name, "line": node.lineno})
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                pkg = mod.split(".")
                is_pkg = PurePosixPath(path).name == "__init__.py"
                keep = len(pkg) - node.level + (1 if is_pkg else 0)
                base = ".".join(pkg[:max(keep, 0)] + ([base] if base else []))
            for a in node.names:
                if a.name == "*":
                    import_records.append({"module": base, "line": node.lineno})
                    continue
                imports[a.asname or a.name] = f"{base}.{a.name}" if base else a.name
                import_records.append({"module": base, "name": a.name, "line": node.lineno})

    entrypoint = any(
        isinstance(n, ast.If) and isinstance(n.test, ast.Compare) and isinstance(n.test.left, ast.Name)
        and n.test.left.id == "__name__" for n in tree.body
    )
    all_exports = None
    for n in tree.body:
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__all__" for t in n.targets):
            if isinstance(n.value, (ast.List, ast.Tuple)):
                all_exports = [e.value for e in n.value.elts if isinstance(e, ast.Constant)]

    lines = text.count("\n") + 1
    facts.node(
        mod_id, "module", mod, ev.AST, qualname=mod, file=path, line_start=1, line_end=lines,
        text=ast.get_docstring(tree),
        attrs={"imports": imports, "entrypoint": entrypoint, "test": is_test, "exports": all_exports,
               "loc": lines},
    )
    facts.edge(file_id, mod_id, "defines", ev.AST)
    for rec in import_records:
        facts.ref(mod_id, rec["module"], "imports", rec["line"], module=mod, name=rec.get("name"))

    _Visitor(facts, mod, path, imports, is_test).visit(tree)
    _notes(facts, mod_id, path, text)
    return facts


class _Visitor(ast.NodeVisitor):
    def __init__(self, facts: Facts, mod: str, path: str, imports: dict[str, str], is_test: bool):
        self.f = facts
        self.mod = mod
        self.path = path
        self.imports = imports
        self.is_test = is_test
        self.stack: list[tuple[str, str]] = []   # (qualname, kind)
        self.cls: str | None = None
        # Lightweight type inference: instance attributes per class and local variables per function.
        # Values are "TYPE:<Name>" (a class name) or "RET:<callable>" (the return type of a call).
        self.attr_types: dict[str, dict[str, str]] = {}
        self.local_types: list[dict[str, str]] = []

    # ---------------------------------------------------------------- helpers
    def _id(self, qual: str) -> str:
        return f"symbol:{self.mod}:{qual}"

    def _owner(self) -> str:
        return self._id(self.stack[-1][0]) if self.stack else f"module:{self.mod}"

    def _ctx(self) -> dict:
        return {"module": self.mod, "cls": self.cls}

    def _root_module(self, expr: str) -> str | None:
        head = expr.split(".", 1)[0]
        return self.imports.get(head)

    # ---------------------------------------------------------------- definitions
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        qual = ".".join([q for q, _ in self.stack] + [node.name])
        sid = self._id(qual)
        bases = [d for d in (dotted(b) for b in node.bases) if d]
        decorators = [d for d in (dotted(x) for x in node.decorator_list) if d]
        abstract = any(b in ("ABC", "abc.ABC", "Protocol", "typing.Protocol") for b in bases) or any(
            isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            and any((dotted(d) or "").endswith("abstractmethod") for d in n.decorator_list)
            for n in node.body
        )
        methods = [n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        class_attrs = [t.id for n in node.body if isinstance(n, (ast.Assign, ast.AnnAssign))
                       for t in (n.targets if isinstance(n, ast.Assign) else [n.target]) if isinstance(t, ast.Name)]
        kind = "test" if self.is_test and node.name.startswith("Test") else "class"
        self.f.node(
            sid, kind, node.name, ev.AST, qualname=f"{self.mod}.{qual}", file=self.path, line_start=node.lineno,
            line_end=getattr(node, "end_lineno", node.lineno), text=ast.get_docstring(node),
            attrs={"bases": bases, "abstract": abstract, "methods": methods, "decorators": decorators,
                   "class_attrs": class_attrs, "symbol_type": "class"},
        )
        self.f.edge(self._owner(), sid, "contains" if self.stack else "defines", ev.AST, line=node.lineno)
        for b in bases:
            if b not in ("object", "ABC", "abc.ABC", "Protocol", "typing.Protocol", "Generic"):
                self.f.ref(sid, b, "inherits", node.lineno, **self._ctx())
        self.attr_types[qual] = _instance_attr_types(node)
        prev_cls = self.cls
        self.cls = qual
        self.stack.append((qual, "class"))
        self.generic_visit(node)
        self.stack.pop()
        self.cls = prev_cls

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        qual = ".".join([q for q, _ in self.stack] + [node.name])
        sid = self._id(qual)
        in_class = bool(self.stack) and self.stack[-1][1] == "class"
        decorators = [d for d in (dotted(x) for x in node.decorator_list) if d]
        args = node.args
        params = []
        all_args = args.posonlyargs + args.args + args.kwonlyargs
        for a in all_args:
            if a.arg in ("self", "cls"):
                continue
            params.append({"name": a.arg, "type": ast.unparse(a.annotation) if a.annotation else None})
        returns = ast.unparse(node.returns) if node.returns else None
        is_test_fn = self.is_test and node.name.startswith("test")
        kind = "test" if is_test_fn else ("method" if in_class else "function")
        defaults_src = " ".join(ast.unparse(d) for d in args.defaults + [d for d in args.kw_defaults if d])
        auth = [d for d in decorators if AUTH_HINT.search(d)] + (
            ["dependency:" + m.group(0)] if (m := AUTH_HINT.search(defaults_src)) else [])
        self.f.node(
            sid, kind, node.name, ev.AST, qualname=f"{self.mod}.{qual}", file=self.path, line_start=node.lineno,
            line_end=getattr(node, "end_lineno", node.lineno), text=ast.get_docstring(node),
            attrs={"params": params, "returns": returns, "decorators": decorators, "async":
                   isinstance(node, ast.AsyncFunctionDef), "complexity": _complexity(node), "auth": auth,
                   "symbol_type": kind, "class": self.stack[-1][0] if in_class else None,
                   "public": not node.name.startswith("_"),
                   "cli_command": any(re.search(r"(^|\.)(command|group)(\(\))?$", d) for d in decorators)},
        )
        self.f.edge(self._owner(), sid, "contains" if self.stack else "defines", ev.AST, line=node.lineno)

        # Framework wiring the call graph cannot see: pytest fixtures and FastAPI dependencies.
        if is_test_fn:
            for a in all_args:
                if a.arg not in ("self", "cls"):
                    self.f.ref(sid, a.arg, "invokes", node.lineno, fixture=True, wiring="pytest fixture",
                               **self._ctx())
        for d in args.defaults + [d for d in args.kw_defaults if d]:
            if isinstance(d, ast.Call) and (dotted(d.func) or "").endswith("Depends") and d.args:
                dep = dotted(d.args[0])
                if dep:
                    self.f.ref(sid, dep, "invokes", node.lineno, wiring="dependency injection (Depends)",
                               **self._ctx())

        # Type references in the signature (also the basis for dependency-injection detection).
        for a in all_args:
            for t in _annotation_names(a.annotation):
                self.f.ref(sid, t, "references", node.lineno, injected=node.name == "__init__", **self._ctx())
        for t in _annotation_names(node.returns):
            self.f.ref(sid, t, "references", node.lineno, **self._ctx())

        for dec in node.decorator_list:
            self._route(dec, sid, auth)

        local: dict[str, str] = {}
        for a in all_args:
            names = _annotation_names(a.annotation)
            if names:
                local[a.arg] = "TYPE:" + names[0]
        self.local_types.append(local)
        self.stack.append((qual, "function"))
        self.generic_visit(node)
        self.stack.pop()
        self.local_types.pop()

    def visit_Assign(self, node: ast.Assign) -> None:
        if self.local_types and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            inferred = _infer(node.value)
            if inferred:
                self.local_types[-1][node.targets[0].id] = inferred
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if self.local_types and isinstance(node.target, ast.Name):
            names = _annotation_names(node.annotation)
            if names:
                self.local_types[-1][node.target.id] = "TYPE:" + names[0]
        self.generic_visit(node)

    def _typed_target(self, expr: str) -> str | None:
        """Rewrite 'var.method' / 'self.attr.method' using inferred receiver types."""
        parts = expr.split(".")
        if len(parts) >= 3 and parts[0] == "self" and self.cls:
            t = self.attr_types.get(self.cls, {}).get(parts[1])
            if t:
                return f"{t}.{'.'.join(parts[2:])}"
        if len(parts) >= 2 and self.local_types:
            t = self.local_types[-1].get(parts[0])
            if t:
                return f"{t}.{'.'.join(parts[1:])}"
        return None

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def _route(self, dec: ast.AST, handler: str, auth: list[str]) -> None:
        if not isinstance(dec, ast.Call) or not isinstance(dec.func, ast.Attribute):
            return
        verb = dec.func.attr
        if verb not in ROUTE_METHODS or not dec.args:
            return
        first = dec.args[0]
        if not (isinstance(first, ast.Constant) and isinstance(first.value, str) and first.value.startswith("/")):
            return
        methods = [verb.upper()] if verb not in ("route", "api_route") else ["GET"]
        for kw in dec.keywords:
            if kw.arg == "methods" and isinstance(kw.value, (ast.List, ast.Tuple)):
                methods = [e.value.upper() for e in kw.value.elts if isinstance(e, ast.Constant)]
        router = dotted(dec.func.value) or "app"
        for m in methods:
            self._endpoint(m, first.value, handler, dec.lineno, router, auth)

    def _endpoint(self, method: str, route: str, handler: str, line: int, router: str, auth: list[str]) -> None:
        api_id = f"api:{method} {route}"
        self.f.node(api_id, "endpoint", f"{method} {route}", ev.AST, file=self.path, line_start=line,
                    line_end=line, owner_file=self.path,
                    attrs={"method": method, "route": route, "router": router, "handler": handler, "auth": auth})
        self.f.edge(api_id, handler, "routes_to", ev.AST, line=line)
        self.f.edge(f"module:{self.mod}", api_id, "exposes", ev.AST, line=line)

    # ---------------------------------------------------------------- behaviour
    def visit_Call(self, node: ast.Call) -> None:
        expr = dotted(node.func)
        src = self._owner()
        if expr:
            self._dynamic_wiring(node, expr, src)
            # Django-style routing: path("route/", view)
            if expr in ("path", "re_path", "url") and len(node.args) >= 2 and isinstance(node.args[0], ast.Constant) \
                    and isinstance(node.args[0].value, str):
                view = dotted(node.args[1])
                if view:
                    route = "/" + node.args[0].value.lstrip("^/")
                    api_id = f"api:ANY {route}"
                    self.f.node(api_id, "endpoint", f"ANY {route}", ev.AST, file=self.path, line_start=node.lineno,
                                line_end=node.lineno, attrs={"method": "ANY", "route": route, "router": "django"})
                    self.f.ref(api_id, view.removesuffix(".as_view()"), "routes_to", node.lineno, **self._ctx())
            if expr.endswith("add_url_rule") and node.args and isinstance(node.args[0], ast.Constant):
                for kw in node.keywords:
                    if kw.arg == "view_func" and (view := dotted(kw.value)):
                        api_id = f"api:ANY {node.args[0].value}"
                        self.f.node(api_id, "endpoint", f"ANY {node.args[0].value}", ev.AST, file=self.path,
                                    line_start=node.lineno, attrs={"method": "ANY", "route": node.args[0].value})
                        self.f.ref(api_id, view, "routes_to", node.lineno, **self._ctx())

            if expr in ("os.getenv", "os.environ.get", "getenv", "environ.get") and node.args and \
                    isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                self._config(src, node.args[0].value, node.lineno)
            else:
                root_mod = self._root_module(expr)
                res = classify_call(expr.replace("()", ""), root_mod)
                if res:
                    resource, verb = res
                    self.f.node(f"resource:{resource}", "resource", resource, ev.HEURISTIC,
                                attrs={"category": resource})
                    self.f.edge(src, f"resource:{resource}", verb, ev.HEURISTIC, line=node.lineno,
                                confidence=0.7, attrs={"call": expr, "classifier": "heuristic"})
                elif "." in expr and root_mod and not self._is_local(root_mod):
                    # Unclassified call into an external library: kept for the ML classification tier.
                    self.f.ref(src, expr, "external_call", node.lineno, library=root_mod, **self._ctx())
                typed = self._typed_target(expr.replace("()", ""))
                self.f.ref(src, typed or expr, "calls", node.lineno, **self._ctx())
        self.generic_visit(node)

    def _dynamic_wiring(self, node: ast.Call, expr: str, src: str) -> None:
        """Reflection, dynamic imports, callbacks and events: links a plain call graph misses."""
        last = expr.rsplit(".", 1)[-1]
        args = node.args
        const = [a.value for a in args if isinstance(a, ast.Constant) and isinstance(a.value, str)]
        if expr in ("getattr", "builtins.getattr") and len(args) >= 2 and isinstance(args[1], ast.Constant) \
                and isinstance(args[1].value, str):
            obj = dotted(args[0])
            if obj:
                self.f.ref(src, f"{obj}.{args[1].value}", "calls", node.lineno, wiring="reflection (getattr)",
                           **self._ctx())
        elif expr in ("importlib.import_module", "import_module", "__import__") and const:
            self.f.ref(f"module:{self.mod}", const[0], "imports", node.lineno, module=self.mod,
                       wiring="dynamic import")
        elif last in EVENT_EMIT and const:
            eid = f"event:{const[0]}"
            self.f.node(eid, "event", const[0], ev.AST, attrs={"source": "event wiring"})
            self.f.edge(src, eid, "produces", ev.AST, line=node.lineno)
        elif last in EVENT_ON and const and len(args) >= 2:
            eid = f"event:{const[0]}"
            self.f.node(eid, "event", const[0], ev.AST, attrs={"source": "event wiring"})
            self.f.edge(src, eid, "consumes", ev.AST, line=node.lineno)
            handler = dotted(args[1])
            if handler:
                self.f.ref(eid, handler, "invokes", node.lineno, wiring="event handler", **self._ctx())
        elif last == "connect" and args and "." in expr:   # signal.connect(handler) (Django, blinker, Qt)
            handler = dotted(args[0])
            eid = f"event:{expr.rsplit('.', 1)[0]}"
            if handler:
                self.f.node(eid, "event", expr.rsplit(".", 1)[0], ev.AST, attrs={"source": "signal"})
                self.f.edge(src, eid, "consumes", ev.AST, line=node.lineno)
                self.f.ref(eid, handler, "invokes", node.lineno, wiring="signal receiver", **self._ctx())
        callbacks = [kw.value for kw in node.keywords if kw.arg in CALLBACK_KWARGS]
        if last in REGISTRATION_CALLS:
            callbacks += [a for a in args if isinstance(a, (ast.Name, ast.Attribute))]
        for cb in callbacks:
            target = dotted(cb)
            if target and target not in ("self", "None"):
                self.f.ref(src, target, "invokes", node.lineno, wiring=f"callback ({last})", **self._ctx())

    def visit_Constant(self, node: ast.Constant) -> None:
        # Dotted strings naming code ("app.services.Mailer") are how DI containers and settings wire classes.
        if isinstance(node.value, str) and DOTTED_PATH.match(node.value):
            self.f.ref(self._owner(), node.value, "references", node.lineno, wiring="string reference",
                       exact=True, **self._ctx())

    def _is_local(self, qualified: str) -> bool:
        return qualified.split(".")[0] == self.mod.split(".")[0]

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if dotted(node.value) in ("os.environ", "environ") and isinstance(node.slice, ast.Constant) \
                and isinstance(node.slice.value, str):
            self._config(self._owner(), node.slice.value, node.lineno)
        self.generic_visit(node)

    def _config(self, src: str, key: str, line: int) -> None:
        cid = f"config:{key}"
        self.f.node(cid, "config", key, ev.AST, owner_file=None, attrs={"source": "environment"})
        self.f.edge(src, cid, "reads", ev.AST, line=line)

    def visit_Raise(self, node: ast.Raise) -> None:
        if node.exc is not None:
            exc = dotted(node.exc.func if isinstance(node.exc, ast.Call) else node.exc)
            if exc and exc[:1].isupper():
                self.f.ref(self._owner(), exc, "raises", node.lineno, **self._ctx())
        self.generic_visit(node)


def _infer(value: ast.AST) -> str | None:
    if isinstance(value, ast.Call):
        callee = dotted(value.func)
        if not callee:
            return None
        last = callee.rsplit(".", 1)[-1]
        return f"TYPE:{callee}" if last[:1].isupper() else f"RET:{callee}"
    return None


def _instance_attr_types(cls: ast.ClassDef) -> dict[str, str]:
    types: dict[str, str] = {}
    for fn in cls.body:
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        params = {a.arg: _annotation_names(a.annotation) for a in fn.args.args + fn.args.kwonlyargs}
        for n in ast.walk(fn):
            if isinstance(n, (ast.Assign, ast.AnnAssign)):
                targets = n.targets if isinstance(n, ast.Assign) else [n.target]
                for t in targets:
                    if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == "self":
                        if isinstance(n, ast.AnnAssign) and (names := _annotation_names(n.annotation)):
                            types[t.attr] = "TYPE:" + names[0]
                        elif isinstance(n.value, ast.Name) and params.get(n.value.id):
                            types[t.attr] = "TYPE:" + params[n.value.id][0]
                        elif n.value is not None and (inferred := _infer(n.value)):
                            types.setdefault(t.attr, inferred)
    return types


def _notes(facts: Facts, mod_id: str, path: str, text: str) -> None:
    """Decision-bearing and maintenance comments (the AST drops comments, so tokenize them)."""
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return
    block: list[tuple[int, str]] = []

    def flush() -> None:
        if not block:
            return
        body = " ".join(c for _, c in block)
        if NOTE_RE.search(body):
            line = block[0][0]
            low = body.lower()
            category = ("todo" if re.search(r"\b(todo|fixme|xxx)\b", low) else
                        "warning" if re.search(r"\b(hack|workaround|warning|security)\b", low) else "rationale")
            nid = f"note:{path}:{line}"
            facts.node(nid, "code_note", body[:80], ev.DOCUMENTATION, file=path, line_start=line,
                       line_end=block[-1][0], text=body, attrs={"category": category, "classifier": "heuristic"})
            facts.edge(mod_id, nid, "documents", ev.DOCUMENTATION, line=line)
        block.clear()

    last_line = -2
    for tok in tokens:
        if tok.type == tokenize.COMMENT:
            line = tok.start[0]
            if line != last_line + 1:
                flush()
            block.append((line, tok.string.lstrip("#").strip()))
            last_line = line
    flush()
