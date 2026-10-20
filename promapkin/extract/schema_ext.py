"""Data and API schemas as graph nodes: SQL DDL (tables, columns, keys, views, functions), Prisma models,
GraphQL SDL and Protocol Buffers.

- Tables are `table:<name>` (lowercase, default schema dropped), so code that queries a table, ORM classes
  that map to it and the DDL that defines it meet on one node. Foreign keys become `depends_on` edges
  between tables with the columns as attributes.
- GraphQL `Query`/`Mutation`/`Subscription` fields and protobuf `rpc`s are endpoints (protocol in attrs);
  messages and types are `schema_type` nodes, linked by the types their fields use.
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Any

from promapkin import evidence as ev
from promapkin.extract import Facts

SCHEMA_LANGUAGES = {"sql", "prisma", "graphql", "proto"}
DEFAULT_SCHEMAS = ("public.", "dbo.", "main.")
SQL_NOT_TABLES = {"select", "where", "set", "values", "dual", "lateral", "unnest", "only", "as", "on", "using",
                  "the", "a", "an", "if", "exists", "not", "table", "into", "from", "join", "to", "with", "by"}


def table_id(name: str) -> str:
    return f"table:{norm_table(name)}"


def norm_table(name: str) -> str:
    n = re.sub(r'["`\[\]]', "", name.strip()).lower()
    for prefix in DEFAULT_SCHEMAS:
        if n.startswith(prefix):
            n = n[len(prefix):]
    return n


def extract(path: str, text: str, language: str) -> Facts:
    facts = Facts(path, language)
    mid = f"manifest:{path}"
    facts.node(mid, "manifest", PurePosixPath(path).name, ev.MANIFEST, file=path, line_start=1,
               line_end=text.count("\n") + 1, attrs={"format": language})
    facts.edge(f"file:{path}", mid, "defines", ev.MANIFEST)
    try:
        {"sql": _sql, "prisma": _prisma, "graphql": _graphql, "proto": _proto}[language](facts, path, text, mid)
    except (ValueError, KeyError, IndexError, AttributeError) as exc:
        facts.errors.append(f"{language} extraction incomplete: {type(exc).__name__}: {exc}")
    return facts


# ------------------------------------------------------------------ SQL
def sql_statements(text: str) -> list[tuple[str, int]]:
    """Statements with their first line; comments removed, dollar-quoted bodies kept intact."""
    out, buf, line, start_line = [], [], 1, 1
    i, n, quote, dollar, started = 0, len(text), "", "", False
    while i < n:
        c = text[i]
        if dollar:
            if text.startswith(dollar, i):
                buf.append(dollar)
                i += len(dollar)
                dollar = ""
                continue
        elif quote:
            if c == quote:
                quote = ""
        elif text.startswith("--", i):
            j = text.find("\n", i)
            i = n if j < 0 else j
            continue
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            seg = text[i:n if j < 0 else j + 2]
            line += seg.count("\n")
            i = n if j < 0 else j + 2
            continue
        elif c in "'\"`":
            quote = c
        elif c == "$":
            m = re.match(r"\$[A-Za-z_]*\$", text[i:])
            if m:
                if not started:
                    started, start_line = True, line
                dollar = m.group(0)
                buf.append(dollar)
                i += len(dollar)
                continue
        elif c == ";":
            stmt = "".join(buf).strip()
            if stmt:
                out.append((stmt, start_line))
            buf, started = [], False
            i += 1
            continue
        if not started and not c.isspace():
            started, start_line = True, line
        if c == "\n":
            line += 1
        buf.append(c)
        i += 1
    stmt = "".join(buf).strip()
    if stmt:
        out.append((stmt, start_line))
    return out


IDENT = r'(?:"[^"]+"|`[^`]+`|\[[^\]]+\]|[A-Za-z_][\w$]*)(?:\.(?:"[^"]+"|`[^`]+`|\[[^\]]+\]|[A-Za-z_][\w$]*))?'
TABLES_READ = re.compile(rf"\b(?:FROM|JOIN)\s+({IDENT})", re.I)
TABLES_WRITTEN = re.compile(rf"\b(?:INSERT\s+(?:OR\s+\w+\s+)?INTO|UPDATE|DELETE\s+FROM|MERGE\s+INTO|"
                            rf"TRUNCATE(?:\s+TABLE)?|COPY)\s+({IDENT})", re.I)


def tables_in(sql: str) -> tuple[set[str], set[str]]:
    """(read, written) table names in one SQL statement or snippet."""
    reads = {norm_table(m.group(1)) for m in TABLES_READ.finditer(sql)}
    writes = {norm_table(m.group(1)) for m in TABLES_WRITTEN.finditer(sql)}
    ctes = {m.group(1).lower() for m in re.finditer(r"\b(?:WITH|,)\s*([A-Za-z_]\w*)\s+AS\s*\(", sql, re.I)}
    clean = lambda s: {t for t in s if t and t not in SQL_NOT_TABLES and t not in ctes and  # noqa: E731
                       not t.startswith("(")}
    return clean(reads) - clean(writes), clean(writes)


def _sql(facts: Facts, path: str, text: str, mid: str) -> None:
    for stmt, line in sql_statements(text):
        head = " ".join(stmt.split()[:6]).upper()
        m = re.match(rf"CREATE\s+(?:OR\s+REPLACE\s+)?(?:GLOBAL\s+|LOCAL\s+)?(?:TEMP(?:ORARY)?\s+|UNLOGGED\s+)?"
                     rf"TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?({IDENT})\s*\((.*)\)", stmt, re.I | re.S)
        if m:
            _create_table(facts, path, mid, m.group(1), m.group(2), line, stmt)
            continue
        m = re.match(rf"CREATE\s+(?:OR\s+REPLACE\s+)?(?:MATERIALIZED\s+)?VIEW\s+(?:IF\s+NOT\s+EXISTS\s+)?({IDENT})"
                     r"[\s\S]*?\bAS\b([\s\S]*)", stmt, re.I)
        if m:
            vid = table_id(m.group(1))
            facts.node(vid, "table", norm_table(m.group(1)), ev.MANIFEST, file=path, line_start=line,
                       line_end=line + stmt.count("\n"), attrs={"view": True, "columns": [], "format": "sql"})
            facts.edge(mid, vid, "defines", ev.MANIFEST, line=line)
            reads, _ = tables_in(m.group(2))
            for t in sorted(reads):
                facts.ref(vid, f"TABLE:{t}", "link", line, edge="depends_on", evidence=ev.MANIFEST)
            continue
        m = re.match(rf"ALTER\s+TABLE\s+(?:ONLY\s+)?(?:IF\s+EXISTS\s+)?({IDENT})\s+(.*)", stmt, re.I | re.S)
        if m:
            target = norm_table(m.group(1))
            facts.ref(mid, f"TABLE:{target}", "link", line, edge="alters", evidence=ev.MANIFEST,
                      attrs={"change": " ".join(m.group(2).split())[:200]})
            for fk in re.finditer(rf"REFERENCES\s+({IDENT})", m.group(2), re.I):
                facts.ref(table_id(target), f"TABLE:{norm_table(fk.group(1))}", "link", line,
                          edge="depends_on", evidence=ev.MANIFEST, attrs={"foreign_key": True, "via": "alter table"})
            continue
        m = re.match(rf"CREATE\s+(?:UNIQUE\s+)?INDEX\s+(?:CONCURRENTLY\s+)?(?:IF\s+NOT\s+EXISTS\s+)?({IDENT})?\s*"
                     rf"ON\s+(?:ONLY\s+)?({IDENT})\s*(?:USING\s+\w+\s*)?\(([^)]*)\)", stmt, re.I)
        if m:
            facts.ref(mid, f"TABLE:{norm_table(m.group(2))}", "link", line, edge="indexes", evidence=ev.MANIFEST,
                      attrs={"index": (m.group(1) or "").strip('"'), "columns": m.group(3).strip()[:120]})
            continue
        m = re.match(rf"CREATE\s+(?:OR\s+REPLACE\s+)?(FUNCTION|PROCEDURE|TRIGGER)\s+({IDENT})", stmt, re.I)
        if m:
            fid = f"sqlfn:{m.group(2).lower()}"
            facts.node(fid, "function", m.group(2), ev.MANIFEST, file=path, line_start=line,
                       line_end=line + stmt.count("\n"), attrs={"sql": m.group(1).lower(), "symbol_type": "function"})
            facts.edge(mid, fid, "defines", ev.MANIFEST, line=line)
            reads, writes = tables_in(stmt)
            on = re.search(rf"\bON\s+({IDENT})", stmt, re.I) if m.group(1).upper() == "TRIGGER" else None
            if on:
                facts.ref(fid, f"TABLE:{norm_table(on.group(1))}", "link", line, edge="triggers_on",
                          evidence=ev.MANIFEST)
            for t in sorted(reads):
                facts.ref(fid, f"TABLE:{t}", "link", line, edge="reads", evidence=ev.MANIFEST)
            for t in sorted(writes):
                facts.ref(fid, f"TABLE:{t}", "link", line, edge="writes", evidence=ev.MANIFEST)
            continue
        m = re.match(rf"CREATE\s+TYPE\s+({IDENT})\s+AS\s+ENUM\s*\(([^)]*)\)", stmt, re.I)
        if m:
            tid = f"sqltype:{m.group(1).lower()}"
            facts.node(tid, "schema_type", m.group(1), ev.MANIFEST, file=path, line_start=line,
                       attrs={"enum": re.findall(r"'([^']*)'", m.group(2)), "format": "sql"})
            facts.edge(mid, tid, "defines", ev.MANIFEST, line=line)
            continue
        if re.match(r"(INSERT|UPDATE|DELETE|MERGE|COPY|SELECT|WITH)\b", head):
            reads, writes = tables_in(stmt)
            for t in sorted(writes):
                facts.ref(mid, f"TABLE:{t}", "link", line, edge="writes", evidence=ev.MANIFEST)
            for t in sorted(reads):
                facts.ref(mid, f"TABLE:{t}", "link", line, edge="reads", evidence=ev.MANIFEST)


def _create_table(facts: Facts, path: str, mid: str, raw_name: str, body: str, line: int, stmt: str) -> None:
    name = norm_table(raw_name)
    tid = table_id(raw_name)
    columns, pk, fks = [], [], []
    for part in _split_top(body):
        p = part.strip()
        up = p.upper()
        fk = re.match(rf"(?:CONSTRAINT\s+\S+\s+)?FOREIGN\s+KEY\s*\(([^)]*)\)\s*REFERENCES\s+({IDENT})\s*"
                      r"(?:\(([^)]*)\))?", p, re.I)
        if fk:
            fks.append((fk.group(1).replace('"', "").strip(), norm_table(fk.group(2)), (fk.group(3) or "").strip()))
            continue
        pkm = re.match(r"(?:CONSTRAINT\s+\S+\s+)?PRIMARY\s+KEY\s*\(([^)]*)\)", p, re.I)
        if pkm:
            pk += [c.strip().strip('"`') for c in pkm.group(1).split(",")]
            continue
        if re.match(r"(CONSTRAINT|UNIQUE|CHECK|INDEX|KEY|EXCLUDE)\b", up):
            continue
        cm = re.match(r'("[^"]+"|`[^`]+`|\[[^\]]+\]|[A-Za-z_][\w$]*)\s+([A-Za-z][\w ]*?(?:\([^)]*\))?(?:\[\])?)'
                      r'(?=\s|$|,)(.*)', p, re.S)
        if not cm:
            continue
        col = cm.group(1).strip('"`[]')
        rest = cm.group(3)
        columns.append({"name": col, "type": cm.group(2).strip().lower(),
                        "nullable": "NOT NULL" not in rest.upper() and "PRIMARY KEY" not in rest.upper()})
        if "PRIMARY KEY" in rest.upper():
            pk.append(col)
        ref = re.search(rf"REFERENCES\s+({IDENT})\s*(?:\(([^)]*)\))?", rest, re.I)
        if ref:
            fks.append((col, norm_table(ref.group(1)), (ref.group(2) or "").strip()))
    facts.node(tid, "table", name, ev.MANIFEST, file=path, line_start=line, line_end=line + stmt.count("\n"),
               attrs={"columns": columns[:200], "primary_key": pk, "foreign_keys": [
                   {"columns": c, "table": t, "references": r} for c, t, r in fks], "format": "sql"})
    facts.edge(mid, tid, "defines", ev.MANIFEST, line=line)
    for cols, target, refcols in fks:
        if target != name:
            facts.ref(tid, f"TABLE:{target}", "link", line, edge="depends_on", evidence=ev.MANIFEST,
                      attrs={"foreign_key": True, "columns": cols, "references": refcols})


def _split_top(body: str) -> list[str]:
    parts, depth, cur = [], 0, []
    for ch in body:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    if "".join(cur).strip():
        parts.append("".join(cur))
    return parts


# ------------------------------------------------------------------ Prisma
def _prisma(facts: Facts, path: str, text: str, mid: str) -> None:
    models = {m.group(2) for m in re.finditer(r"^\s*(model|view)\s+(\w+)\s*\{", text, re.M)}
    enums = {m.group(1) for m in re.finditer(r"^\s*enum\s+(\w+)\s*\{", text, re.M)}
    ds = re.search(r'datasource\s+\w+\s*\{[^}]*provider\s*=\s*"(\w+)"', text)
    if ds:
        facts.nodes[0]["attrs"]["provider"] = ds.group(1)
    for m in re.finditer(r"^\s*(model|view|enum|type)\s+(\w+)\s*\{(.*?)^\s*\}", text, re.M | re.S):
        kind, name, body = m.group(1), m.group(2), m.group(3)
        line = text.count("\n", 0, m.start()) + 1
        if kind == "enum":
            eid = f"prisma:{name}"
            facts.node(eid, "schema_type", name, ev.MANIFEST, file=path, line_start=line,
                       attrs={"enum": re.findall(r"^\s*(\w+)", body, re.M), "format": "prisma"})
            facts.edge(mid, eid, "defines", ev.MANIFEST, line=line)
            continue
        mapped = re.search(r'@@map\(\s*(?:name:\s*)?"([^"]+)"', body)
        tname = mapped.group(1) if mapped else name
        tid = table_id(tname)
        columns, fks = [], []
        for f in re.finditer(r"^\s*(\w+)\s+(\w+)(\[\])?(\?)?(.*)$", body, re.M):
            fname, ftype, is_list, optional, rest = f.groups()
            if fname.startswith("@@"):
                continue
            if ftype in models:
                rel = re.search(r"@relation\([^)]*fields:\s*\[([^\]]*)\][^)]*references:\s*\[([^\]]*)\]", rest)
                if rel:
                    fks.append((rel.group(1).strip(), ftype, rel.group(2).strip()))
                continue
            col = re.search(r'@map\(\s*"([^"]+)"', rest)
            columns.append({"name": col.group(1) if col else fname, "type": ftype.lower(),
                            "nullable": bool(optional), "enum": ftype in enums,
                            "primary_key": "@id" in rest})
        facts.node(tid, "table", norm_table(tname), ev.MANIFEST, file=path, line_start=line,
                   line_end=text.count("\n", 0, m.end()) + 1,
                   attrs={"columns": columns, "model": name, "view": kind == "view", "format": "prisma",
                          "primary_key": [c["name"] for c in columns if c.get("primary_key")]})
        facts.edge(mid, tid, "defines", ev.MANIFEST, line=line)
        for cols, target, refcols in fks:
            facts.ref(tid, f"PRISMA:{target}", "link", line, edge="depends_on", evidence=ev.MANIFEST,
                      attrs={"foreign_key": True, "columns": cols, "references": refcols})


# ------------------------------------------------------------------ GraphQL
ROOT_TYPES = {"Query", "Mutation", "Subscription"}
SCALARS = {"String", "Int", "Float", "Boolean", "ID"}


def _graphql(facts: Facts, path: str, text: str, mid: str) -> None:
    clean = re.sub(r'"""[\s\S]*?"""|#[^\n]*', lambda m: re.sub(r"[^\n]", " ", m.group(0)), text)
    for m in re.finditer(r"^\s*(extend\s+)?(type|input|interface|enum|union|scalar)\s+(\w+)([^{\n]*)(\{([^}]*)\})?",
                         clean, re.M):
        kind, name, extra, body = m.group(2), m.group(3), m.group(4) or "", m.group(6) or ""
        line = clean.count("\n", 0, m.start()) + 1
        fields = re.findall(r"^\s*(\w+)\s*(\([^)]*\))?\s*:\s*([\[\]\w!]+)", body, re.M)
        if name in ROOT_TYPES and kind == "type":
            for fname, args, ftype in fields:
                eid = f"graphql:{name}.{fname}"
                fline = line + body[:body.find(fname)].count("\n") + 1 if fname in body else line
                facts.node(eid, "endpoint", f"{name}.{fname}", ev.MANIFEST, file=path, line_start=fline,
                           attrs={"protocol": "graphql", "operation": name.lower(), "returns": ftype,
                                  "args": (args or "").strip("()")[:200]})
                facts.edge(mid, eid, "defines", ev.MANIFEST, line=fline)
                base = re.sub(r"[\[\]!]", "", ftype)
                if base not in SCALARS:
                    facts.ref(eid, f"ID:graphql:type:{base}", "link", fline, edge="produces", evidence=ev.MANIFEST)
                facts.ref(eid, f"SYMBOL:{fname}", "link", fline, edge="specifies", evidence=ev.NAME_MATCH,
                          confidence=0.4)
            continue
        tid = f"graphql:type:{name}"
        facts.node(tid, "schema_type", name, ev.MANIFEST, file=path, line_start=line,
                   attrs={"kind": kind, "fields": [f[0] for f in fields][:100], "format": "graphql",
                          "implements": re.findall(r"\w+", extra.replace("implements", ""))[:10]})
        facts.edge(mid, tid, "defines", ev.MANIFEST, line=line)
        for _f, _a, ftype in fields:
            base = re.sub(r"[\[\]!]", "", ftype)
            if base not in SCALARS and base != name:
                facts.ref(tid, f"ID:graphql:type:{base}", "link", line, edge="references", evidence=ev.MANIFEST)


# ------------------------------------------------------------------ Protocol Buffers
def _proto(facts: Facts, path: str, text: str, mid: str) -> None:
    clean = re.sub(r"//[^\n]*|/\*[\s\S]*?\*/", lambda m: re.sub(r"[^\n]", " ", m.group(0)), text)
    pkg_m = re.search(r"^\s*package\s+([\w.]+)\s*;", clean, re.M)
    pkg = pkg_m.group(1) if pkg_m else ""
    q = (lambda n: f"{pkg}.{n}" if pkg and "." not in n else n)  # noqa: E731
    for m in re.finditer(r'^\s*import\s+(?:public\s+|weak\s+)?"([^"]+)"', clean, re.M):
        facts.ref(mid, f"FILE:{m.group(1)}", "link", clean.count("\n", 0, m.start()) + 1, edge="imports",
                  evidence=ev.MANIFEST)
    for m in re.finditer(r"^\s*(message|enum)\s+(\w+)\s*\{", clean, re.M):
        name, line = m.group(2), clean.count("\n", 0, m.start()) + 1
        body = _braced(clean, m.end() - 1)
        tid = f"proto:{q(name)}"
        fields = re.findall(r"(?:^|[;{])\s*(?:repeated\s+|optional\s+|required\s+)?([\w.]+)\s+(\w+)\s*=\s*\d+",
                            body, re.M)
        facts.node(tid, "schema_type", name, ev.MANIFEST, file=path, line_start=line,
                   attrs={"kind": m.group(1), "package": pkg, "fields": [f[1] for f in fields][:100],
                          "format": "protobuf"})
        facts.edge(mid, tid, "defines", ev.MANIFEST, line=line)
        for ftype, _ in fields:
            if ftype[:1].isupper() or "." in ftype:
                facts.ref(tid, f"ID:proto:{q(ftype)}", "link", line, edge="references", evidence=ev.MANIFEST)
    for m in re.finditer(r"^\s*service\s+(\w+)\s*\{", clean, re.M):
        svc = m.group(1)
        body = _braced(clean, m.end() - 1)
        base_line = clean.count("\n", 0, m.start()) + 1
        for r in re.finditer(r"rpc\s+(\w+)\s*\(\s*(stream\s+)?([\w.]+)\s*\)\s*returns\s*\(\s*(stream\s+)?([\w.]+)\s*\)",
                             body):
            line = base_line + body.count("\n", 0, r.start())
            eid = f"rpc:{q(svc)}/{r.group(1)}"
            facts.node(eid, "endpoint", f"{svc}/{r.group(1)}", ev.MANIFEST, file=path, line_start=line,
                       attrs={"protocol": "grpc", "service": q(svc), "request": r.group(3), "response": r.group(5),
                              "streaming": [s for s, on in (("client", r.group(2)), ("server", r.group(4))) if on]})
            facts.edge(mid, eid, "defines", ev.MANIFEST, line=line)
            facts.ref(eid, f"ID:proto:{q(r.group(3))}", "link", line, edge="consumes", evidence=ev.MANIFEST)
            facts.ref(eid, f"ID:proto:{q(r.group(5))}", "link", line, edge="produces", evidence=ev.MANIFEST)
            facts.ref(eid, f"SYMBOL:{r.group(1)}", "link", line, edge="specifies", evidence=ev.NAME_MATCH,
                      confidence=0.4)


def _braced(text: str, open_pos: int) -> str:
    depth, i = 0, open_pos
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[open_pos + 1:i]
        i += 1
    return text[open_pos + 1:]

