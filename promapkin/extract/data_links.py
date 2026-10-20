"""Links from code to data stores, for any language: SQL in string literals, ORM table mappings and schema
migrations. Runs on the facts of a code file after its extractor.

- A string literal that is a SQL statement (`SELECT ... FROM orders`, `INSERT INTO users ...`) gives the
  enclosing function a `reads`/`writes` edge to `table:<name>` (lexical evidence).
- ORM mappings (`__tablename__`, Django models and `db_table`, `@Entity`/`@Table`, Sequelize `define`,
  GORM `TableName()`, ActiveRecord classes, Diesel `table_name`) give the class a `maps_to` edge.
- Prisma client calls (`prisma.user.findMany()`) read or write the Prisma model's table.
- Migrations (`op.create_table`, `create_table :users`, `knex.schema.createTable`, Django `CreateModel`)
  give the migration a `migrates` edge.

Tables that no DDL in the repository defines are created on first use as shared `table:` nodes (marked as
inferred from code), so data flow reaches the store either way.
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Any

from promapkin import evidence as ev
from promapkin.extract import Facts
from promapkin.extract.schema_ext import norm_table, tables_in

STRINGS = re.compile(r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'|`(?:\\.|[^`\\])*`|"(?:\\.|[^"\\\n])*"|'
                     r"'(?:\\.|[^'\\\n])*'")
SQL_START = re.compile(r"^\s*(?:\(\s*)?(SELECT|INSERT|UPDATE|DELETE|WITH|MERGE|REPLACE\s+INTO|UPSERT|TRUNCATE|COPY)\b",
                       re.I)
ORM_PATTERNS = [
    (re.compile(r"""__tablename__\s*=\s*["']([\w.]+)["']"""), "sqlalchemy"),
    (re.compile(r"""\bdb_table\s*=\s*["']([\w.]+)["']"""), "django"),
    (re.compile(r"""@Entity\(\s*(?:\{\s*name\s*:\s*)?["']([\w.]+)["']"""), "typeorm"),
    (re.compile(r"""@Table\(\s*(?:name\s*=\s*|\{\s*tableName\s*:\s*)?["']([\w.]+)["']"""), "jpa"),
    (re.compile(r"""\btableName\s*:\s*["']([\w.]+)["']"""), "sequelize"),
    (re.compile(r"""\.define\(\s*["']([\w.]+)["']\s*,\s*\{"""), "sequelize"),
    (re.compile(r"""TableName\(\)\s*string\s*\{\s*return\s*"([\w.]+)\""""), "gorm"),
    (re.compile(r"""self\.table_name\s*=\s*["']([\w.]+)["']"""), "activerecord"),
    (re.compile(r"""table_name\s*=\s*["']?([\w.]+)["']?\s*\)\]"""), "diesel/sea-orm"),
    (re.compile(r"""#\[sqlx?\(.*?table\s*=\s*"([\w.]+)\""""), "sqlx"),
]
MIGRATIONS = [
    re.compile(r"""\bop\.(?:create_table|drop_table|rename_table)\(\s*["']([\w.]+)["']"""),
    re.compile(r"""\bop\.(?:add_column|drop_column|alter_column|create_index|create_foreign_key)\(\s*["'][\w.]*["']\s*,\s*["']([\w.]+)["']"""),
    re.compile(r"""\b(?:create_table|drop_table|change_table|add_column|remove_column|add_reference)\s*\(?\s*[:"']([\w.]+)"""),
    re.compile(r"""\bschema\.(?:createTable|alterTable|table|dropTable)\(\s*["']([\w.]+)["']"""),
    re.compile(r"""migrations\.CreateModel\(\s*name\s*=\s*["'](\w+)["']"""),
]
PRISMA_CALL = re.compile(r"\bprisma\.(\w+)\.(findMany|findUnique|findUniqueOrThrow|findFirst|findFirstOrThrow|count|"
                         r"aggregate|groupBy|create|createMany|update|updateMany|upsert|delete|deleteMany)\b")
PRISMA_WRITES = {"create", "createMany", "update", "updateMany", "upsert", "delete", "deleteMany"}


def annotate(facts: Facts, text: str, language: str) -> int:
    """Add data links to `facts`; returns how many were added."""
    spans = _spans(facts)
    module = next((n["id"] for n in facts.nodes if n["kind"] == "module"), f"file:{facts.path}")
    added = 0

    def owner_at(line: int, kinds: tuple[str, ...] = ("function", "method", "test")) -> str:
        best = None
        for start, end, nid, kind in spans:
            if kind in kinds and start <= line <= end and (best is None or start >= best[0]):
                best = (start, nid)
        return best[1] if best else module

    def link(src: str, table: str, edge: str, line: int, how: str, conf: float, create: bool = True) -> None:
        nonlocal added
        name = norm_table(table)
        if not name or not re.match(r"[a-z_]", name):
            return
        facts.ref(src, f"TABLE:{name}", "link", line, edge=edge, evidence=ev.LEXICAL if how == "sql" else ev.HEURISTIC,
                  confidence=conf, create=create, attrs={"via": how})
        added += 1

    if language not in ("sql", "markdown", "manifest"):
        for m in STRINGS.finditer(text):
            body = m.group(0)
            body = body[3:-3] if body[:3] in ('"""', "'''") else body[1:-1]
            ddl = re.match(r"\s*CREATE\s+(?:TEMP(?:ORARY)?\s+)?TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?"
                           r"([\w.\"`\[\]]+)", body, re.I)
            if ddl:
                link(owner_at(text.count("\n", 0, m.start()) + 1), ddl.group(1), "migrates",
                     text.count("\n", 0, m.start()) + 1, "sql", 0.8)
                continue
            if not SQL_START.match(body) or len(body) > 20000:
                continue
            if re.match(r"\s*UPDATE\b", body, re.I) and not re.search(r"\bSET\b", body, re.I):
                continue
            if re.match(r"\s*\(?\s*SELECT\b", body, re.I) and not re.search(r"\bFROM\b", body, re.I):
                continue
            line = text.count("\n", 0, m.start()) + 1
            reads, writes = tables_in(body)
            owner = owner_at(line)
            for t in sorted(writes):
                link(owner, t, "writes", line, "sql", 0.75)
            for t in sorted(reads):
                link(owner, t, "reads", line, "sql", 0.75)
    for pattern, orm in ORM_PATTERNS:
        for m in pattern.finditer(text):
            line = text.count("\n", 0, m.start()) + 1
            cls = _class_at(spans, line, skip=("Meta",))
            if cls:
                link(cls, m.group(1), "maps_to", line, orm, 0.9)
    _django_implicit(facts, text, spans, link)
    _activerecord_implicit(facts, text, spans, link)
    for m in PRISMA_CALL.finditer(text):
        line = text.count("\n", 0, m.start()) + 1
        facts.ref(owner_at(line), f"PRISMA:{m.group(1)}", "link", line,
                  edge="writes" if m.group(2) in PRISMA_WRITES else "reads", evidence=ev.STATIC_RESOLVED,
                  confidence=0.85, attrs={"via": f"prisma.{m.group(1)}.{m.group(2)}"})
        added += 1
    if "migration" in facts.path.lower() or "/versions/" in facts.path or "alembic" in text[:3000]:
        for pattern in MIGRATIONS:
            for m in pattern.finditer(text):
                line = text.count("\n", 0, m.start()) + 1
                name = m.group(1)
                if "CreateModel" in pattern.pattern:
                    name = f"{_django_app(facts.path)}_{name.lower()}"
                link(module, name, "migrates", line, "migration", 0.8)
    return added


def _spans(facts: Facts) -> list[tuple[int, int, str, str]]:
    out = []
    for n in facts.nodes:
        if n.get("line_start") and n["kind"] in ("class", "function", "method", "test"):
            out.append((n["line_start"], n.get("line_end") or n["line_start"], n["id"], n["kind"]))
    return out


def _class_at(spans: list[tuple[int, int, str, str]], line: int, skip: tuple[str, ...] = ()) -> str | None:
    best = None
    for start, end, nid, kind in spans:
        if kind == "class" and start <= line <= end and nid.rsplit(".", 1)[-1].rsplit(":", 1)[-1] not in skip:
            if best is None or start >= best[0]:
                best = (start, nid)
    return best[1] if best else None


def _django_app(path: str) -> str:
    parts = PurePosixPath(path).parts
    if "migrations" in parts:
        idx = parts.index("migrations")
        return parts[idx - 1] if idx > 0 else "app"
    return PurePosixPath(path).parent.name or "app"


def _django_implicit(facts: Facts, text: str, spans: list, link: Any) -> None:
    if "models.Model" not in text or "db_table" in text and "class Meta" in text and text.count("db_table") >= \
            text.count("models.Model"):
        return
    app = _django_app(facts.path)
    explicit = {r["src"] for r in facts.refs if r["context"].get("attrs", {}).get("via") == "django"}
    for n in facts.nodes:
        if n["kind"] == "class" and any("models.Model" in b or b.endswith("Model") and "models." in b
                                        for b in n.get("attrs", {}).get("bases", [])) and n["id"] not in explicit:
            if n.get("attrs", {}).get("abstract"):
                continue
            link(n["id"], f"{app}_{n['name'].lower()}", "maps_to", n.get("line_start") or 1, "django", 0.7)


def _activerecord_implicit(facts: Facts, text: str, spans: list, link: Any) -> None:
    if facts.language != "ruby" or not re.search(r"<\s*(ApplicationRecord|ActiveRecord::Base)", text):
        return
    if "self.table_name" in text:
        return
    for n in facts.nodes:
        if n["kind"] == "class" and any(b in ("ApplicationRecord", "ActiveRecord::Base", "ActiveRecord.Base")
                                        for b in n.get("attrs", {}).get("bases", [])):
            snake = re.sub(r"(?<!^)(?=[A-Z])", "_", n["name"]).lower()
            link(n["id"], snake + ("es" if snake.endswith(("s", "x", "ch", "sh")) else "s"), "maps_to",
                 n.get("line_start") or 1, "activerecord", 0.6)
