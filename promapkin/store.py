"""Embedded, transactional storage for the knowledge graph.

SQLite keeps Promapkin free of external infrastructure. All writes performed by
a scan happen inside one transaction, so a failed analysis job never corrupts
the last valid index. The storage API is deliberately small so the backend can
be replaced behind it.
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator

from promapkin import SCHEMA_VERSION

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);

CREATE TABLE IF NOT EXISTS files (
    path TEXT PRIMARY KEY,
    sha TEXT NOT NULL,
    language TEXT,
    size INTEGER,
    revision TEXT,
    scanned_at REAL
);

CREATE TABLE IF NOT EXISTS nodes (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    name TEXT NOT NULL,
    qualname TEXT,
    file TEXT,
    line_start INTEGER,
    line_end INTEGER,
    owner_file TEXT,           -- file whose extraction produced this node (for incremental updates)
    evidence TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 1.0,
    status TEXT NOT NULL DEFAULT 'verified',
    revision TEXT,
    verified_at REAL,
    text TEXT,                 -- searchable description (docstring, doc section, claim)
    attrs TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS nodes_kind ON nodes(kind);
CREATE INDEX IF NOT EXISTS nodes_name ON nodes(name);
CREATE INDEX IF NOT EXISTS nodes_owner ON nodes(owner_file);

CREATE TABLE IF NOT EXISTS edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    src TEXT NOT NULL,
    dst TEXT NOT NULL,
    type TEXT NOT NULL,
    evidence TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 1.0,
    status TEXT NOT NULL DEFAULT 'verified',
    file TEXT,
    line INTEGER,
    owner_file TEXT,
    revision TEXT,
    verified_at REAL,
    attrs TEXT NOT NULL DEFAULT '{}',
    UNIQUE(src, dst, type, evidence)
);
CREATE INDEX IF NOT EXISTS edges_src ON edges(src);
CREATE INDEX IF NOT EXISTS edges_dst ON edges(dst);
CREATE INDEX IF NOT EXISTS edges_owner ON edges(owner_file);

-- Unresolved references collected during extraction; resolved into edges after every scan.
CREATE TABLE IF NOT EXISTS refs (
    owner_file TEXT NOT NULL,
    src TEXT NOT NULL,
    target TEXT NOT NULL,
    type TEXT NOT NULL,
    line INTEGER,
    context TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS refs_owner ON refs(owner_file);

CREATE VIRTUAL TABLE IF NOT EXISTS search USING fts5(
    id UNINDEXED, kind UNINDEXED, name, body, tokenize = 'porter unicode61'
);

CREATE TABLE IF NOT EXISTS handoffs (
    id TEXT NOT NULL,
    version INTEGER NOT NULL,
    task TEXT NOT NULL,
    status TEXT NOT NULL,
    parent_id TEXT,
    agent TEXT,
    revision TEXT,
    created_at REAL NOT NULL,
    data TEXT NOT NULL,
    PRIMARY KEY (id, version)
);

CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at REAL NOT NULL,
    finished_at REAL,
    revision TEXT,
    detail TEXT
);

CREATE TABLE IF NOT EXISTS embeddings (
    id TEXT NOT NULL,
    model TEXT NOT NULL,
    sha TEXT NOT NULL,
    vec BLOB NOT NULL,
    PRIMARY KEY (id, model)
);

-- Memoized classifier decisions, keyed by task + input text hash.
CREATE TABLE IF NOT EXISTS classifications (
    task TEXT NOT NULL,
    key TEXT NOT NULL,
    label TEXT NOT NULL,
    confidence REAL NOT NULL,
    tier TEXT NOT NULL,
    model TEXT,
    detail TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (task, key, tier)
);

-- Bounded analysis tasks for specialist agents (internal providers or external agents over MCP).
CREATE TABLE IF NOT EXISTS agent_tasks (
    id TEXT PRIMARY KEY,
    specialist TEXT NOT NULL,
    status TEXT NOT NULL,          -- pending | claimed | completed | rejected
    revision TEXT,
    contract TEXT NOT NULL,
    result TEXT,
    claimed_by TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS artifacts (
    name TEXT PRIMARY KEY,
    revision TEXT,
    path TEXT,
    created_at REAL,
    meta TEXT NOT NULL DEFAULT '{}'
);
"""

_IDENT_SPLIT = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")


def split_identifier(text: str) -> list[str]:
    """Split identifiers so 'VectorIndex::search_fast' also matches 'vector', 'index', 'fast'."""
    return [t.lower() for t in _IDENT_SPLIT.findall(text)]


def searchable(*parts: str | None) -> str:
    words: list[str] = []
    for part in parts:
        if not part:
            continue
        words.append(part)
        split = split_identifier(part)
        if split:
            words.append(" ".join(split))
    return " ".join(words)


class Store:
    def __init__(self, path: Path | str):
        self.path = str(path)
        self.conn = sqlite3.connect(self.path, isolation_level=None, timeout=30)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(SCHEMA)
        if self.get_meta("schema_version") is None:
            self.set_meta("schema_version", str(SCHEMA_VERSION))
        self._depth = 0

    def close(self) -> None:
        self.conn.close()

    # ------------------------------------------------------------------ transactions
    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Nestable write transaction; the outermost one commits or rolls back."""
        if self._depth == 0:
            self.conn.execute("BEGIN IMMEDIATE")
        self._depth += 1
        try:
            yield
        except BaseException:
            self._depth -= 1
            if self._depth == 0:
                self.conn.execute("ROLLBACK")
            raise
        else:
            self._depth -= 1
            if self._depth == 0:
                self.conn.execute("COMMIT")

    # ------------------------------------------------------------------ meta
    def get_meta(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", (key, value))

    # ------------------------------------------------------------------ files
    def file_index(self) -> dict[str, str]:
        return {r["path"]: r["sha"] for r in self.conn.execute("SELECT path, sha FROM files")}

    def upsert_file(self, path: str, sha: str, language: str, size: int, revision: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO files(path, sha, language, size, revision, scanned_at) VALUES (?,?,?,?,?,?)",
            (path, sha, language, size, revision, time.time()),
        )

    def forget_file(self, path: str) -> None:
        """Remove everything extracted from a file (before re-extraction or after deletion)."""
        ids = [r[0] for r in self.conn.execute("SELECT id FROM nodes WHERE owner_file=?", (path,))]
        self.conn.execute("DELETE FROM nodes WHERE owner_file=?", (path,))
        self.conn.execute("DELETE FROM edges WHERE owner_file=?", (path,))
        self.conn.execute("DELETE FROM refs WHERE owner_file=?", (path,))
        self.conn.execute("DELETE FROM files WHERE path=?", (path,))
        for chunk in _chunks(ids, 500):
            marks = ",".join("?" * len(chunk))
            self.conn.execute(f"DELETE FROM search WHERE id IN ({marks})", chunk)

    # ------------------------------------------------------------------ nodes
    def upsert_node(
        self,
        id: str,
        kind: str,
        name: str,
        *,
        evidence: str,
        qualname: str | None = None,
        file: str | None = None,
        line_start: int | None = None,
        line_end: int | None = None,
        owner_file: str | None = None,
        confidence: float = 1.0,
        status: str = "verified",
        revision: str | None = None,
        text: str | None = None,
        attrs: dict[str, Any] | None = None,
    ) -> None:
        now = time.time()
        self.conn.execute(
            """INSERT INTO nodes(id, kind, name, qualname, file, line_start, line_end, owner_file, evidence,
                                 confidence, status, revision, verified_at, text, attrs)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET kind=excluded.kind, name=excluded.name, qualname=excluded.qualname,
                 file=COALESCE(excluded.file, nodes.file), line_start=COALESCE(excluded.line_start, nodes.line_start),
                 line_end=COALESCE(excluded.line_end, nodes.line_end),
                 owner_file=COALESCE(excluded.owner_file, nodes.owner_file),
                 evidence=excluded.evidence, confidence=excluded.confidence, status=excluded.status,
                 revision=excluded.revision, verified_at=excluded.verified_at,
                 text=COALESCE(excluded.text, nodes.text), attrs=excluded.attrs""",
            (id, kind, name, qualname, file, line_start, line_end, owner_file, evidence, confidence, status,
             revision, now, text, json.dumps(attrs or {}, sort_keys=True)),
        )
        self.conn.execute("DELETE FROM search WHERE id=?", (id,))
        self.conn.execute(
            "INSERT INTO search(id, kind, name, body) VALUES (?,?,?,?)",
            (id, kind, searchable(name, qualname), searchable(file, text)),
        )

    def node(self, id: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM nodes WHERE id=?", (id,)).fetchone()
        return _node(row) if row else None

    def nodes(self, kind: str | None = None, where: str = "", params: Iterable[Any] = ()) -> list[dict[str, Any]]:
        sql = "SELECT * FROM nodes"
        clauses, args = [], list(params)
        if kind:
            clauses.append("kind=?")
            args.insert(0, kind)
        if where:
            clauses.append(where)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY id"
        return [_node(r) for r in self.conn.execute(sql, args)]

    def find_nodes_by_name(self, name: str, kinds: Iterable[str] | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM nodes WHERE (name=? OR qualname=? OR id=?)"
        args: list[Any] = [name, name, name]
        if kinds:
            kinds = list(kinds)
            sql += f" AND kind IN ({','.join('?' * len(kinds))})"
            args += kinds
        return [_node(r) for r in self.conn.execute(sql, args)]

    def set_node_status(self, id: str, status: str) -> None:
        self.conn.execute("UPDATE nodes SET status=? WHERE id=?", (status, id))

    def delete_nodes(self, where: str, params: Iterable[Any] = ()) -> int:
        ids = [r[0] for r in self.conn.execute(f"SELECT id FROM nodes WHERE {where}", list(params))]
        for chunk in _chunks(ids, 500):
            marks = ",".join("?" * len(chunk))
            self.conn.execute(f"DELETE FROM nodes WHERE id IN ({marks})", chunk)
            self.conn.execute(f"DELETE FROM search WHERE id IN ({marks})", chunk)
            self.conn.execute(f"DELETE FROM edges WHERE src IN ({marks}) OR dst IN ({marks})", chunk + chunk)
        return len(ids)

    # ------------------------------------------------------------------ edges
    def add_edge(
        self,
        src: str,
        dst: str,
        type: str,
        *,
        evidence: str,
        confidence: float = 1.0,
        file: str | None = None,
        line: int | None = None,
        owner_file: str | None = None,
        revision: str | None = None,
        status: str = "verified",
        attrs: dict[str, Any] | None = None,
    ) -> None:
        self.conn.execute(
            """INSERT INTO edges(src, dst, type, evidence, confidence, status, file, line, owner_file, revision,
                                 verified_at, attrs)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(src, dst, type, evidence) DO UPDATE SET confidence=MAX(edges.confidence, excluded.confidence),
                 status=excluded.status, file=COALESCE(excluded.file, edges.file),
                 line=COALESCE(excluded.line, edges.line), revision=excluded.revision,
                 verified_at=excluded.verified_at, attrs=excluded.attrs""",
            (src, dst, type, evidence, confidence, status, file, line, owner_file, revision, time.time(),
             json.dumps(attrs or {}, sort_keys=True)),
        )

    def edges(self, *, src: str | None = None, dst: str | None = None, types: Iterable[str] | None = None,
              where: str = "", params: Iterable[Any] = ()) -> list[dict[str, Any]]:
        clauses, args = [], []
        if src:
            clauses.append("src=?")
            args.append(src)
        if dst:
            clauses.append("dst=?")
            args.append(dst)
        if types:
            types = list(types)
            clauses.append(f"type IN ({','.join('?' * len(types))})")
            args += types
        if where:
            clauses.append(where)
            args += list(params)
        sql = "SELECT * FROM edges" + (" WHERE " + " AND ".join(clauses) if clauses else "") + " ORDER BY id"
        return [_edge(r) for r in self.conn.execute(sql, args)]

    def delete_edges(self, where: str, params: Iterable[Any] = ()) -> int:
        return self.conn.execute(f"DELETE FROM edges WHERE {where}", list(params)).rowcount

    # ------------------------------------------------------------------ refs
    def add_ref(self, owner_file: str, src: str, target: str, type: str, line: int | None,
                context: dict[str, Any] | None = None) -> None:
        self.conn.execute(
            "INSERT INTO refs(owner_file, src, target, type, line, context) VALUES (?,?,?,?,?,?)",
            (owner_file, src, target, type, line, json.dumps(context or {}, sort_keys=True)),
        )

    def refs(self) -> list[dict[str, Any]]:
        out = []
        for r in self.conn.execute("SELECT * FROM refs"):
            d = dict(r)
            d["context"] = json.loads(d["context"])
            out.append(d)
        return out

    # ------------------------------------------------------------------ search
    def search(self, query: str, limit: int = 50, kinds: Iterable[str] | None = None) -> list[tuple[str, float]]:
        terms = sorted({t for t in re.findall(r"\w+", query.lower()) if len(t) > 1}
                       | {t for w in re.findall(r"\w+", query) for t in split_identifier(w) if len(t) > 1})
        if not terms:
            return []
        match = " OR ".join(f'"{t}"' for t in terms)
        sql = "SELECT id, bm25(search, 3.0, 1.0) AS score FROM search WHERE search MATCH ?"
        args: list[Any] = [match]
        if kinds:
            kinds = list(kinds)
            sql += f" AND kind IN ({','.join('?' * len(kinds))})"
            args += kinds
        sql += " ORDER BY score LIMIT ?"
        args.append(limit)
        try:
            return [(r[0], -r[1]) for r in self.conn.execute(sql, args)]
        except sqlite3.OperationalError:
            return []

    # ------------------------------------------------------------------ handoffs
    def handoff_versions(self, id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM handoffs WHERE id=? ORDER BY version", (id,)).fetchall()
        return [_handoff(r) for r in rows]

    def latest_handoff(self, id: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM handoffs WHERE id=? ORDER BY version DESC LIMIT 1", (id,)).fetchone()
        return _handoff(row) if row else None

    def insert_handoff(self, rec: dict[str, Any]) -> None:
        self.conn.execute(
            "INSERT INTO handoffs(id, version, task, status, parent_id, agent, revision, created_at, data) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (rec["id"], rec["version"], rec["task"], rec["status"], rec.get("parent_id"), rec.get("agent"),
             rec.get("revision"), rec["created_at"], json.dumps(rec["data"], sort_keys=True)),
        )

    def list_handoffs(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """SELECT h.* FROM handoffs h JOIN (SELECT id, MAX(version) v FROM handoffs GROUP BY id) m
               ON h.id=m.id AND h.version=m.v ORDER BY h.created_at DESC"""
        ).fetchall()
        return [_handoff(r) for r in rows]

    # ------------------------------------------------------------------ embeddings / classifications
    def embedding_shas(self, model: str) -> dict[str, str]:
        return {r[0]: r[1] for r in self.conn.execute("SELECT id, sha FROM embeddings WHERE model=?", (model,))}

    def put_embedding(self, id: str, model: str, sha: str, vec: bytes) -> None:
        self.conn.execute("INSERT OR REPLACE INTO embeddings(id, model, sha, vec) VALUES (?,?,?,?)",
                          (id, model, sha, vec))

    def embeddings(self, model: str) -> list[tuple[str, bytes]]:
        return [(r[0], r[1]) for r in self.conn.execute(
            "SELECT e.id, e.vec FROM embeddings e JOIN nodes n ON n.id=e.id WHERE e.model=?", (model,))]

    def prune_embeddings(self) -> None:
        self.conn.execute("DELETE FROM embeddings WHERE id NOT IN (SELECT id FROM nodes)")

    def cached_classification(self, task: str, key: str, tier: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM classifications WHERE task=? AND key=? AND tier=?",
                                (task, key, tier)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["detail"] = json.loads(d["detail"])
        return d

    def cache_classification(self, task: str, key: str, tier: str, label: str, confidence: float,
                             model: str | None, detail: dict[str, Any]) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO classifications(task, key, label, confidence, tier, model, detail) "
            "VALUES (?,?,?,?,?,?,?)", (task, key, label, confidence, tier, model, json.dumps(detail)))

    # ------------------------------------------------------------------ agent tasks
    def put_agent_task(self, id: str, specialist: str, revision: str, contract: dict[str, Any]) -> None:
        now = time.time()
        self.conn.execute(
            "INSERT OR REPLACE INTO agent_tasks(id, specialist, status, revision, contract, created_at, updated_at) "
            "VALUES (?,?,?,?,?,?,?)", (id, specialist, "pending", revision, json.dumps(contract), now, now))

    def agent_task(self, id: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM agent_tasks WHERE id=?", (id,)).fetchone()
        return _task(row) if row else None

    def agent_tasks(self, status: str | None = None) -> list[dict[str, Any]]:
        sql, args = "SELECT * FROM agent_tasks", []
        if status:
            sql += " WHERE status=?"
            args.append(status)
        return [_task(r) for r in self.conn.execute(sql + " ORDER BY created_at", args)]

    def update_agent_task(self, id: str, status: str, result: dict[str, Any] | None = None,
                          claimed_by: str | None = None) -> None:
        self.conn.execute(
            "UPDATE agent_tasks SET status=?, result=COALESCE(?, result), claimed_by=COALESCE(?, claimed_by), "
            "updated_at=? WHERE id=?",
            (status, json.dumps(result) if result is not None else None, claimed_by, time.time(), id))

    # ------------------------------------------------------------------ jobs / artifacts
    def start_job(self, kind: str, revision: str | None) -> int:
        cur = self.conn.execute(
            "INSERT INTO jobs(kind, status, started_at, revision) VALUES (?,?,?,?)",
            (kind, "running", time.time(), revision),
        )
        return int(cur.lastrowid)

    def finish_job(self, job_id: int, status: str, detail: dict[str, Any] | None = None) -> None:
        self.conn.execute(
            "UPDATE jobs SET status=?, finished_at=?, detail=? WHERE id=?",
            (status, time.time(), json.dumps(detail or {}), job_id),
        )

    def last_jobs(self, limit: int = 5) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM jobs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def record_artifact(self, name: str, revision: str | None, path: str, meta: dict[str, Any]) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO artifacts(name, revision, path, created_at, meta) VALUES (?,?,?,?,?)",
            (name, revision, path, time.time(), json.dumps(meta, sort_keys=True)),
        )

    def artifacts(self) -> list[dict[str, Any]]:
        return [dict(r) for r in self.conn.execute("SELECT * FROM artifacts ORDER BY name")]

    def counts(self) -> dict[str, Any]:
        kinds = {r[0]: r[1] for r in self.conn.execute("SELECT kind, COUNT(*) FROM nodes GROUP BY kind")}
        etypes = {r[0]: r[1] for r in self.conn.execute("SELECT type, COUNT(*) FROM edges GROUP BY type")}
        files = self.conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        return {"files": files, "nodes": kinds, "edges": etypes}


def _chunks(seq: list[Any], n: int) -> Iterator[list[Any]]:
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def _node(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    d["attrs"] = json.loads(d["attrs"] or "{}")
    return d


def _edge(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    d["attrs"] = json.loads(d["attrs"] or "{}")
    return d


def _task(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    d["contract"] = json.loads(d["contract"])
    d["result"] = json.loads(d["result"]) if d["result"] else None
    return d


def _handoff(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    d["data"] = json.loads(d["data"])
    return d
