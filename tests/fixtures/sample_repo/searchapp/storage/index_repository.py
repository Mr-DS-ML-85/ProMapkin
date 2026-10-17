"""Persistence for indexed documents (SQLite)."""
import sqlite3


class IndexRepository:
    """Repository that stores and loads documents."""

    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("CREATE TABLE IF NOT EXISTS docs (id TEXT PRIMARY KEY, text TEXT)")

    def add(self, doc_id: str, text: str) -> None:
        self.conn.execute("INSERT OR REPLACE INTO docs VALUES (?, ?)", (doc_id, text))
        self.conn.commit()

    def find_all(self) -> list:
        rows = self.conn.execute("SELECT id, text FROM docs").fetchall()
        return [{"id": r[0], "text": r[1]} for r in rows]

    def get(self, doc_id: str):
        row = self.conn.execute("SELECT id, text FROM docs WHERE id = ?", (doc_id,)).fetchone()
        return {"id": row[0], "text": row[1]} if row else None
