-- core schema
CREATE TABLE IF NOT EXISTS docs (
  id TEXT PRIMARY KEY,
  text TEXT NOT NULL,
  created_at TIMESTAMP DEFAULT now()
);

CREATE TABLE tags (
  id SERIAL PRIMARY KEY,
  doc_id TEXT NOT NULL REFERENCES docs(id) ON DELETE CASCADE,
  label TEXT NOT NULL,
  CONSTRAINT uq UNIQUE (doc_id, label)
);

CREATE VIEW tagged AS SELECT d.id, t.label FROM docs d JOIN tags t ON t.doc_id = d.id;
CREATE INDEX ix_tags_label ON tags (label);
CREATE TYPE status AS ENUM ('draft', 'live');
