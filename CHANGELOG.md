# Changelog

## 0.2.0

### Fixed
- `IndexError` in ML disambiguation when only one candidate remained, which rolled back whole scans on real
  repositories. Enrichment passes (classification, embeddings, graph ML) can no longer discard the structural index:
  a failure becomes a warning.
- Quadratic full-text index maintenance (deletes scanned an unindexed FTS column). fastapi (2,886 files) now scans
  in ~36 s instead of more than 5 minutes; the resolver's documentation-mention lookup is now a dictionary lookup
  (76 s → 1.9 s).
- Path tracing was too strict: `trace` now falls back from flow edges to structural edges to undirected relations,
  reports which mode answered and prints a one-line summary per path.
- Exact identifier queries (`Command`) now rank definitions above test helpers and modules with the same name.

### Added
- Languages: Go, Rust, Java, C#, Kotlin, Ruby, PHP, C, C++, Swift and Scala via tree-sitter (optional), with a
  lexical fallback; Rust `crate::`/`super::`/workspace-crate import resolution and `Type::method` calls; generic
  path-style import resolution for package-path languages.
- JavaScript/TypeScript through the TypeScript compiler API (optional): real AST, type-checker call resolution
  (`this.repo.findAll()` through the declared type), interfaces, implements, routes and event wiring.
- Dynamic wiring: reflection (`getattr` with constants), dynamic imports, callbacks, FastAPI `Depends`, pytest
  fixtures, event emit/subscribe, signals, Celery-style `.delay`, dotted-string references; generated-code marking.
- Runtime evidence: `promapkin trace run -- <your command>` (Python profiling hook, Node V8 coverage) and ingestion
  into runtime call edges, test→code links and execution counts; runtime edges go stale when code changes.
- Incremental reference resolution: only files whose references could change are re-resolved.
- LLM mode: `--llm [anthropic|openai]` on every command (official SDKs; OpenAI-compatible base URLs) replaces
  regex/heuristic interpretation with grounded LLM analysis; `ask`, `review`, `auth`; LLM providers for external
  agents mode (`provider: anthropic|openai` with base URL and key).
- Graph insights: `graph hubs|communities|surprises|html` (offline interactive explorer) and matching MCP tools.
- `doctor`, `parsers status|install`, schema migrations, per-phase scan timings, CI workflow, security policy.

## 0.1.0
- Initial release: Python AST + lexical JS/TS extraction, evidence-aware graph, blueprints, retrieval, handoffs,
  MCP server, ML tier, benchmarks.
