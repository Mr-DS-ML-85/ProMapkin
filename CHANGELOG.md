# Changelog

## Unreleased

### Fixed
- `--llm` with OpenAI-compatible proxies that ignore `response_format` (for example OmniRoute's DeepSeek Web
  provider, `ds-web/...`): document analysis failed with a bare "model returned invalid JSON". Structured replies now
  report what went wrong (empty, truncated, wrapped in other text, malformed, not an object, schema mismatch) with the
  finish reason and length, and are validated against the requested schema instead of being accepted on shape alone.
  New `openai_structured_output` setting (`json_schema` | `json_object` | `prompt`; env `PROMAPKIN_LLM_STRUCTURED`)
  for endpoints that do not enforce schemas. `PROMAPKIN_LLM_DEBUG=1` keeps invalid replies (secrets redacted) in
  `.promapkin/llm-debug/`.
- A failed LLM document analysis could leave a document's facts half-replaced; it is now all-or-nothing and the
  deterministic Markdown facts are kept. These failures now count as fallbacks in the usage line. Cached responses
  that no longer match their schema are ignored.
- `database is locked` when two promapkin processes wrote at once (for example a `--llm` scan or the MCP server and
  `blueprint generate`): scans parsed files and called the LLM while holding the index's write lock, for minutes.
  Parsing and every LLM call now happen with no lock held, so write transactions are short. If the lock still cannot
  be acquired within `db_busy_timeout` (default 30 s), commands exit with code 3 and an "index busy" message
  instead of a traceback.
- `ask`: the question is repeated in the system prompt, so proxies that replay only the last messages of a long
  tool loop no longer lose it.
- `graph similar "<free text>"` treated the text as a node name (exit 2, or neighbours of an arbitrary node).

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
