# Changelog

## Unreleased

### Added
- **`promapkin prs`: pull-request dashboard.**
  - Shows open PRs from GitHub (through `gh`, or `GITHUB_TOKEN`/`GH_TOKEN`, with `GH_HOST` for Enterprise), or
    local branches with `--local`.
  - For each PR: CI state, review state, the worktree that has the branch checked out, and the changed modules,
    communities and symbols.
  - Also: blast radius, tests to run, a risk score, and merge-conflict risk between PRs from overlapping files,
    symbols, modules and communities (`--conflicts`).
  - `--triage` ranks the PRs (deterministic, or with an LLM under `--llm`).
  - MCP tool: `get_pull_requests`.
- **Automatic re-indexing.**
  - `promapkin hook install|uninstall|status` adds a marked block to `post-commit`, `post-checkout`,
    `post-merge` and `post-rewrite`.
    - The block runs an incremental scan in the background; `--blueprint` runs `sync` instead.
    - Existing hooks are kept, `core.hooksPath` is honoured, and `PROMAPKIN_HOOKS=0` turns the hooks off.
  - `promapkin watch` (or `scan --watch`) polls the tree and re-indexes once changes settle. It needs no
    dependencies.
- **`promapkin add <url|path>`** brings outside material into the index:
  - arXiv papers: abstract, metadata and PDF text;
  - YouTube videos: captions through `yt-dlp`, or a transcript through `faster-whisper` when installed;
  - web pages: converted to Markdown;
  - local files.

  Sources are stored in `.promapkin/sources/` and linked to the code they mention. Only http(s) is fetched, and
  private or loopback hosts are refused unless you pass `--allow-private`. Manage them with
  `promapkin sources list|remove`; MCP tools `add_source` and `list_sources`.
- **`promapkin graph export --format json|graphml`** and **`promapkin graph merge`**.
  - `merge` combines several repositories (or exported graphs) into one index with namespaced ids.
  - It adds cross-repository links: a client call → the endpoint that serves it (`provided_by`), an OpenAPI
    spec → its implementation (`specifies`), shared tables and protobuf messages (`same_as`), and an
    infrastructure repo → the code it deploys (`deploys`).
  - MCP tool: `export_graph`.
- **`promapkin wiki`** (or `scan --wiki`) writes a linked Markdown wiki:
  - an index;
  - one page per subsystem and per module;
  - endpoints, data, infrastructure and decisions pages.

  MCP tool: `export_wiki`.
- **`promapkin graph callflow [entry]`** writes Mermaid call-flow diagrams grouped by file, as an HTML page plus
  `.mmd` sources. Dashed arrows are inferred calls. MCP tool: `render_callflow`.
- **Hyperedges:** relationships among more than two nodes.
  - Kinds: files that change together in git history (`co_change`), entry point → sink flows (`flow`), the
    participants of a design pattern (`pattern`), and communities.
  - `graph hyperedges` lists them; `graph.html` shades them as hulls you can toggle; MCP tool `get_hyperedges`.
- **`graph surprises --semantic`:** near-duplicate logic across communities that has no structural link
  (`similar_to` from embeddings). MCP tool: `find_duplicate_logic`.

### Changed
- **Faster scans on large repositories.**
  - Parsing and extraction run across processes when a scan touches 200 or more files and no LLM is attached.
  - Set the number of workers with `scan_workers` in the config or `PROMAPKIN_WORKERS`.
  - On fastapi, the parse phase drops from 10.7 s to 2.9 s with 4 processes.

### Fixed
- **Python 3.10 support:**
  - a built-in TOML reader replaces `tomllib` where it is missing, so Cargo workspaces and `pyproject.toml`
    resolve again;
  - runtime tracing attributes calls made inside comprehensions (which run in their own frames before 3.12) to
    the enclosing function.

## 0.3.0

### Added
- **50 languages.** Tree-sitter extraction for 35 more:
  - functional: Elixir, Erlang, Haskell, OCaml, F#, Clojure, Common Lisp, Scheme, Racket, Elm, Gleam;
  - systems: Zig, Nim, D, Odin, V, Objective-C;
  - scripting and data: Dart, Lua, R, Julia, Perl, Bash, PowerShell, Crystal;
  - hardware and GPU: Verilog, VHDL, CUDA, GLSL, HLSL;
  - other: Solidity, Fortran, Pascal, MATLAB, Apex.

  Also Groovy (lexical) and Vue/Svelte single-file components. New languages are declarative specs; a grammar
  that cannot load falls back to lexical extraction per file, and `parsers install` pre-downloads all grammars.
- **`promapkin install` for 19 coding assistants:** Claude Code, Codex, Gemini CLI, OpenCode, VS Code/Copilot,
  Cursor, Windsurf, Cline, Roo Code, Kiro, Zed, Continue, Amazon Q, Junie, Goose, Crush, Augment, Trae and
  `AGENTS.md`.
  - What it writes: MCP registration, instructions, and where supported, hooks, skills and slash commands.
  - Changes are merge-safe and recorded; `--dry-run` previews them and `promapkin uninstall` restores the files.
- **`promapkin brief`** and the MCP tool `get_session_brief`: a compact factual orientation used by
  session-start hooks.
- **Infrastructure and schemas as graph nodes**, linked to code:
  - formats: Dockerfile, Docker Compose, Kubernetes (including Helm templates), Terraform, GitHub Actions and
    GitLab CI, OpenAPI, SQL DDL, Prisma, GraphQL and protobuf;
  - links: what each container, workload or CI job runs; config keys shared between infrastructure and the
    code that reads them; Service→workload by selector; Ingress→Service; Terraform references; spec→endpoint;
  - views: `graph infra`, `graph schema`, diagrams `infrastructure` and `er`, MCP `get_infrastructure` and
    `get_data_model`, and Infrastructure and Data model sections in the architecture blueprint.
- **Code→data links in any language:** SQL in string literals (reads and writes per function), ORM mappings
  (SQLAlchemy, Django, TypeORM, JPA, Sequelize, GORM, ActiveRecord, Diesel), Prisma client calls and
  migrations. Data flow now ends at tables and cloud resources.
- **Assets:**
  - PDF text (built in; pypdf when installed), Word and PowerPoint text, image dimensions and SVG text;
  - optional OCR, and `--llm` vision descriptions of diagrams whose drawn arrows become claims;
  - video duration;
  - extracted text is searchable and linked to the code it mentions;
  - `graph assets`, MCP `list_assets`.
- **Published retrieval benchmark** on fastapi, clap and Promapkin itself: hand-written specs and raw results in
  `examples/bench/`, write-up in docs/BENCHMARK.md.
- **New `graph.html`:**
  - shows infrastructure, data and asset nodes;
  - shapes by category;
  - legend toggles;
  - directed, colour-coded edges;
  - 1–2 hop focus;
  - search suggestions;
  - edge-type filter;
  - theme toggle;
  - PNG export.
- `doctor` reports pypdf, PyYAML, OCR and detected assistants. New extras: `assets` (pypdf) and `yaml` (PyYAML).
- `scripts/gen_mcp_docs.py` regenerates docs/MCP_TOOLS.md, which now lists 56 tools.

### Changed
- Retrieval down-weights tests, examples and vendored code unless the question asks for them. Documentation
  sections that match a question pass their score to the code they reference. On fastapi, the share of context
  packages containing a right file rose from 27% to 68%.
- Tree-sitter doc comments are read in full, including across attributes and decorators (`#[cfg]`,
  `@Annotation`). clap recall@10 went from 0.36 to 0.64.
- Context packages are cached per retrieval version; `bench retrieval` never reads the cache.

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
