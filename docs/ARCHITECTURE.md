# Promapkin internals

```
Interface layer         cli.py · mcp_server.py (56 tools, resources, prompts) · installer.py · brief.py · JSON outputs
        │                         └──── service.py (one engine, two interfaces)
Core orchestration      agents/orchestrator.py (plan → contracts → specialists → validate → merge)
        │
Code intelligence       extract/* (AST, tree-sitter specs, lexical, docs/ADRs, manifests, infra, schemas, assets,
                        data links) · scanner.py (resolver, type inference, link resolution) · infra_view.py
Agent runtime           agents/specialists.py · agents/contract.py · agents/providers.py
ML tier                 ml/embedders.py · ml/classify.py (two-tier) · ml/graph_ml.py · ml/models.py (consent)
Knowledge graph         store.py (nodes/edges/refs/FTS/embeddings) · graph.py · knowledge.py · evidence.py
Retrieval engine        retrieval.py (hybrid search, graph expansion, budget-aware compiler)
Persistence/artifacts   .promapkin/index.db (SQLite WAL) · blueprints/ · handoffs/ · maps/ · training/
```

## Data model

- **Nodes.** Kinds: repository, directory, file, module, class, function, method, test, endpoint,
  package, resource, config, document, doc_section, decision, code_note, manifest, entrypoint,
  pattern, finding, concept, constraint, handoff, roadmap_step, map. Infrastructure and data add service,
  deployment, container_image, ingress, cloud_resource, iac_module, iac_variable, pipeline, config_source, table,
  schema_type, api_operation and asset. Each node carries:
  - `evidence`, `confidence` and `status` (verified, stale, unsupported, superseded);
  - `revision` and `verified_at`;
  - `owner_file`, used for incremental updates;
  - `attrs`, a JSON blob.
- **Edges.** Typed: contains, defines, imports, calls, instantiates, inherits, overrides,
  references, depends_on, routes_to, reads, writes, invokes, verifies, documents, constrains,
  affects, supports, conflicts_with, supersedes, similar_to, implements_pattern, annotated_as, runs, builds,
  configures, maps_to, migrates, specifies, scales, and more. Each edge carries evidence, confidence, status, `file:line`, revision and attrs.
  Parallel edges with different evidence classes count as **corroboration**.
- **Refs.** Unresolved references from extraction. They are re-resolved into edges after every scan.

## Indexing (scanner.py)

1. List files with `git ls-files -co --exclude-standard` and apply include/exclude globs. Skip binary and
   oversized files, except assets: PDFs, Office files and images are read up to their own limits, and video is
   identified from its header without being read whole. YAML and JSON files are kept only when their content
   identifies them as Kubernetes, OpenAPI, Compose or CI; `.m` and `.v` files are told apart by content.
2. Hash every file. Only files whose content changed are re-extracted; deleted files are forgotten.
   Extraction, including any LLM extraction, runs before the write lock is taken. After each code file is
   extracted, a data-links pass adds SQL-in-strings, ORM mappings and migrations as `link` references.
3. Resolve refs. Python type inference covers annotated parameters, `self.x = param`, `x = Cls()`
   and `x = factory()` via return annotations, so `self.repo.find_all()` resolves statically.
   Ambiguous name matches go to ML disambiguation.
4. Classification pass: layers, library-call resources, doc and note intent. It uses the two-tier
   classifier, and results are memoized by input hash.
5. Graph ML: embeddings for changed nodes, propagation, `similar_to` edges and disambiguation.
6. Freshness: agent, handoff and runtime facts that cite changed files become `stale`.

Steps 1–2 (including any LLM extraction) run **before** the write lock is taken. Steps 3–6 run in **one
transaction**: a failure rolls back and is recorded as a failed job, so the last valid index survives. In `--llm`
mode, step 4's model calls run after that commit with no lock held, and a second short transaction applies the
cached answers. No network call ever happens while the index is locked, so other processes (the MCP server, a
second terminal) wait seconds at most. A writer that cannot get the lock within `db_busy_timeout` gets a clear
"index busy" error.

## Evidence-aware traversal (graph.py)

Edge trust = evidence weight × freshness × confidence. Corroboration from independent evidence
classes raises it.
- `impact`: reverse traversal over dependency edges with a best-first search on path trust. It
  groups results into symbols, endpoints, tests, docs and knowledge, and lists the tests to run.
- `trace`: k most-trusted paths, using Dijkstra on −log(trust). Dynamic dispatch through
  `overrides` is followed as an inferred `dispatches` step.
- `dataflow`: forward flows to sinks. A flow is labelled `static` when every step is observed.

## Phases (from the specification)

| Phase | Status |
|---|---|
| I. Structural core: Git, parsing, symbol index, dependency graph, local persistence, CLI | done (Python AST; JS/TS lexical) |
| II. Blueprint compiler: specialists, architecture, flows, API, patterns, decisions, roadmap | done (deterministic specialists + optional provider synthesis) |
| III. Semantic retrieval: hybrid search, typed traversal, evidence ranking, context compiler, revision-aware cache | done |
| IV. Agent interoperability: MCP, handoffs, schemas, provider adapters, cross-session workflows | done (56 tools; command / anthropic / openai / mcp-queue providers; `--llm` CLI; `install` for 19 assistants) |
| V. Continuous intelligence: incremental re-analysis, stale claims, impact, regression checks, benchmarks | done (`sync`, `verify`, retrieval benchmark, agent A/B harness); real-agent A/B runs not yet collected |

## MVP acceptance checklist

- [x] Scan a Git repository and record its revision (`+dirty` marks uncommitted work)
- [x] Extract symbols, imports, references and supported call relationships
- [x] Architecture, data-flow, API, design-pattern and decision reports where evidence is available
- [x] Dependency-aware implementation roadmaps with validation steps
- [x] Source references and evidence classifications on important claims
- [x] Query graph relationships and identify change impact
- [x] Core retrieval and handoff tools through MCP
- [x] Create a structured handoff and resume it from a separate agent session (covered by tests,
      MCP, CLI and the stdio transport)
- [x] Incrementally update the index after code changes
- [~] Benchmark retrieval quality, token usage and end-to-end outcomes. The retrieval benchmark is
      measured. The end-to-end harness is built and tested with a scripted agent; real-agent runs
      are still pending.

## Parsers and resolution

| Source | Extractor | Evidence |
|---|---|---|
| Python | `ast` + local type inference (annotations, `self.x = param`, constructor/factory return types) | `ast`, `static_resolved` |
| JS/TS | TypeScript compiler API via `extract/ts_bridge.js` (one batched program per scan; checker resolves call targets) | `ast`, `static_resolved` |
| Go, Rust, Java, C#, Kotlin, Ruby, PHP, C, C++, Swift, Scala, CUDA, GLSL, HLSL, Apex | tree-sitter walker (`extract/treesitter_ext.py`) | `ast` |
| 31 more (Dart … Verilog, see README) | declarative specs (`extract/treesitter_generic.py`): node types plus name getters, form readers for call-shaped definitions, lexical calls where a grammar has no call node | `ast` |
| Vue, Svelte | `<script>` blocks through the JS/TS extractor, template component references (`extract/embedded.py`) | as JS/TS |
| Dockerfile, Compose, Kubernetes, Terraform, CI, OpenAPI | `extract/infra_ext.py` (YAML via PyYAML or the built-in subset parser `yamlish.py`; HCL block parser) | `manifest` |
| SQL, Prisma, GraphQL, protobuf | `extract/schema_ext.py` | `manifest` |
| SQL strings, ORM mappings, migrations in code | `extract/data_links.py` | `lexical`, `heuristic` |
| PDF, Word, PowerPoint, images, video | `extract/assets_ext.py`; text goes through the docs extractor | `documentation` |
| Any of the above without its parser | lexical (`js_ext.py`, `generic_lexical.py`), or grounded LLM extraction in `--llm` mode | `lexical`, `llm_extraction` |

`link` references carry their own edge type and evidence and resolve by typed lookups: `ID:` (exact node),
`TABLE:` (created on demand as a shared node when no DDL defines it), `PRISMA:`, `MODULE:`, `FILE:`, `SYMBOL:`,
`LABELS:` (Kubernetes selectors to workloads) and `API:` (OpenAPI operation to code endpoint, path parameters
normalised).

Resolution order for a call: type-checker target (`ID:`), inferred receiver type (`TYPE:`/`RET:`), import
alias (Python imports, JS imports, Rust `use` paths through workspace crates), same module/class, `Type::method`
through a unique type name, then unique name match. Ambiguous names go to graph-embedding disambiguation.
Incremental scans re-resolve only references owned by changed files, files whose edges pointed into changed files,
and files whose references mention names that were added or removed.

## Dynamic behaviour

- Static detection of wiring the plain call graph misses: `getattr` with constant names, `importlib`, callbacks
  passed to registration APIs, `Depends(...)`, pytest fixtures, event emit/subscribe (Python and TS), signal
  `connect`, task-queue `.delay`, dotted-string references (DI containers, settings). These edges carry a `wiring`
  attribute and inferred (`heuristic`) evidence.
- Runtime evidence via `promapkin trace run -- <command>`: Python profiling (caller → callee, test → reached code)
  and Node V8 coverage. Runtime edges not present statically are reported as dynamic links; they are marked stale
  when the traced code changes.
- Generated files are indexed and flagged (`generated`), so readers know not to edit them.

## Infrastructure, data and assets

`infra_view.py` builds the read models behind `graph infra`, `graph schema`, `get_infrastructure`,
`get_data_model`, the architecture blueprint sections and the `infrastructure` and `er` diagrams:
- deployables with the code they run, what routes to them, what they build from and the config they set;
- cloud resources by category;
- pipelines;
- the config surface (read by code but declared nowhere, and the reverse);
- tables with keys, ORM classes, readers and writers, and migrations;
- schema types, endpoints by protocol, and API-spec coverage.

## Installers

`installer.py` writes each assistant's MCP configuration, instructions file and (where supported) hooks,
skills and commands.
- JSON is merged by key, and invalid JSON is never rewritten.
- Text files get marked blocks, and owned files carry a marker so edited copies are left alone.
- Every change is recorded in `.promapkin/installs.json`, and `uninstall` replays it in reverse.

## Workflow modules

- **`prs.py`: pull-request dashboard.**
  - Sources: GitHub REST through `gh`, `GITHUB_TOKEN`/`GH_TOKEN` or `GH_HOST`, or local branches against the
    default branch.
  - Changed hunks are mapped to the symbols whose line ranges they cover, then to modules and communities.
  - `impact()` gives the blast radius and the tests to run.
  - Conflict risk weights shared files, then shared symbols, then shared modules and communities.
  - `git worktree list --porcelain` maps branches to checkouts.
- **`autoupdate.py`: keeping the index fresh.**
  - Git hook blocks between `# >>> promapkin` and `# <<< promapkin` run a detached background scan.
  - The polling watcher compares `(mtime, size)` of the scanner's candidate files and debounces before an
    incremental scan.
- **`sources.py`: `promapkin add`.**
  - Each source type has a fetcher; output lands as Markdown or text under `.promapkin/sources/`, which the
    scanner indexes as documents.
  - The SSRF guard resolves the host and refuses private, loopback, link-local and reserved addresses.
- **`exports.py`: graph and documentation export.**
  - JSON and GraphML export.
  - Multi-repo merge: ids become `namespace::id` and cross-repo links are derived from endpoints, specs,
    tables, messages and images.
  - The Markdown wiki.
  - Mermaid call-flow diagrams.
- **`insights.hyperedges`** builds n-ary groups:
  - co-change sets mined from `git log`, with support ≥ 3 commits;
  - data-flow paths;
  - pattern participants;
  - communities.

  `semantic_surprises` uses embedding `similar_to` edges between communities that have no structural edge.
- **Parallel extraction.** `scanner._extract_parallel` forks worker processes for scans of ≥ 200 files when no
  LLM is attached. Extraction is pure, so the results are identical to a sequential scan; a test checks this.

## Known limitations

- Tree-sitter and TypeScript extraction are optional installs; without them the lexical fallback is used (weighted
  lower) unless `--llm` is on. Kotlin/Swift/Scala inheritance and call extraction is shallower than Python/TS,
  and the 31 spec-driven languages extract declarations, imports and calls but no type information.
- Infrastructure parsing covers the common shapes of each format. Helm values are not rendered (template
  expressions are blanked), Terraform expressions are not evaluated, and Kustomize overlays are not merged.
- PDF text without pypdf comes only from simple encodings; scanned documents need OCR or `--llm`. Video content
  is not analysed.
- Cross-file resolution for tree-sitter languages relies on imports, crate paths and unique names; there is no
  per-language type checker for them (Rust trait-object dispatch, Java overloads and Go interface satisfaction are
  approximated through `implements`/`overrides` edges and name matching, tagged accordingly).
- Runtime tracing covers Python call edges and Node function coverage; other runtimes are not traced.
- The built-in hashing embedder is mostly lexical; semantic recall on vocabulary-gap queries needs a downloaded
  neural model or `--llm`.
