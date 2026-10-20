# Promapkin internals

```
Interface layer         cli.py · mcp_server.py (tools, resources, prompts) · JSON outputs
        │                         └──── service.py (one engine, two interfaces)
Core orchestration      agents/orchestrator.py (plan → contracts → specialists → validate → merge)
        │
Code intelligence       extract/* (AST, lexical, docs/ADRs, manifests) · scanner.py (resolver, type inference)
Agent runtime           agents/specialists.py · agents/contract.py · agents/providers.py
ML tier                 ml/embedders.py · ml/classify.py (two-tier) · ml/graph_ml.py · ml/models.py (consent)
Knowledge graph         store.py (nodes/edges/refs/FTS/embeddings) · graph.py · knowledge.py · evidence.py
Retrieval engine        retrieval.py (hybrid search, graph expansion, budget-aware compiler)
Persistence/artifacts   .promapkin/index.db (SQLite WAL) · blueprints/ · handoffs/ · maps/ · training/
```

## Data model

- **Nodes.** Kinds: repository, directory, file, module, class, function, method, test, endpoint,
  package, resource, config, document, doc_section, decision, code_note, manifest, entrypoint,
  pattern, finding, concept, constraint, handoff, roadmap_step, map, and more. Each node carries:
  - `evidence`, `confidence` and `status` (verified, stale, unsupported, superseded);
  - `revision` and `verified_at`;
  - `owner_file`, used for incremental updates;
  - `attrs`, a JSON blob.
- **Edges.** Typed: contains, defines, imports, calls, instantiates, inherits, overrides,
  references, depends_on, routes_to, reads, writes, invokes, verifies, documents, constrains,
  affects, supports, conflicts_with, supersedes, similar_to, implements_pattern, annotated_as, and
  more. Each edge carries evidence, confidence, status, `file:line`, revision and attrs.
  Parallel edges with different evidence classes count as **corroboration**.
- **Refs.** Unresolved references from extraction. They are re-resolved into edges after every scan.

## Indexing (scanner.py)

1. List files with `git ls-files -co --exclude-standard`, apply include/exclude globs, and skip
   binary and oversized files.
2. Hash every file. Only files whose content changed are re-extracted; deleted files are forgotten.
3. Resolve refs. Python type inference covers annotated parameters, `self.x = param`, `x = Cls()`
   and `x = factory()` via return annotations, so `self.repo.find_all()` resolves statically.
   Ambiguous name matches go to ML disambiguation.
4. Classification pass: layers, library-call resources, doc and note intent. It uses the two-tier
   classifier, and results are memoized by input hash.
5. Graph ML: embeddings for changed nodes, propagation, `similar_to` edges and disambiguation.
6. Freshness: agent, handoff and runtime facts that cite changed files become `stale`.

All of this runs in **one transaction**. A failure rolls back and is recorded as a failed job, so
the last valid index survives.

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
| IV. Agent interoperability: MCP, handoffs, schemas, provider adapters, cross-session workflows | done (52 tools; command / anthropic / openai / mcp-queue providers; `--llm` CLI) |
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
| Go, Rust, Java, C#, Kotlin, Ruby, PHP, C, C++, Swift, Scala | tree-sitter (`extract/treesitter_ext.py`) | `ast` |
| Any of the above without its parser | lexical (`js_ext.py`, `generic_lexical.py`), or grounded LLM extraction in `--llm` mode | `lexical`, `llm_extraction` |

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

## Known limitations

- Tree-sitter and TypeScript extraction are optional installs; without them the lexical fallback is used (weighted
  lower) unless `--llm` is on. Kotlin/Swift/Scala inheritance and call extraction is shallower than Python/TS.
- Cross-file resolution for tree-sitter languages relies on imports, crate paths and unique names; there is no
  per-language type checker for them (Rust trait-object dispatch, Java overloads and Go interface satisfaction are
  approximated through `implements`/`overrides` edges and name matching, tagged accordingly).
- Runtime tracing covers Python call edges and Node function coverage; other runtimes are not traced.
- The built-in hashing embedder is mostly lexical; semantic recall on vocabulary-gap queries needs a downloaded
  neural model or `--llm`.
