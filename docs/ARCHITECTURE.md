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
| IV. Agent interoperability: MCP, handoffs, schemas, provider adapters, cross-session workflows | done (46 tools; command / anthropic / mcp-queue providers) |
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

## Known limitations

- JavaScript/TypeScript extraction is lexical: no full parser, no type inference. Its facts carry
  `lexical` evidence and are weighted accordingly. Other languages are indexed only through docs
  and manifests.
- Static analysis cannot see reflection, generated code or runtime wiring. Such paths appear only
  as `inferred` (dispatch, name match, ML) or are missing. Runtime evidence can be recorded by
  agents (`evidence: runtime`) but is never collected by executing code.
- Neural embeddings are optional. The built-in hashing embedder is mostly lexical, so semantic
  recall on vocabulary-gap queries needs a downloaded model.
- References are re-resolved in full on every scan. That is cheap at tested sizes, but
  incremental resolution would be the next optimization for very large repositories.
