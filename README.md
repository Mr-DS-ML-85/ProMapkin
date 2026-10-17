# Promapkin

**Agentic codebase intelligence and knowledge transfer.** Promapkin turns a repository and its
documentation into an evidence-backed architectural blueprint, a semantic knowledge graph, and
reusable task context. Coding agents share all of it through a CLI and an MCP server.

The core idea: a coding agent should not have to rediscover a project's architecture every time
it starts a task or takes over from another agent.

- **Analyze before generating.** Deterministic extraction (ASTs, imports, calls, routes, config,
  docs, ADRs) comes first. Specialist agents, ML classifiers and optional LLM providers interpret
  those facts afterwards; they never replace them.
- **Evidence everywhere.** Every node and edge records how it was obtained. Possible origins are
  `ast`, `static_resolved`, `lexical`, `manifest`, `documentation`, `heuristic`, `ml_classifier`,
  `name_match`, `agent_inference`, `handoff` and `runtime`. Each also records the revision, source
  location and freshness. Traversal and retrieval weight results by that trust.
- **Persist knowledge, not conversations.** Findings, decisions, constraints and handoffs are
  versioned records linked into the graph.
- **Retrieve instead of replay.** A budget-aware context compiler gives each agent the smallest
  sufficient package. It never drops a constraint to fit a budget.
- **Local by default.** It uses SQLite, has no mandatory services and never executes repository
  code. Model downloads and remote providers are opt-in.

Zero required dependencies (Python ≥ 3.10). The optional ML/DL extras add small embedding models.

---

## Quick start

```bash
pip install -e .                       # or: python -m promapkin ...
cd /path/to/your/repo
promapkin init
promapkin scan                         # incremental; asks before downloading any model
promapkin blueprint generate --goal "add pagination to search results"
promapkin status
```

Blueprints land in `.promapkin/blueprints/`:

| File | Contents |
|---|---|
| `00-reconnaissance.md` | languages, entry points, conventions, missing context, security notices |
| `01-architecture.md` | subsystems, layers (rules + ML), dependencies, violations, integrations, style, semantic components |
| `02-dataflow.md` | entry point → sink flows (database, HTTP, files, queues, config), static vs inferred |
| `03-api-atlas.md` | endpoints, handlers, params, auth boundaries, public interfaces, packages, hotspots |
| `04-patterns.md` | design patterns (Strategy, Adapter, Repository, DI, Factory, …) with roles and classifier agreement |
| `05-decisions.md` | ADRs, rationale stated in code, inferred decisions with rationale *unknown* unless documented |
| `06-risks.md` | risky symbols (fan-in, complexity, untested, writes, churn), structural risks, TODOs |
| `07-verification.md` | documentation claims checked against code; contradictions; stale knowledge |
| `08-roadmap.md` | dependency-ordered implementation plan with prerequisites, tests, migrations, risks |
| `diagrams/*.mmd`, `blueprint.json`, `AGENT_GUIDE.md` | Mermaid sources, machine-readable blueprint, agent orientation |

Every claim is tagged, for example `` `[observed · ast · 1.00]` (searchapp/api/routes.py:21) ``.

## Connect a coding agent (MCP)

```bash
promapkin mcp config          # prints a ready-to-paste mcpServers block
promapkin mcp inspect         # lists the 46 tools, prompts and resources
```

For Claude Code: `claude mcp add promapkin -- python -m promapkin --root /path/to/repo mcp serve`.

The MCP surface covers the whole engine, including tools that let agents **build** knowledge:

| Area | Tools |
|---|---|
| Index | `get_index_status`, `scan_repository`, `sync_repository`, `verify_index`, `get_project_overview` |
| Blueprint & design engine | `generate_blueprint`, `get_blueprint`, `get_architecture_map`, `get_design_patterns`, `get_design_decisions`, `get_risk_register`, `plan_implementation`, `run_specialist`, `render_diagram` |
| Graph (read) | `search_graph`, `get_node`, `get_neighbors`, `query_graph`, `trace_path`, `trace_dataflow`, `trace_dependencies`, `analyze_impact`, `get_source`, `find_similar` |
| Graph (write) | `add_knowledge_node`, `record_finding`, `add_relationship`, `annotate_node`, `supersede_fact`, `create_map`, `get_map`, `list_maps` |
| Retrieval | `build_task_context`, `verify_claims` |
| Handoffs | `create_handoff`, `update_handoff`, `resume_handoff`, `branch_handoff`, `list_handoffs`, `get_handoff` |
| Agents as specialists | `get_analysis_tasks`, `submit_analysis` |
| ML | `get_ml_status`, `classify`, `submit_classification_feedback` |
| Benchmark | `run_benchmark` |

Full argument reference: [docs/MCP_TOOLS.md](docs/MCP_TOOLS.md). The server also exposes blueprint
reports and handoffs as MCP resources, and workflow prompts (`investigate`,
`implement_from_handoff`, `review_change`, `map_subsystem`).

Rules for shared state:
- Mutating tools validate their inputs and record the calling agent.
- They append rather than overwrite.
- Static facts cannot be overwritten by agents.
- Uncited findings are stored as `unsupported` with capped confidence.
- `update_handoff` takes `expected_version` for optimistic concurrency.
- Results are size-bounded. Repository text is marked as untrusted data.

## CLI

```bash
promapkin init | scan [--full] | sync | verify [--strict] | status

promapkin blueprint generate [--sections ...] [--goal "..."]
promapkin blueprint architecture|dataflow|api|patterns|decisions|risks|verification|recon
promapkin blueprint roadmap "goal"

promapkin graph query "VectorIndex"
promapkin graph impact "CandidateSelector::select"
promapkin graph trace --from api:search --to resource:database
promapkin graph dataflow "api:POST /documents"
promapkin graph deps [module] | node <ref> | neighbors <ref> | similar <ref> | source <ref>
promapkin graph diagram architecture|dataflow|patterns|roadmap|neighborhood|dependencies|map [--focus ..]
promapkin graph finding "claim" --cite path:line      # agent-written knowledge
promapkin graph relate <src> depends_on <dst> --rationale "..."
promapkin graph map "Checkout flow" <node> <node> ...

promapkin context build "optimize vector search" --budget 3000 [--constraint ..] [--validation ..]

promapkin handoff create --task "..." --finding "claim|path:line|verified|benchmark" \
    --constraint "preserve existing API" --symbol VectorIndex.search --next "implement" --validation "run tests"
promapkin handoff resume <id> [--claim] | update <id> ... [--expected-version N] | branch <id> "subtask" | list | show <id>

promapkin agent tasks --sections patterns decisions       # queue/claim specialist contracts
promapkin agent submit <task-id> result.json

promapkin models status | download [--backend fastembed|sentence-transformers] | remove
promapkin models classify layer "app.repositories.users" | feedback <task> <label> "<text>"
promapkin config show | set <key> <value>

promapkin bench retrieval [--spec examples/bench/promapkin-self.json]
promapkin bench agents --spec agent-bench.json            # same agent with vs without Promapkin
promapkin mcp serve | inspect | config
```

Every command accepts `--json`, `--root` and `--agent <name>` (the agent name is recorded for
provenance).

## Machine learning: ML/DL first, heuristics as fallback

ML is used throughout. Every ML decision runs **two classifiers**, deterministic rules and a
statistical/neural model, and records whether they agree. Agreement raises confidence; on
disagreement the stronger answer wins and the alternative is kept.

| Where | ML / DL | Without ML (`ml_backend=heuristic`) |
|---|---|---|
| Layer classification of modules | rules + embedding prototype / naive Bayes | path rules, static confidence |
| Design-pattern detection | rules + classifier over structural feature descriptions | rules, static confidence |
| Resource classification of library calls (DB/HTTP/queue/cache/fs) | classifier for calls the rules don't cover | rules only |
| Doc-section and code-comment intent | rules + classifier | rules |
| Graph: node embeddings | text embeddings propagated over the code graph (parameter-free GNN-style message passing) | — |
| Graph: `similar_to` edges | cosine over graph embeddings (`ml_classifier` evidence) | identifier-token Jaccard (`heuristic`, confidence 0.4) |
| Graph: ambiguous call resolution | nearest candidate in graph-embedding space | the only candidate in an imported module (confidence 0.45) |
| Architecture: semantic components | label propagation over imports + embedding similarity | directory grouping |
| Retrieval | dense + lexical + symbol + graph fusion | lexical + symbol + graph |

**Model tiers.**
1. **Neural** (`fastembed` ONNX `BAAI/bge-small-en-v1.5`, or `sentence-transformers`
   `all-MiniLM-L6-v2`) is used when installed and cached.
2. If no model is cached, Promapkin **asks before downloading**:
   - `ml_mode=ask` (default) prompts in an interactive terminal; your answer is remembered.
   - `ml_mode=auto` downloads without asking.
   - `ml_mode=never` never downloads.
   - The MCP server and other non-interactive runs never prompt. They fall back and report that a
     download is available.
3. If you decline, Promapkin uses the rules plus a **built-in model**. That model is
   feature-hashed embeddings with naive Bayes, pure Python, with nothing downloaded.
4. `promapkin config set ml_backend heuristic` turns off every statistical model. You get rules
   with static confidence only.

Downloads fetch model weights only; source code is never uploaded. Install extras with
`pip install -e '.[ml]'` (fastembed) or `'.[dl]'` (sentence-transformers). Teach the classifiers
with `promapkin models feedback` or the MCP `submit_classification_feedback` tool. Examples are
stored in `.promapkin/training/`.

## Agent handoffs

A handoff is a compact, versioned index into project knowledge. It is not a transcript.

```yaml
handoff_version: 1
task: optimize vector search
status: investigation_complete
findings:
  - claim: candidate selection is a bottleneck
    sources: [searchapp/domain/ranking.py:38]
    evidence: benchmark
    status: verified
constraints: [preserve existing API, maintain recall threshold]
relevant_context: {symbols: [SearchService.search, select_candidates], dependencies: [IndexRepository]}
next_action: implement and benchmark
validation: [run targeted tests, compare latency, verify recall]
```

Handoffs are projected into the graph. Findings and constraints become nodes linked to the
symbols they affect, so retrieval and impact analysis see them. `resume_handoff` returns:
- the handoff itself;
- a **freshness report**: whether the revision changed, which referenced symbols changed, and
  which verified findings now need revalidation;
- lineage and branches;
- a budgeted context package that always contains the constraints and validation steps.

Handoffs support branching, concurrent agents (optimistic versions), conflicting findings
(recorded rather than overwritten), abandonment and superseding.

## External agents as specialists

Set `provider` to:
- `none` (default): deterministic specialists only.
- `command`: any agent CLI that reads a task contract JSON on stdin and writes a result on stdout.
- `anthropic`: requires `allow_remote=true` and `ANTHROPIC_API_KEY`; every request is logged to
  `.promapkin/egress.log`.
- `mcp`: queues contracts for MCP agents.

A task contract carries:
- the objective and bounded scope;
- the revision;
- extracted facts;
- allowed tools;
- the output schema;
- evidence requirements;
- budget, validation and retry limits.

Results are validated before merging as `agent_inference`. Citations must resolve to existing
nodes or `path:line`.

## Benchmarks

`promapkin bench retrieval` measures:
- recall@k and MRR;
- context hit rate;
- tokens versus a naive "grep and open files" exploration baseline;
- latency;
- an ML-vs-heuristic ablation.

`promapkin bench agents --spec …` runs **the same coding agent** on identical tasks in fresh copies
of the repository, once without and once with Promapkin context and MCP. It records token
consumption, task completion, regressions (test failures that appear after the change), retrieval
accuracy and setup overhead. See [docs/BENCHMARK.md](docs/BENCHMARK.md) for the methodology and
the numbers measured so far.

## Security model

- Repository code is never executed. Analysis is parse-only (Python `ast`, a lexical JS/TS scanner).
- All file access goes through a path guard confined to the repository root, with `.git` excluded.
- Documentation, comments and code are treated as untrusted. Text that looks like an instruction
  (for example "ignore previous instructions", `curl | sh`) is flagged in reports and never treated
  as a directive.
- Remote providers and model downloads are opt-in. Egress is logged.

## Project layout

```
promapkin/
  extract/      python_ext (AST + type inference), js_ext (lexical), docs_ext (sections, ADRs, claims), manifest_ext
  scanner.py    incremental, transactional indexing + cross-file resolver + classification/graph-ML passes
  graph.py      evidence-aware traversal: resolve, neighbors, impact, trace (Dijkstra on -log trust), dataflow, deps
  ml/           embedders (hash / fastembed / sentence-transformers), two-tier classifiers, graph ML, model consent
  agents/       specialists, task contract + validation, provider adapters, orchestrator
  retrieval.py  hybrid search + budget-aware context compiler
  handoff.py    versioned handoffs, concurrency, conflicts, freshness, graph projection
  knowledge.py  agent-writable knowledge (validated, append-only)
  blueprint.py  Markdown / JSON / Mermaid compiler
  service.py    one engine behind both interfaces
  mcp_server.py MCP (JSON-RPC over stdio): tools, resources, prompts
  cli.py        command line
  bench.py      retrieval benchmark + agent A/B harness
```

Internals, phase status and the MVP acceptance checklist are in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Development

```bash
python -m unittest discover -s tests -t .     # 86 tests, no network, no models needed
```

`tests/fixtures/sample_repo` is a small layered search service: FastAPI routes, a service, a
Strategy-based ranker, a SQLite repository, an HTTP adapter, a TypeScript client, an ADR and a
deliberately false documentation claim. The tests use it end to end.
