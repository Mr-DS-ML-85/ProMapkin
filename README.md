<p align="center">
  <img src="docs/assets/banner.svg" alt="Promapkin: evidence-aware codebase intelligence for agents and developers" width="100%">
</p>

<p align="center">
  <b>A persistent, evidence-aware model of your codebase.</b><br>
  Blueprints, a semantic code graph, task context and agent handoffs, over CLI, MCP and optional LLMs.
</p>

---

Promapkin turns a repository and its documentation into an evidence-backed architectural blueprint, a
queryable code graph, and reusable task context. Coding agents share all of it through **MCP**. Developers use
the same engine from the **CLI**, and can add `--llm` to any command to bring Anthropic or OpenAI models into the
analysis.

It rests on one idea: a coding agent should not have to rediscover a project's architecture every time it starts
a task or picks up another agent's work.

- **Evidence everywhere.** Every node and edge records how it was obtained, along with its revision, source
  location and freshness. Evidence kinds include `ast`, `static_resolved`, `lexical`, `manifest`,
  `documentation`, `runtime`, `heuristic`, `ml_classifier`, `llm_extraction`, `llm_inference`, `name_match`,
  `agent_inference` and `handoff`. Traversal and retrieval rank results by that trust.
- **Real parsers first.**
  - Python uses the AST.
  - JS/TS uses the TypeScript compiler, including type-checked call resolution.
  - Go, Rust, Java, C#, Kotlin, Ruby, PHP, C, C++, Swift and Scala use tree-sitter.
  - Every language has a lexical fallback, so nothing is invisible.
- **Static + dynamic.**
  - Reflection, dynamic imports, callbacks, DI, fixtures and event wiring are detected statically.
  - Opt-in runtime tracing (`trace run -- pytest`) adds the call edges only execution shows.
- **Local by default.** SQLite, zero required dependencies, no repository code executed by scanning. Models,
  parsers and LLMs are opt-in.

## Quick start

```bash
pip install -e .                          # core: zero dependencies
pip install -e '.[all]'                   # + LLM SDKs, tree-sitter, embeddings, numpy
promapkin parsers install                 # optional: TypeScript compiler + tree-sitter grammars
cd /path/to/repo
promapkin init && promapkin scan          # incremental, transactional
promapkin blueprint generate --goal "add rate limiting to the API"
promapkin graph html                      # interactive, offline graph explorer
promapkin doctor                          # what is enabled, what is missing
```

Measured on real repositories (single core, no neural models; numpy installed):

| Repository | Files | Scan | Blueprint |
|---|---|---|---|
| fastapi/fastapi (Python) | 2,886 | 36 s (was 310 s in v0.1) | 3 s |
| clap-rs/clap (Rust, tree-sitter) | 419 | 6 s | — |
| clap-rs/clap (Rust, lexical fallback) | 419 | 17 s | — |

Rescans only re-extract changed files and re-resolve the references those changes can affect.

## What you get

`promapkin blueprint generate` writes `.promapkin/blueprints/`:

| File | Contents |
|---|---|
| `00-reconnaissance.md` | languages, entry points, conventions, missing context, security notices |
| `01-architecture.md` | subsystems, layers, dependencies, layer violations, integrations, style, semantic components |
| `02-dataflow.md` | entry point → sink flows (DB, HTTP, files, queues, config), static vs inferred steps |
| `03-api-atlas.md` | endpoints, handlers, params, auth boundaries, public interfaces, packages, hotspots |
| `04-patterns.md` | Strategy, Adapter, Repository, DI, Factory… with roles, confidence and classifier agreement |
| `05-decisions.md` | ADRs, rationale stated in code, inferred decisions (rationale "unknown" unless documented) |
| `06-risks.md` | risky symbols (fan-in, complexity, untested, writes, churn), structural risks, TODOs |
| `07-verification.md` | documentation claims checked against code; contradictions; stale knowledge |
| `08-roadmap.md` | dependency-ordered plan with prerequisites, tests, migrations, risks |
| `diagrams/*.mmd`, `blueprint.json`, `AGENT_GUIDE.md`, `graph.html` | Mermaid sources, machine-readable blueprint, agent orientation, graph explorer |

Each claim is tagged, for example `` `[observed · ast · 1.00]` (searchapp/api/routes.py:21) ``.

## CLI

```bash
promapkin scan | sync | verify | status | doctor
promapkin blueprint generate [--goal "..."] | architecture | dataflow | api | patterns | decisions | risks | roadmap "goal"

promapkin graph query "Command"                       # hybrid search: exact symbols, BM25, embeddings, graph
promapkin graph impact "APIRouter.include_router"     # ranked blast radius + tests to run
promapkin graph trace --from FastAPI --to APIRouter   # flow → structural → undirected, one-line summary
promapkin graph dataflow "api:POST /items"            # entry point → sinks
promapkin graph hubs | communities | surprises        # god nodes, module communities, rare cross-community links
promapkin graph html                                  # self-contained interactive explorer
promapkin graph deps | node | neighbors | similar | source | diagram | map | relate | finding

promapkin context build "optimize vector search" --budget 3000
promapkin handoff create --task ... --finding "claim|path:line|verified" --constraint ... --next ...
promapkin handoff resume <id> --claim | update | branch | list | show

promapkin trace run -- python -m pytest -q            # runtime evidence from YOUR command
promapkin parsers status | install [typescript|tree-sitter]
promapkin models status | download | classify | feedback
promapkin auth set anthropic|openai | status | remove
promapkin mcp serve | inspect | config
promapkin bench retrieval [--spec ...] | agents --spec ...
```

Every command takes `--json`, `--root` and `--agent <name>` (provenance).

## LLM mode: `--llm` on any command

```bash
promapkin auth set anthropic                          # or ANTHROPIC_API_KEY / OPENAI_API_KEY
promapkin --llm graph impact "APIRouter"              # result + plain-language explanation
promapkin --llm scan                                  # LLM replaces regex/heuristic interpretation
promapkin --llm openai --base-url http://localhost:11434/v1 --model qwen2.5-coder blueprint generate
promapkin ask "what happens between the HTTP route and the database?"   # LLM agent over Promapkin's tools
promapkin --llm review                                # review uncommitted changes against the graph
```

`ask` lets the model pick Promapkin's own read-only tools, then answers with sources and keeps observed facts
apart from inferred ones. A real run against a local OpenAI-compatible model (output trimmed):

```text
$ promapkin --llm openai --base-url http://localhost:8080/v1 ask "can you tell me the fragmented parts of the code"
→ get_surprising_connections {"limit": 30}
→ get_communities {}
→ get_architecture_map {}
→ get_risk_register {}
→ get_design_patterns {}
→ get_project_overview {}
→ search_graph {"query": "duplicate solver implementation repeated logic across modules", "limit": 15}
→ get_node {"node": "module:src", "limit": 10}
→ get_index_status {}

## 1. Duplicate / competing solver implementations (the biggest fragment)
There are at least three separate implementations of the same "detect challenge type → route to a solver"
idea ... `src/pow/solver.py:489` (`PoWSolverFactory`) and `page-agent/solvers/universal.py` each define their
own `PoWSolution` ...

## 2. An empty root package that everything imports
## 3. `capify` is split across two codebases
## 4. Low-cohesion communities          (table: community · cohesion · members)
## 5. Layer violations (24 total)
## 6. page-agent is itself split
## 7. Test-coverage fragmentation

Observed (AST-parsed): the duplicate solver symbols, the import edges, the 24 layer violations ...
Inferred (ML classifier / heuristic): the community/cohesion numbers ...

[openai/<local model>] 6 call(s), 0 cached · 78,534 in / 5,378 out tokens · 0 fallback(s) · 0 secret(s) redacted
```

With `--llm`:
- Files without a real parser are extracted by the model. Every reported symbol and call must be found at its
  line in the source, or it is dropped.
- Docs are mined for claims, decisions and requirements, each with a verbatim quote.
- All classification is done by the model: layers, design patterns, resource calls, doc and comment intent.
- Search gets query expansion and reranking.
- Blueprint specialists get an LLM synthesis, and roadmaps get refined.
- Query commands print an explanation of the result.

Safeguards:
- Official `anthropic` and `openai` SDKs. OpenAI-compatible base URLs (vLLM, Ollama, LM Studio, gateways) work too.
- Strict JSON schemas, validated on every reply. For proxies that ignore them (for example OmniRoute's `ds-web`),
  set `openai_structured_output: "prompt"`. See [docs/LLM.md](docs/LLM.md).
- No LLM call ever runs while the index is locked, so the MCP server and the CLI can work side by side.
- A response cache, so unchanged inputs are never re-sent.
- A per-command call budget.
- Secret redaction and an egress log.
- Deterministic fallback on any failure.

See [docs/LLM.md](docs/LLM.md).

## MCP for coding agents

```bash
promapkin mcp config                                  # paste into your agent's MCP settings
claude mcp add promapkin -- python -m promapkin --root /path/to/repo mcp serve
```

There are 52 tools. They cover:
- **Index:** status, scan, sync, verify, overview, toolchain.
- **Design engine:** blueprint generation and sections, architecture, patterns, decisions, risks, implementation
  plans, specialist runs, diagrams.
- **Graph read:** search, node, neighbours, structured query, trace, data flow, dependencies, impact, source,
  similar, hubs, communities, surprising connections, HTML export.
- **Graph write**, for agents that build knowledge: findings, nodes, relationships, annotations, supersede, maps.
- **Retrieval:** budgeted task context and claim verification.
- **Handoffs:** create, update, resume, branch, list, get.
- **Specialist task queue:** for external agents.
- **ML:** classification and feedback, plus benchmarks and runtime-trace ingestion.

Writes are validated, append-only and attributed to the calling agent. Static facts cannot be overwritten, and
results are size-bounded. Full reference: [docs/MCP_TOOLS.md](docs/MCP_TOOLS.md).

## Languages

| Language | With optional parser | Fallback |
|---|---|---|
| Python | AST, light type inference, reflection/DI/fixture/event wiring | — |
| JavaScript / TypeScript | TypeScript compiler API: type-checked call targets, interfaces, routes, events | lexical |
| Go, Rust, Java, C#, Kotlin, Ruby, PHP, C, C++, Swift, Scala | tree-sitter: declarations, impl/receiver methods, inheritance, imports, calls | lexical |
| Markdown, ADRs, manifests (pip, npm, Cargo, Go) | sections, decisions, claims, dependencies | — |
| Anything without a parser, in `--llm` mode | grounded LLM extraction | — |

Import resolution understands Python packages, relative and `tsconfig` module paths, Rust `crate::`/`super::` and
workspace crates (Cargo.toml), and package-path languages.

## Machine learning

Every classification runs rules and a model side by side and records whether they agree. Classification covers
layers, design patterns, resource calls, doc intent and comment intent. Graph ML adds:
- node embeddings propagated over the graph;
- `similar_to` edges;
- disambiguation of ambiguous calls;
- semantic communities.

Small neural embedding models (fastembed, sentence-transformers) are downloaded only with consent. Otherwise a
built-in pure-Python model is used. `ml_backend: heuristic` switches everything to rules with static confidence.

## Handoffs

A handoff is a compact, versioned index into project knowledge: findings with sources, constraints, symbols,
decisions, next action and validation. It supports optimistic concurrency, branching and conflict recording.
`resume` reports which findings went stale because their code changed, and always returns the constraints, even
under a tight token budget.

## Benchmarks

- `promapkin bench retrieval` measures recall@k, MRR and token reduction against naive exploration, with an
  ML-vs-heuristic ablation.
- `promapkin bench agents` runs the same coding agent with and without Promapkin on identical tasks. It records
  tokens, completion, regressions, retrieval accuracy and setup time.

See [docs/BENCHMARK.md](docs/BENCHMARK.md).

## Security

- Scanning parses and never executes repository code.
- File access is confined to the repository.
- Repository text is passed to agents and LLMs as untrusted data.
- Downloads and remote calls are opt-in and logged.
- Credentials live outside the repository.

See [SECURITY.md](SECURITY.md).

## Development

```bash
python -m unittest discover -s tests -t .             # core suite, no network
pip install -e '.[all]' && python -m unittest discover -s tests -t .   # + parsers and LLM SDK paths
```

CI runs Python 3.10–3.13, with and without extras. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for
internals and [CHANGELOG.md](CHANGELOG.md) for release notes.
