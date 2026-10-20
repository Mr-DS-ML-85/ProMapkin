# MCP tool reference

Generated from `promapkin/mcp_server.py` by `scripts/gen_mcp_docs.py` (`promapkin mcp inspect --json` prints the same data with full schemas).

R = read-only, W = writes shared knowledge or artifacts (validated, append-only, provenance recorded).

56 tools.

| Tool | | Arguments | Description |
|---|---|---|---|
| `get_index_status` | R | — | Revision, freshness (pending file changes), counts, handoffs, queued agent tasks and ML tier of the Promapkin index. |
| `get_session_brief` | R | max_chars | Compact orientation for the start of a session: index freshness, languages, architecture, subsystems, entry points, infrastructure, most depended-on symbols, open handoffs and their constraints (about 1k tokens). |
| `scan_repository` | W | full | Index the repository (incremental by default; full=true re-extracts everything). Never executes repository code. |
| `sync_repository` | W | — | Incremental re-analysis: rescan changed files and re-run only the affected blueprint specialists. |
| `verify_index` | W | — | Stale-claim detection: index freshness, agent/handoff knowledge whose sources changed, documentation contradictions, dangling references, integrity. |
| `get_project_overview` | R | — | Compact architectural summary: styles, subsystems, layers, entry points, integrations, semantic components, missing context. |
| `generate_blueprint` | W | sections, goal, write | Run specialist analysis and write blueprint artifacts (Markdown, JSON, Mermaid). sections ⊆ ['reconnaissance', 'architecture', 'dataflow', 'api', 'patterns', 'decisions', 'risks', 'verification', 'roadmap']; goal drives the roadmap. |
| `get_blueprint` | R | **section**, format, refresh | Retrieve one blueprint section (json or markdown); regenerated if stale. |
| `get_architecture_map` | R | — | Subsystems, layer classification (rules+ML), dependencies between subsystems, layer violations, integrations, styles, semantic components. |
| `get_design_patterns` | R | — | Design patterns detected by the two-tier classifier (rules + ML/DL) with roles, confidence and classifier agreement. |
| `get_design_decisions` | R | — | Documented decisions (ADRs, code rationale) and inferred decisions with evidence; unknown rationale is labelled unknown. |
| `get_risk_register` | R | — | Risky symbols (fan-in, complexity, untested, writes, churn), structural risks, open TODO/warning notes. |
| `plan_implementation` | W | **goal** | Design engine: dependency-aware implementation roadmap for a goal with prerequisites, affected components, constraints, test plan, migration steps and risk assessment. |
| `run_specialist` | W | **specialist**, goal | Run one specialist agent (reconnaissance, architecture, dataflow, api, patterns, decisions, risks, verification, roadmap) over its bounded scope and return its structured section. |
| `get_infrastructure` | R | — | Deployables (containers, Compose services, Kubernetes workloads) with the code they run, what routes to them and the config they set; ingress; Terraform resources by category with their dependencies; CI pipelines; and the configuration surface (keys read by code but never declared, and the reverse). |
| `list_assets` | R | media, limit | PDFs, Word/PowerPoint files, images and videos in the repository: pages, dimensions, duration, extracted text availability, image descriptions (--llm), which docs reference them and which code they mention. Their text is searchable with search_graph. |
| `get_data_model` | R | — | Tables and views (columns, keys, foreign keys), ORM classes mapped to them, the symbols that read and write each table, migrations, GraphQL/protobuf/OpenAPI schema types, endpoints by protocol, and API-spec coverage (spec operations without code, code endpoints missing from the spec). |
| `render_diagram` | R | **kind**, focus, depth | Mermaid source for architecture / dataflow / patterns / roadmap / neighborhood (focus) / dependencies / map (focus=map id) / infrastructure / er (entity-relationship). Dotted edges = inferred evidence. |
| `search_graph` | R | **query**, limit, kinds | Hybrid search (symbol lookup + BM25 + semantic embeddings + graph) for symbols, endpoints, docs, decisions, findings. |
| `get_node` | R | **node**, limit | One graph node with provenance and its incoming/outgoing relationships, each with evidence class and trust. |
| `get_neighbors` | R | **node**, direction, types, depth, min_trust, limit | Neighbourhood of a node, filtered by relationship types and minimum evidence trust. |
| `query_graph` | R | kinds, name_pattern, edge_types, evidence, min_trust, status, file_prefix, limit | Structured query: nodes by kind / name pattern (* wildcard) / evidence / status / file prefix, optionally with edges of given types above a trust threshold. |
| `trace_path` | R | **source**, **target**, max_depth, k, types, mode | Most-trusted paths between two nodes with the evidence of every step and a one-line summary. mode auto tries flow edges, then structural edges, then undirected relations. |
| `get_hubs` | R | limit | Most-connected code symbols ('god nodes') by distinct neighbours over code edges. |
| `get_communities` | R | — | Communities of modules (ML: imports + embedding similarity; heuristic: directories). |
| `get_surprising_connections` | R | limit | Rare observed edges between otherwise separate communities (name matches and inferred links excluded, so no false cross-script links). |
| `export_graph_html` | W | path, max_nodes | Write a self-contained interactive HTML graph explorer (works offline). |
| `trace_dataflow` | R | **source**, max_depth, limit | Data paths from an entry point/endpoint/function to sinks (database, HTTP, files, queues, config) labelled static vs inferred. |
| `trace_dependencies` | R | module, depth | Module dependency paths, dependents, external packages and import cycles (whole repo if module omitted). |
| `analyze_impact` | R | **target**, depth, min_trust, limit | Modules, symbols, endpoints, docs, knowledge and tests potentially affected by changing a node, ranked by evidence-weighted path trust. |
| `get_source` | R | **ref**, context_lines, max_lines | Read source for a node id/name or 'path[:start[-end]]' (repository-confined, bounded). |
| `find_similar` | R | node, text, limit, kinds | Semantically similar nodes (graph-embedding cosine) to a node or free text; lexical when ML is disabled. |
| `add_knowledge_node` | W | **kind**, **name**, text, citations, evidence, confidence, relation, attrs | Append an agent fact to the graph (kinds: ['annotation', 'bug', 'component', 'concept', 'constraint', 'data_entity', 'decision', 'event', 'finding', 'map', 'note', 'pattern', 'question', 'requirement', 'risk', 'roadmap_step', 'subsystem', 'task']). Citations (node ids or path:line) are verified; uncited findings are marked unsupported. Static facts cannot be overwritten. |
| `record_finding` | W | **claim**, citations, confidence, evidence, details | Shortcut: record a finding with citations (appends, never overwrites). |
| `add_relationship` | W | **source**, **target**, **type**, rationale, evidence, confidence | Add a typed, evidence-tagged relationship between existing nodes (types: ['about', 'affects', 'annotated_as', 'blocks', 'calls', 'configures', 'conflicts_with', 'constrains', 'consumes', 'contains', 'defines', 'depends_on', 'documents', 'implements', 'implements_pattern', 'imports', 'invokes', 'part_of', 'produces', 'reads', 'references', 'routes_to', 'similar_to', 'supersedes', 'supports', 'verifies', 'writes']). Reports corroboration and conflicts. |
| `annotate_node` | W | **node**, **concept**, confidence | Attach a semantic concept to a node (e.g. 'vector_candidate_selection'). |
| `supersede_fact` | W | **old**, **new**, reason | Mark an agent/handoff fact as superseded by a newer one (history is preserved). |
| `create_map` | W | **title**, **nodes**, description, include_edges, extra_edges | Create/replace a named architectural map: a curated subgraph (plus optional new relationships) persisted with Mermaid source. |
| `get_map` | R | **map** | Retrieve a saved map with its Mermaid source. |
| `list_maps` | R | — | List saved maps. |
| `build_task_context` | R | **task**, budget, constraints, validation, focus, include_source, format | Token-budgeted context package for a task: constraints and validation (never dropped), ranked facts with evidence, source excerpts, relationships, next actions. |
| `verify_claims` | R | **claims** | Check claims against source evidence: citations must resolve; source/target relationships must be supported by observed edges or paths. |
| `create_handoff` | W | **handoff**, parent_id | Persist a durable, versioned task record (findings with sources, constraints, relevant symbols, decisions, completed work, next action, validation). parent_id branches a task. |
| `update_handoff` | W | **handoff_id**, **patch**, expected_version | Append to a handoff (lists are appended, scalars replaced). Pass expected_version for optimistic concurrency; contradictory findings are recorded as conflicts. |
| `resume_handoff` | R | **handoff_id**, budget, claim, include_source | Reconstruct task context for another agent: handoff, freshness of symbols/findings since its revision, lineage, and a budgeted context package. claim=true marks it in_progress. |
| `branch_handoff` | W | **parent_id**, **task**, next_action | Start a sub-task that inherits a handoff's constraints and context. |
| `list_handoffs` | R | status | List handoffs (latest versions), optionally by status. |
| `get_handoff` | R | **handoff_id**, version | A handoff at its latest or a specific version, plus version history. |
| `get_analysis_tasks` | W | sections, goal, status, claim, limit | Act as a specialist agent: optionally queue bounded analysis contracts for sections, then claim pending ones. Each contract has scope, facts, output schema and evidence rules. |
| `submit_analysis` | W | **task_id**, **result** | Submit a specialist result for a claimed task. Validated against the schema; claims and relationships merge into the graph as agent inference with verified citations. |
| `get_ml_status` | R | — | Embedding/classifier tier in use, model availability and the classification tasks. |
| `classify` | R | **task**, **text** | Run the two-tier classifier (rules + ML/DL) on text for a task: layer, resource, design_pattern, doc_intent, note_intent. |
| `submit_classification_feedback` | W | **task**, **text**, **label** | Teach the classifiers a labelled example (stored in the repo's .promapkin/training). |
| `ingest_runtime_trace` | W | **directory** | Ingest a runtime trace directory produced by `promapkin trace run` (inside the repository) as runtime evidence: real call edges, test→code links, dynamic links static analysis missed. Promapkin never executes code from MCP. |
| `get_toolchain_status` | R | — | Which parsers are active per language (TypeScript compiler, tree-sitter, lexical) and how to install the real ones. |
| `run_benchmark` | R | spec, budget, limit | Retrieval benchmark: recall@k, MRR, context hit rate and token reduction vs naive exploration, with an ML-vs-heuristic ablation. Optional spec {tasks:[{query, expected:[files/symbols]}]}. |

Required arguments are **bold**.

## Prompts

- `investigate` (issue): Investigate an issue and leave a handoff
- `implement_from_handoff` (handoff_id): Continue another agent's work
- `review_change` (handoff_id, changed_symbols): Review a change against the blueprint
- `map_subsystem` (subsystem): Build an architectural map for a subsystem

## Resources

- `promapkin://overview` — project overview (JSON)
- `promapkin://agent-guide` — compact orientation for agents
- `promapkin://blueprint/<section>` — blueprint report (Markdown)
- `promapkin://handoff/<id>` — handoff record (JSON)
