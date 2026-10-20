# Benchmarking Promapkin

The specification asks for a measurable first benchmark. It should compare an ordinary coding
agent against the same agent using Promapkin on identical tasks, and record:
- token consumption;
- task completion;
- regressions;
- retrieval accuracy;
- setup overhead.

Promapkin ships two instruments for this.

## 1. Retrieval benchmark (`promapkin bench retrieval`)

Measures the retrieval layer alone, cheaply and repeatably.

- **Tasks.** You can pass a spec (`{"tasks": [{"query": "...", "expected": ["file or symbol", ...]}]}`).
  Without one, tasks are generated self-supervised: the query is a symbol's docstring with the
  symbol's own name tokens removed.
- **Metrics:**
  - recall@5 and recall@10 and MRR, over the files of ranked hits;
  - context hit rate: whether an expected file appears in the context package;
  - context precision;
  - mean context tokens;
  - naive exploration tokens: tokens an agent reads when it opens files ranked by query-term hits
    until every expected file has been read;
  - token reduction against that baseline and against reading the whole repository;
  - search and context latency.
- **Ablation.** Each run is repeated with ML disabled (heuristic mode) so the ML contribution is
  visible.
- **Setup overhead.** Last scan time, files indexed and index size.

### Results (v0.3.0)

Hand-written specs of natural-language developer questions. The queries avoid the identifiers they are looking
for, and the expected files were checked by hand: [`examples/bench/fastapi.json`](../examples/bench/fastapi.json),
[`clap.json`](../examples/bench/clap.json) and [`promapkin-self.json`](../examples/bench/promapkin-self.json).
Raw results, including every query's rank, are in [`examples/bench/results/`](../examples/bench/results/).
Reproduce with `promapkin bench retrieval --spec examples/bench/<repo>.json` in a checkout of the same revision.

Built-in ML tier (hashing embeddings and naive Bayes; neural models could not be downloaded in the
environment these were measured in), single core, budget 2,000 tokens per package:

| Repository | Questions | Files | Scan | recall@5 | recall@10 | MRR | Package contains a right file | Package tokens | Grep-and-read tokens | Ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| fastapi/fastapi 0.150.0 (Python) | 22 | 3,155 | 44 s | 0.68 | 0.68 | 0.47 | 0.68 | 1,813 | 437,772 | 242× |
| clap-rs/clap @ccc89b8 (Rust) | 14 | 482 | 8 s | 0.50 | 0.64 | 0.34 | 0.71 | 1,951 | 305,809 | 157× |
| Promapkin (Python) | 20 | 114 | 3 s | 0.80 | 0.85 | 0.66 | 0.90 | 1,946 | 72,779 | 37× |

How to read it:
- **Grep-and-read tokens** is a deliberately simple baseline: open files ranked by how often they contain the
  question's words, whole, until every expected file has been read. A capable agent reads less than this, so
  the ratio is an upper bound on the saving, not a typical one.
- **Package contains a right file** is the number to weigh the ratio against. For 32% of the fastapi questions
  the 2k-token package did not include an expected file, and an agent would still have to explore.
- Search takes 2–70 ms with the heuristic tier and up to ~370 ms on fastapi with built-in embeddings (pure
  Python); building a package takes 25–450 ms.
- The heuristic-only ablation scores within ±0.05 of the built-in ML tier on every repository. The built-in
  embeddings are mostly lexical, so the ML tier's value is expected from neural models (`promapkin models
  download`), which have not been measured yet.

What changed in v0.3 to get here, measured on the same specs:
- Tests, examples (`docs_src/`, `examples/`) and vendored code are down-weighted unless the question asks for
  them. Documentation sections that match the question pass their score to the code they reference, skipping
  changelogs and contributing guides. fastapi MRR went from 0.37 to 0.47, and the package hit rate from 0.27 to 0.68.
- Doc comments are read in full, including when attributes such as `#[cfg(...)]` separate them from the
  declaration. clap recall@10 went from 0.36 to 0.64. One clap expectation was widened at the same time: for
  "write a man page", `clap_mangen/src/lib.rs` (`Man`) is as correct an answer as `render.rs`.
- Context packages are cached per retrieval version, and the benchmark never reads the cache. The first v0.3
  measurement had silently reused packages built by older ranking code.

Remaining misses are vocabulary gaps such as "cross-origin" for `cors.py` and "thread pool" for
`run_in_threadpool`, and re-export modules with no code of their own. These are the cases neural embeddings and
`--llm` query expansion target.

### Earlier results (v0.2, Promapkin itself, built-in tier)

recall@10 0.80, MRR 0.71, context hit 0.90, about 1,950 context tokens versus about 32,500 for the naive baseline.
That spec has since grown with the codebase, so the numbers are not directly comparable.

### Real repositories: scan cost

| Repository | Files | Scan | Notes |
|---|---|---|---|
| fastapi/fastapi | 3,155 | 44 s | v0.3 indexes docs, examples, CI and infrastructure too; v0.1 took 310 s on 2,886 files |
| clap-rs/clap (tree-sitter) | 482 | 8 s | 641 types, 1,373 methods, 93 trait impls; `Command::new` has 1,006 resolved callers |
| clap-rs/clap (lexical fallback) | 419 | ~17 s | Rust symbols, `use crate::` paths and `Type::method` calls still resolved |

## 2. Agent A/B harness (`promapkin bench agents --spec spec.json`)

For each task, the harness makes two fresh copies of the repository (`.promapkin/` excluded):

| Arm | What the agent gets |
|---|---|
| `baseline` | the task only (`PROMAPKIN_TASK`) |
| `promapkin` | the task plus `PROMAPKIN_CONTEXT_FILE` / `PROMAPKIN_CONTEXT_JSON` (a budgeted package) and `PROMAPKIN_MCP_COMMAND` (an MCP server for that copy) |

In both arms the steps are:
1. Optional setup commands.
2. The regression command, run before the agent.
3. The agent command.
4. The task's check command.
5. The regression command, run again.

| Recorded | How |
|---|---|
| Token consumption | the agent's last JSON line on stdout: `{"input_tokens": N, "output_tokens": M}` |
| Task completion | `check_command` exit code |
| Regressions | test failures present after the agent ran but not before |
| Retrieval accuracy | share of the task's `expected` files present in the context package |
| Setup overhead | init + scan + context build time in the Promapkin arm |

Print an example spec with `promapkin bench example-spec`. The agent command can be any CLI agent
wrapped to report usage, so the comparison is vendor-neutral.

### Choosing the repository

Pick a repository that has tests, several modules and real cross-module dependencies, for example
a web service with API, service, domain and persistence layers. Write 10–30 tasks that each need
code from at least two modules (a feature, a bug fix, a refactor). Give each task a check command
(a targeted test) and list its expected files. Run each arm several times, because agent runs are
stochastic. Report means and spread.

The harness is tested with a scripted agent (`tests/test_cli_bench_security.py`). **No real-agent
A/B results have been collected yet.** That requires running real agents with API access, and it
is the next measurement to make.
