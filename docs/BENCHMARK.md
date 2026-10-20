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

### Results so far

Environment: the cloud container this was built in. HuggingFace downloads were blocked there, so
these numbers use the **built-in** tier (hashing embedder + naive Bayes), not a neural model.

**Promapkin's own repository** (62 files, about 121k tokens), hand-written natural-language spec
`examples/bench/promapkin-self.json`. It has 20 tasks whose queries avoid exact identifiers.

| Mode | recall@5 | recall@10 | MRR | context hit | context tokens | naive tokens | token reduction |
|---|---|---|---|---|---|---|---|
| hybrid (built-in ML) | 0.80 | 0.80 | 0.71 | 0.90 | ~1,950 | ~32,500 | 94% |
| heuristic only | 0.80 | 0.85 | 0.72 | 0.90 | ~1,930 | ~32,500 | 94% |

Setup: a full scan of 62 files took about 2.2 s, and incremental rescans touch only changed files.
Search took about 2–10 ms; building a context package took about 20–30 ms.

Self-supervised tasks on the same repository give recall@10 of 1.0. That is an easy upper bound
because docstrings share vocabulary with their code. Use a hand-written spec for honest numbers.

Observations:
- With the built-in hashing embedder, a full-weight "semantic" signal **reduced** MRR from 0.72 to
  0.58. It mostly duplicates lexical matching and adds noise. Non-neural embeddings are therefore
  down-weighted (0.25), which brought MRR back to 0.71. Neural embeddings keep full weight.
- The remaining misses are vocabulary gaps, such as "risky" for `risks` or "npm and pypi" for
  manifest parsing. These are the cases neural embeddings target. Re-run the spec after
  `promapkin models download` to measure that tier. It has not been measured yet.

### Real repositories (v0.2)

An external audit of v0.1 found that scans crashed (`IndexError` in disambiguation), that fastapi took 310 s, and
that Rust code was invisible. The same repositories measured on v0.2 in this environment (numpy installed, no
neural model, `PROMAPKIN_TIMINGS=1` prints per-phase times):

| Repository | Files | Scan | Notes |
|---|---|---|---|
| fastapi/fastapi | 2,886 | 36.5 s | extract 12.5 s · resolve 1.9 s · classify 6.1 s · embed 6.5 s · propagate 6.4 s · similarity 0.6 s; 19,198 references resolved; blueprint 3.2 s |
| clap-rs/clap (tree-sitter) | 419 | 6.3 s | 641 types, 1,373 methods, 93 trait impls; `Command::new` has 1,006 resolved callers |
| clap-rs/clap (lexical fallback) | 419 | ~17 s | Rust symbols, `use crate::` paths and `Type::method` calls still resolved |

The two main fixes were full-text index maintenance (deletes had scanned an unindexed column) and the
documentation-mention resolver (a linear scan per mention).

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
