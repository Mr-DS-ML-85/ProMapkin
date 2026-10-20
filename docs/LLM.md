# LLM mode (`--llm`)

Coding agents use Promapkin over MCP and bring their own model. Developers at a terminal can use one through
the CLI: add `--llm` to **any** command and its interpretive work is done by Anthropic or OpenAI models instead
of regexes, rules and small classifiers.

```bash
pip install -e '.[llm]'                     # official anthropic + openai SDKs
promapkin auth set anthropic                # or: export ANTHROPIC_API_KEY=...
promapkin --llm scan                        # LLM extraction/classification where regex was used
promapkin --llm graph impact "APIRouter"    # result + plain-language explanation
promapkin --llm blueprint generate --goal "add rate limiting"
promapkin ask "where is request validation done, and what calls it?"
promapkin --llm review                      # review uncommitted changes against the graph
```

## What changes per command

| Command | Without `--llm` | With `--llm` |
|---|---|---|
| `scan` / `sync` | lexical extraction for files without a parser; regex claim mining in docs; two-tier classifiers | LLM structural extraction for those files (grounded, see below); LLM analysis of docs (claims, decisions, requirements, constraints); every classification task by the LLM |
| `blueprint …` | deterministic specialists | same facts, plus LLM synthesis per specialist, LLM design-pattern classification, LLM-refined roadmap |
| `graph query` | symbol + BM25 + embeddings + graph | LLM query expansion and re-ranking, plus an explanation |
| `graph impact/trace/dataflow/deps/node/hubs/communities/surprises` | ranked JSON / text | same, plus an explanation of what matters and what to do |
| `context build` | budgeted package | LLM-reranked seeds plus a brief |
| `verify` | static corroboration of doc claims | plus an LLM verdict per claim |
| `handoff create --notes` | notes stored as summary | notes structured into findings/constraints/next action/validation |
| `ask` | — | agent loop: the LLM calls Promapkin's own read tools (MCP tool set) to answer |
| `review` | changed symbols, impact, tests to run, decisions | plus LLM review findings tied to diff lines |

Python keeps using the AST and JS/TS the TypeScript compiler when installed: real parsers beat a model at
extracting syntax. `--llm` replaces the *guessing* parts.

## Providers

| | Anthropic | OpenAI / compatible |
|---|---|---|
| SDK | `anthropic` | `openai` |
| Default model | `claude-opus-5-5` (`anthropic_model`) | `gpt-5` (`openai_model`) |
| Key | `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN` or `ant auth login` profile, or `promapkin auth set anthropic` | `OPENAI_API_KEY` or `promapkin auth set openai` |
| Base URL | `anthropic_base_url` / `--base-url` (gateways) | `openai_base_url` / `--base-url` (Azure gateways, vLLM, Ollama, LM Studio …) |

Flags override the config: `--llm openai --model qwen2.5-coder --base-url http://localhost:11434/v1`.
For OpenAI-compatible local servers no key is required.

**External agents mode** (blueprint specialists without `--llm`) uses the same clients: set `"provider":
"anthropic"` or `"openai"`, `"allow_remote": true`, and optionally `openai_base_url` / `provider_model` in
`.promapkin/config.json`.

### Endpoints that ignore `response_format`

Some OpenAI-compatible proxies accept `response_format` but never enforce it. OmniRoute's **DeepSeek Web** provider
(`ds-web/...`) is one: it drives chat.deepseek.com and sends the model a single prompt, so strict JSON schemas are
silently dropped. Tool calls are emulated, so `ask` mostly works, but long structured replies (document analysis)
come back wrapped in markdown or prose. Promapkin then reports, for example: `model wrapped the JSON in other text
(finish_reason=stop, 5123 chars; reply starts '```json\n{\n')`.

Set `openai_structured_output` in `.promapkin/config.json`, or use `PROMAPKIN_LLM_STRUCTURED` for one run:

| Value | Request | Accepted reply |
|---|---|---|
| `json_schema` (default) | strict `response_format: json_schema` | exactly a JSON object |
| `json_object` | JSON mode, schema in the system prompt | exactly a JSON object |
| `prompt` | schema in the system prompt only | a JSON object, bare or as one ```` ```json ```` block |

Every mode validates the reply against the schema; nothing is repaired or guessed. To see what an endpoint
actually returned, run with `PROMAPKIN_LLM_DEBUG=1`: each invalid reply is saved, with secrets redacted, to
`.promapkin/llm-debug/` (local only; it can contain text from your documents).

OmniRoute's DeepSeek Web provider replays only the last 20 messages of a conversation. `ask` repeats your question
in the system prompt so long tool loops keep it.

On the first-party Anthropic API, requests opt into server-side refusal fallback (`llm_fallbacks`, on by
default; skipped automatically behind custom base URLs).

## Guarantees

- **Grounded extraction.** Every symbol, call, import, route and env read the model reports must be found in the
  source at (or next to) the reported line; anything else is dropped and counted ("ungrounded facts dropped").
  Doc claims, decisions and requirements must quote the document verbatim. LLM-read structure is tagged
  `llm_extraction`, LLM interpretations `llm_inference`, so graph traversal weighs them accordingly.
- **Structured outputs.** All analysis calls use strict JSON schemas (`output_config.format` on Anthropic,
  `response_format: json_schema` with `strict: true` on OpenAI).
- **Fallbacks.** Any failure, refusal, invalid reply or budget exhaustion falls back to the deterministic
  implementation for that item; the command still completes. A failed document analysis keeps the regex-extracted
  facts unchanged.
- **No network calls under the index lock.** LLM calls run before or after write transactions, never inside them.
- **Caching.** Responses are cached in the index by a hash of provider, model, task and exact input: unchanged
  files are never re-sent.
- **Cost bound.** `llm_max_calls` (default 2,000 per command) and `llm_concurrency` (default 4). Each run prints
  calls, cached calls, input/output tokens, fallbacks, redactions and dropped facts to stderr.
- **Privacy.** `--llm` is explicit consent for that command. Likely secrets are redacted before sending; every
  request is logged (task, subject, size, never content) to `.promapkin/egress.log`.
