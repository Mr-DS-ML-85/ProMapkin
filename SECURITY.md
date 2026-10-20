# Security policy

## Model

- **No implicit code execution.** Scanning only parses files: the Python `ast`, the TypeScript compiler API
  (parse + type-check, no emit, no evaluation), tree-sitter grammars or lexical rules. The only command that runs
  code is `promapkin trace run -- <command>`, which runs the command *you* typed. MCP clients can ingest an
  existing trace but cannot run one.
- **Repository content is untrusted.** Instruction-like text in docs or comments (for example "ignore previous
  instructions") is flagged and passed to agents and LLMs as data, inside an explicit untrusted-content notice.
- **File access is confined** to the repository root; `.git` internals are excluded.
- **Nothing leaves the machine by default.** Model downloads need consent (`ml_mode`); LLM calls happen only with
  `--llm` on a command or a remote provider plus `allow_remote: true` in the config. Every LLM request is logged
  to `.promapkin/egress.log`, likely secrets (private keys, cloud and API tokens, password assignments) are redacted
  before sending, and a per-command call budget bounds cost.
- **Credentials** are read from environment variables or the user store `~/.config/promapkin/credentials.json`
  (mode 0600, written atomically). They are never written into the repository or `.promapkin/`.
- **Agent writes are append-only and validated.** Agents cannot overwrite parsed facts, claim AST evidence, or
  link to nodes that do not exist; uncited findings are marked unsupported.

## Reporting a vulnerability

Please open a private security advisory on the GitHub repository rather than a public issue. Include the
version (`promapkin --version`), the command or MCP call, and a minimal reproduction.
