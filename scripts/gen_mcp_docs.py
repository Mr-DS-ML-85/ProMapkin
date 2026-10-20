"""Regenerate docs/MCP_TOOLS.md from the MCP server's tool, prompt and resource definitions.

    python scripts/gen_mcp_docs.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from promapkin.mcp_server import PROMPTS, TOOLS  # noqa: E402

RESOURCES = [
    ("promapkin://overview", "project overview (JSON)"),
    ("promapkin://agent-guide", "compact orientation for agents"),
    ("promapkin://blueprint/<section>", "blueprint report (Markdown)"),
    ("promapkin://handoff/<id>", "handoff record (JSON)"),
]


def main() -> None:
    out = ["# MCP tool reference", "",
           "Generated from `promapkin/mcp_server.py` by `scripts/gen_mcp_docs.py` (`promapkin mcp inspect --json` "
           "prints the same data with full schemas).", "",
           "R = read-only, W = writes shared knowledge or artifacts (validated, append-only, provenance recorded).", "",
           f"{len(TOOLS)} tools.", "", "| Tool | | Arguments | Description |", "|---|---|---|---|"]
    for t in TOOLS:
        props = t.schema.get("properties", {})
        required = set(t.schema.get("required", []))
        args = ", ".join(f"**{k}**" if k in required else k for k in props) or "—"
        desc = " ".join(t.description.split()).replace("|", "/")
        out.append(f"| `{t.name}` | {'R' if t.read_only else 'W'} | {args} | {desc} |")
    out += ["", "Required arguments are **bold**.", "", "## Prompts", ""]
    for name, (description, arguments, _template) in PROMPTS.items():
        out.append(f"- `{name}` ({', '.join(arguments)}): {description}")
    out += ["", "## Resources", ""] + [f"- `{uri}` — {desc}" for uri, desc in RESOURCES]
    (ROOT / "docs" / "MCP_TOOLS.md").write_text("\n".join(out) + "\n")
    print(f"docs/MCP_TOOLS.md: {len(TOOLS)} tools, {len(PROMPTS)} prompts")


if __name__ == "__main__":
    main()
