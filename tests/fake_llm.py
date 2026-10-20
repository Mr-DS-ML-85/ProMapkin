"""A local HTTP server speaking the Anthropic Messages and OpenAI Chat Completions wire formats.

Used to test the LLM layer through the official SDKs (via their base_url) without network access.
Answers are derived from the request so grounding, caching and tool loops are exercised for real.
"""

from __future__ import annotations

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


def _fake(schema: dict[str, Any], user: str) -> Any:
    props = schema.get("properties", {})
    if "items" in props and props["items"].get("type") == "array":          # classification batch
        items = json.loads(user)
        enum = props["items"]["items"]["properties"]["label"]["enum"]
        out = []
        for it in items:
            text = it["item"].lower()
            label = enum[0]
            for cand in ("persistence", "repository", "interface", "domain", "integration", "rationale", "todo"):
                if cand in enum and (cand in text or (cand == "persistence" and "storage" in text)
                                     or (cand == "repository" and "repository" in text)
                                     or (cand == "interface" and ("api" in text or "routes" in text))):
                    label = cand
                    break
            out.append({"index": it["index"], "label": label, "confidence": 0.9, "reason": "fake"})
        return {"items": out}
    if "symbols" in props:                                                     # code extraction
        symbols, calls = [], []
        for m in re.finditer(r"^\s*(\d+)\| (.*)$", user, re.M):
            line, text = int(m.group(1)), m.group(2)
            d = re.search(r"\b(?:fn|def|function|func)\s+(\w+)", text)
            if d:
                symbols.append({"name": d.group(1), "kind": "function", "parent": "", "line_start": line,
                                "line_end": line + 2, "doc": "", "params": [], "bases": [], "exported": True})
            c = re.search(r"(\w+)\(", text)
            if c and not d:
                calls.append({"caller": "", "callee": c.group(1), "line": line})
        symbols.append({"name": "ghost_function", "kind": "function", "parent": "", "line_start": 1, "line_end": 1,
                        "doc": "", "params": [], "bases": [], "exported": True})   # ungrounded: must be dropped
        return {"module_doc": "fake module", "symbols": symbols, "imports": [], "calls": calls, "routes": [],
                "env": [], "resources": []}
    if "claims" in props and "decisions" in props:                             # documentation
        m = re.search(r"`(\w+)` uses `(\w+)`[^\n]*", user)
        claims = [{"subject": m.group(1), "relation": "depends_on", "object": m.group(2), "quote": m.group(0)}] if m \
            else []
        claims.append({"subject": "Ghost", "relation": "calls", "object": "Phantom", "quote": "never written here"})
        return {"claims": claims, "decisions": [], "requirements": [], "code_mentions": []}
    if "identifiers" in props:
        return {"identifiers": ["select_candidates"], "keywords": ["candidate"]}
    if "ranking" in props:
        cands = json.loads(user)["candidates"]
        return {"ranking": [{"id": c["id"], "relevance": round(1 - i / max(len(cands), 1), 3)}
                            for i, c in enumerate(reversed(cands))]}
    if "verdicts" in props:
        return {"verdicts": [{"index": 0, "verdict": "contradicted", "explanation": "fake"}]}
    if "summary" in props and "claims" in props:                               # specialist synthesis
        return {"summary": "fake synthesis", "claims": [{"claim": "fake claim", "kind": "finding",
                "citations": ["searchapp/cli.py:1"], "confidence": 0.5, "evidence": "agent_inference",
                "rationale": "fake"}], "relationships": [], "annotations": [], "questions": ["fake?"]}
    return _minimal(schema)


def _minimal(schema: dict[str, Any]) -> Any:
    t = schema.get("type")
    if "enum" in schema:
        return schema["enum"][0]
    if t == "object":
        return {k: _minimal(v) for k, v in schema.get("properties", {}).items()}
    if t == "array":
        return []
    return {"string": "fake", "integer": 1, "number": 0.5, "boolean": True}.get(t, None)


class FakeLLM:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a: Any) -> None:
                pass

            def do_POST(self) -> None:  # noqa: N802
                body = json.loads(self.rfile.read(int(self.headers["content-length"])))
                outer.requests.append({"path": self.path, "body": body, "headers": dict(self.headers)})
                resp = outer.anthropic(body) if self.path.endswith("/messages") else outer.openai(body)
                data = json.dumps(resp).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()

    # ------------------------------------------------------------------ Anthropic
    def anthropic(self, body: dict[str, Any]) -> dict[str, Any]:
        msgs = body["messages"]
        last = msgs[-1]["content"]
        user = last if isinstance(last, str) else json.dumps(last)
        content: list[dict[str, Any]]
        stop = "end_turn"
        fmt = (body.get("output_config") or {}).get("format")
        if fmt:
            content = [{"type": "text", "text": json.dumps(_fake(fmt["schema"], user))}]
        elif body.get("tools") and len(msgs) == 1:
            content = [{"type": "text", "text": "Let me look."},
                       {"type": "tool_use", "id": "toolu_1", "name": "search_graph", "input": {"query": "candidate"}}]
            stop = "tool_use"
        else:
            content = [{"type": "text", "text": "Fake explanation: select_candidates caps candidates."}]
        return {"id": "msg_fake", "type": "message", "role": "assistant", "model": body["model"], "content": content,
                "stop_reason": stop, "stop_sequence": None, "usage": {"input_tokens": 100, "output_tokens": 20}}

    # ------------------------------------------------------------------ OpenAI
    def openai(self, body: dict[str, Any]) -> dict[str, Any]:
        msgs = body["messages"]
        user = msgs[-1].get("content") or ""
        message: dict[str, Any] = {"role": "assistant", "content": None}
        finish = "stop"
        rf = body.get("response_format")
        if rf:
            message["content"] = json.dumps(_fake(rf["json_schema"]["schema"], user))
        elif body.get("tools") and not any(m["role"] == "tool" for m in msgs):
            message["tool_calls"] = [{"id": "call_1", "type": "function", "function": {
                "name": "search_graph", "arguments": json.dumps({"query": "candidate"})}}]
            finish = "tool_calls"
        else:
            message["content"] = "Fake explanation from OpenAI-compatible server."
        return {"id": "chatcmpl-fake", "object": "chat.completion", "created": 0, "model": body["model"],
                "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}}
