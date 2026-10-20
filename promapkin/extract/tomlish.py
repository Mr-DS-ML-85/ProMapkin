"""TOML for Python 3.10, where `tomllib` is missing: the subset manifests use (tables, dotted and quoted keys,
strings, numbers, booleans, arrays, inline tables). Values it cannot read become raw strings."""

from __future__ import annotations

import re
from typing import Any


class TOMLDecodeError(ValueError):
    pass


def loads(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    table = root
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = _strip_comment(lines[i]).strip()
        i += 1
        if not line:
            continue
        m = re.fullmatch(r"\[\[\s*(.+?)\s*\]\]", line)
        if m:
            parent = _descend(root, _keys(m.group(1))[:-1])
            arr = parent.setdefault(_keys(m.group(1))[-1], [])
            table = {}
            arr.append(table)
            continue
        m = re.fullmatch(r"\[\s*(.+?)\s*\]", line)
        if m:
            table = _descend(root, _keys(m.group(1)))
            continue
        key, sep, raw = line.partition("=")
        if not sep:
            continue
        raw = raw.strip()
        while _unbalanced(raw) and i < len(lines):          # multi-line arrays / inline tables
            raw += " " + _strip_comment(lines[i]).strip()
            i += 1
        if raw.startswith(('"""', "'''")) and raw.count(raw[:3]) < 2:
            quote = raw[:3]
            while i < len(lines) and quote not in lines[i]:
                raw += "\n" + lines[i]
                i += 1
            if i < len(lines):
                raw += "\n" + lines[i]
                i += 1
        keys = _keys(key.strip())
        _descend(table, keys[:-1])[keys[-1]] = _value(raw)
    return root


def _keys(raw: str) -> list[str]:
    return [k.strip().strip('"\'') for k in re.findall(r'"[^"]*"|\'[^\']*\'|[^.]+', raw)]


def _descend(node: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    for k in keys:
        nxt = node.setdefault(k, {})
        node = nxt[-1] if isinstance(nxt, list) and nxt else nxt
    return node


def _strip_comment(line: str) -> str:
    quote = ""
    for i, ch in enumerate(line):
        if quote:
            if ch == quote:
                quote = ""
        elif ch in "\"'":
            quote = ch
        elif ch == "#":
            return line[:i]
    return line


def _unbalanced(raw: str) -> bool:
    depth, quote = 0, ""
    for ch in raw:
        if quote:
            if ch == quote:
                quote = ""
        elif ch in "\"'":
            quote = ch
        elif ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
    return depth > 0


def _split(inner: str) -> list[str]:
    parts, depth, quote, cur = [], 0, "", ""
    for ch in inner:
        if quote:
            cur += ch
            if ch == quote:
                quote = ""
            continue
        if ch in "\"'":
            quote = ch
        elif ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
            continue
        cur += ch
    if cur.strip():
        parts.append(cur)
    return parts


def _value(raw: str) -> Any:
    v = raw.strip()
    if v[:3] in ('"""', "'''"):
        return v[3:-3].lstrip("\n")
    if v[:1] in "\"'" and v[-1:] == v[:1]:
        return v[1:-1].replace('\\"', '"').replace("\\\\", "\\") if v[0] == '"' else v[1:-1]
    if v.startswith("["):
        return [_value(p) for p in _split(v[1:-1]) if p.strip()]
    if v.startswith("{"):
        out: dict[str, Any] = {}
        for p in _split(v[1:-1]):
            k, _, val = p.partition("=")
            if k.strip():
                keys = _keys(k.strip())
                _descend(out, keys[:-1])[keys[-1]] = _value(val)
        return out
    if v in ("true", "false"):
        return v == "true"
    if re.fullmatch(r"[+-]?\d[\d_]*", v):
        return int(v.replace("_", ""))
    if re.fullmatch(r"[+-]?\d[\d_]*\.\d+([eE][+-]?\d+)?", v):
        return float(v.replace("_", ""))
    return v
