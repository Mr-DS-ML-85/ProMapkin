"""YAML for infrastructure files without a dependency: PyYAML when installed, otherwise a small parser for
the subset these files use (block mappings and sequences, plain/quoted scalars, one-line flow collections,
block scalars, comments, multiple documents). Helm template directives are blanked first.

Every mapping parsed here also remembers the line its key was on (`LineDict.lines`), so extracted facts can
cite `file:line`.
"""

from __future__ import annotations

import re
from typing import Any

TEMPLATE = re.compile(r"\{\{-?.*?-?\}\}")


class LineDict(dict):
    """A dict that remembers the source line (1-based) of each key."""

    def __init__(self, *a: Any, **kw: Any):
        super().__init__(*a, **kw)
        self.lines: dict[str, int] = {}
        self.line = 0


def load_all(text: str) -> list[Any]:
    """Every document in `text`; a document that cannot be parsed is skipped."""
    text = _detemplate(text)
    docs, start = [], 0
    chunks = re.split(r"^---[^\n]*$", text, flags=re.M)
    for chunk in chunks:
        try:
            doc = _Parser(chunk, start).parse()
        except (ValueError, IndexError):
            doc = None
        if doc is not None:
            docs.append(doc)
        start += chunk.count("\n") + 1
    return docs


def _detemplate(text: str) -> str:
    out = []
    for line in text.splitlines():
        if TEMPLATE.fullmatch(line.strip() or "x"):
            out.append("")                       # a line that is only a template directive
        else:
            out.append(TEMPLATE.sub("TEMPLATED", line))
    return "\n".join(out)


def scalar(raw: str) -> Any:
    s = raw.strip()
    if not s or s == "~" or s == "null":
        return None
    if s[0] in "\"'" and s[-1:] == s[0] and len(s) >= 2:
        return s[1:-1].replace("\\n", "\n") if s[0] == '"' else s[1:-1].replace("''", "'")
    if s[0] in "[{":
        return _flow(s)
    if s in ("true", "True", "yes", "on"):
        return True
    if s in ("false", "False", "no", "off"):
        return False
    if re.fullmatch(r"-?\d+", s):
        return int(s)
    if re.fullmatch(r"-?\d+\.\d*", s):
        return float(s)
    return s.lstrip("&*!").split(" #")[0].strip() if s[0] in "&*!" else s.split(" #")[0].rstrip()


def _flow(s: str) -> Any:
    inner = s[1:-1] if s[-1:] in "]}" else s[1:]
    parts, depth, cur, quote = [], 0, "", ""
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
    if s[0] == "[":
        return [scalar(p) for p in parts]
    out = LineDict()
    for p in parts:
        k, _, v = p.partition(":")
        out[str(scalar(k))] = scalar(v)
    return out


class _Parser:
    def __init__(self, text: str, offset: int):
        self.rows: list[tuple[int, int, str]] = []    # (line number, indent, content)
        lines = text.split("\n")
        i = 0
        while i < len(lines):
            raw = lines[i]
            stripped = _strip_comment(raw)
            if stripped.strip():
                self.rows.append((offset + i + 1, len(raw) - len(raw.lstrip(" ")), stripped.strip()))
            i += 1
        self.i = 0

    def parse(self) -> Any:
        if not self.rows:
            return None
        return self._block(self.rows[0][1])

    def _block(self, indent: int) -> Any:
        if self.i >= len(self.rows):
            return None
        _, ind, content = self.rows[self.i]
        if content.startswith("- ") or content == "-":
            return self._seq(ind)
        return self._map(ind)

    def _map(self, indent: int) -> LineDict:
        out = LineDict()
        out.line = self.rows[self.i][0] if self.i < len(self.rows) else 0
        while self.i < len(self.rows):
            line, ind, content = self.rows[self.i]
            if ind < indent or content.startswith("- ") and ind == indent:
                break
            if ind > indent:
                self.i += 1          # stray deeper line: skip
                continue
            m = re.match(r"""^("[^"]*"|'[^']*'|[^:#{}\[\]][^:]*?)\s*:(?:\s+(.*)|$)""", content)
            if not m:
                self.i += 1
                continue
            key = str(scalar(m.group(1)))
            rest = (m.group(2) or "").strip()
            out.lines[key] = line
            self.i += 1
            out[key] = self._value(rest, ind)
        return out

    def _seq(self, indent: int) -> list[Any]:
        out: list[Any] = []
        while self.i < len(self.rows):
            line, ind, content = self.rows[self.i]
            if ind != indent or not (content.startswith("- ") or content == "-"):
                break
            item = content[1:].strip()
            if not item:
                self.i += 1
                out.append(self._block(self.rows[self.i][1]) if self.i < len(self.rows) and
                           self.rows[self.i][1] > ind else None)
                continue
            if re.match(r"""^("[^"]*"|'[^']*'|[^:#{}\[\]'"][^:]*?)\s*:(\s|$)""", item):
                # "- key: value" starts a mapping whose other keys are indented to the item's column
                col = ind + (len(content) - len(item))
                self.rows[self.i] = (line, col, item)
                out.append(self._map(col))
            else:
                self.i += 1
                out.append(self._value(item, ind))
        return out

    def _value(self, rest: str, ind: int) -> Any:
        if rest in ("|", "|-", "|+", ">", ">-", ">+"):
            parts = []
            while self.i < len(self.rows) and self.rows[self.i][1] > ind:
                parts.append(self.rows[self.i][2])
                self.i += 1
            return ("\n" if rest.startswith("|") else " ").join(parts)
        if rest:
            return scalar(rest)
        if self.i < len(self.rows) and (self.rows[self.i][1] > ind or
                                        (self.rows[self.i][1] == ind and self.rows[self.i][2].startswith("- "))):
            return self._block(self.rows[self.i][1])
        return None


def _strip_comment(line: str) -> str:
    if "#" not in line:
        return line
    quote = ""
    for i, ch in enumerate(line):
        if quote:
            if ch == quote:
                quote = ""
        elif ch in "\"'":
            quote = ch
        elif ch == "#" and (i == 0 or line[i - 1] in " \t"):
            return line[:i]
    return line


def load(text: str) -> list[Any]:
    """PyYAML when available (faster, complete); otherwise the subset parser. Always a list of documents."""
    try:
        import yaml  # type: ignore[import-untyped]
    except ModuleNotFoundError:
        return load_all(text)
    try:
        return [d for d in yaml.safe_load_all(_detemplate(text)) if d is not None]
    except yaml.YAMLError:
        return load_all(text)
