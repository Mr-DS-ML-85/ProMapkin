"""Documentation extraction: sections, architecture decision records, symbol mentions and claims."""

from __future__ import annotations

import re
from pathlib import PurePosixPath

from promapkin import evidence as ev
from promapkin.extract import Facts
from promapkin.security import suspicious_instructions

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$", re.M)
BACKTICK_RE = re.compile(r"`([A-Za-z_][\w.:/-]{2,})(?:\(\))?`")
CAMEL_RE = re.compile(r"\b([A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+)\b")
LINK_RE = re.compile(r"\[[^\]]*\]\(([^)#\s]+)(?:#[^)]*)?\)")
CLAIM_RE = re.compile(
    r"`?(?P<a>[A-Za-z_][\w.:]+)`?\s+(?P<verb>depends on|calls|uses|imports|reads from|writes to|invokes)\s+"
    r"`?(?P<b>[A-Za-z_][\w.:]+)`?", re.IGNORECASE)
ADR_SECTION_NAMES = {"status", "context", "decision", "consequences", "rationale", "alternatives",
                     "alternatives considered", "considered options", "decision outcome"}


def slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-") or "section"


def is_adr(path: str, text: str) -> bool:
    p = PurePosixPath(path.lower())
    in_adr_dir = any(part in ("adr", "adrs", "decisions", "architecture-decisions") for part in p.parts[:-1])
    named = bool(re.match(r"(adr[-_]?)?\d{3,4}[-_]", p.name))
    headings = {h.strip().lower() for _, h in HEADING_RE.findall(text)}
    return in_adr_dir or named or ({"status", "decision"} <= headings and "context" in headings)


def extract(path: str, text: str) -> Facts:
    facts = Facts(path, "markdown")
    file_id = f"file:{path}"
    warnings = suspicious_instructions(text)
    sections = _sections(text)
    title = next((t for lvl, t, _, _ in sections if lvl == 1), PurePosixPath(path).stem)

    doc_id = f"doc:{path}"
    facts.node(doc_id, "document", title, ev.DOCUMENTATION, file=path, line_start=1,
               line_end=text.count("\n") + 1, text=text[:600],
               attrs={"untrusted_instructions": warnings, "sections": len(sections)})
    facts.edge(file_id, doc_id, "defines", ev.DOCUMENTATION)

    if is_adr(path, text):
        parts = {t.strip().lower(): body.strip() for _, t, body, _ in sections}
        status = (parts.get("status") or "unknown").splitlines()[0].strip(" *_") if parts.get("status") else "unknown"
        did = f"decision:{path}"
        decision_text = parts.get("decision") or parts.get("decision outcome") or ""
        facts.node(did, "decision", title, ev.DOCUMENTATION, file=path, line_start=1,
                   text=" ".join(filter(None, [parts.get("context"), decision_text, parts.get("consequences")]))[:2000],
                   attrs={"status": status, "context": (parts.get("context") or "")[:1200],
                          "decision": decision_text[:1200], "consequences": (parts.get("consequences") or "")[:1200],
                          "alternatives": (parts.get("alternatives considered") or parts.get("alternatives") or
                                           parts.get("considered options") or "")[:800],
                          "origin": "adr", "untrusted_instructions": warnings})
        facts.edge(doc_id, did, "records", ev.DOCUMENTATION)
        _mentions(facts, did, text, 1)

    for level, heading, body, line in sections:
        if level == 1 and not body.strip():
            continue
        sid = f"doc:{path}#{slug(heading)}"
        facts.node(sid, "doc_section", heading, ev.DOCUMENTATION, file=path, line_start=line,
                   line_end=line + body.count("\n"), text=body.strip()[:1500],
                   attrs={"level": level, "untrusted_instructions": suspicious_instructions(body)})
        facts.edge(doc_id, sid, "contains", ev.DOCUMENTATION)
        _mentions(facts, sid, body, line)
    if not sections:
        _mentions(facts, doc_id, text, 1)
    return facts


def _mentions(facts: Facts, src: str, body: str, base_line: int) -> None:
    seen = set()
    for rx in (BACKTICK_RE, CAMEL_RE):
        for m in rx.finditer(body):
            name = m.group(1).rstrip(".:")
            if name in seen or len(name) < 3:
                continue
            seen.add(name)
            facts.ref(src, name, "mentions", base_line + body.count("\n", 0, m.start()))
    for m in LINK_RE.finditer(body):
        target = m.group(1)
        if not re.match(r"[a-z]+://", target):
            facts.ref(src, target, "links", base_line + body.count("\n", 0, m.start()),
                      doc=facts.path)
    for m in CLAIM_RE.finditer(body):
        facts.ref(src, m.group("b"), "claims", base_line + body.count("\n", 0, m.start()),
                  subject=m.group("a"), verb=m.group("verb").lower(), sentence=m.group(0)[:200])


def _sections(text: str) -> list[tuple[int, str, str, int]]:
    """(level, heading, body, line) for every markdown heading, ignoring headings in code fences."""
    fenced = _fence_mask(text)
    heads = [m for m in HEADING_RE.finditer(text) if not fenced[m.start()]]
    out = []
    for i, m in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        out.append((len(m.group(1)), m.group(2).strip(), text[m.end():end], text.count("\n", 0, m.start()) + 1))
    return out


def _fence_mask(text: str) -> list[bool]:
    mask = [False] * (len(text) + 1)
    inside = False
    pos = 0
    for line in text.splitlines(keepends=True):
        if line.lstrip().startswith(("```", "~~~")):
            inside = not inside
        if inside:
            for k in range(pos, pos + len(line)):
                mask[k] = True
        pos += len(line)
    return mask
