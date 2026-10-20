"""Single-file components (Vue, Svelte): the `<script>` blocks are extracted as JavaScript/TypeScript with
their original line numbers; the component itself becomes a class-like node so templates that use it can
be linked by name."""

from __future__ import annotations

import re
from pathlib import PurePosixPath

from promapkin import evidence as ev
from promapkin.extract import Facts

SCRIPT = re.compile(r"<script\b([^>]*)>(.*?)</script>", re.S | re.I)
COMPONENT_TAG = re.compile(r"<([A-Z][\w]*|[a-z]+(?:-[a-z]+)+)[\s/>]")


def extract(path: str, text: str) -> Facts:
    from promapkin.extract.js_ext import extract as js_extract

    blocks = list(SCRIPT.finditer(text))
    # Blank everything outside the scripts so line numbers survive and the JS extractor sees only code.
    code = "".join(ch if ch == "\n" else " " for ch in text)
    for m in blocks:
        start = m.start(2)
        code = code[:start] + m.group(2) + code[m.end(2):]
    facts = js_extract(path, code)
    facts.language = PurePosixPath(path).suffix.lstrip(".")
    lang = "typescript" if any(re.search(r"lang\s*=\s*[\"']ts", m.group(1)) for m in blocks) else "javascript"
    name = PurePosixPath(path).stem
    mod = next((n for n in facts.nodes if n["kind"] == "module"), None)
    if mod is not None:
        mod["attrs"]["component"] = name
        mod["attrs"]["script_language"] = lang
    template = SCRIPT.sub("", text)
    used = sorted({m.group(1) for m in COMPONENT_TAG.finditer(template)} - {"template", "script", "style", "slot"})
    owner = mod["id"] if mod is not None else f"file:{path}"
    for tag in used[:100]:
        component = "".join(p.capitalize() for p in tag.split("-")) if "-" in tag else tag
        facts.ref(owner, component, "references", None, module=mod["name"] if mod else name, component=True)
    if mod is not None:
        mod.setdefault("attrs", {})["parser"] = f"{facts.language}+{mod['attrs'].get('parser', 'js')}"
        mod["evidence"] = mod.get("evidence", ev.LEXICAL)
    return facts
