"""Package manifests and build/deploy configuration."""

from __future__ import annotations

import json
import re
from pathlib import PurePosixPath

from promapkin import evidence as ev
from promapkin.extract import Facts

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    from promapkin.extract import tomlish as tomllib  # type: ignore[no-redef]

REQ_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(\[[^\]]*\])?\s*([<>=!~^][^;#]*)?")


def norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def extract(path: str, text: str) -> Facts:
    facts = Facts(path, "manifest")
    name = PurePosixPath(path).name
    mid = f"manifest:{path}"
    facts.node(mid, "manifest", name, ev.MANIFEST, file=path, line_start=1, line_end=text.count("\n") + 1)
    facts.edge(f"file:{path}", mid, "defines", ev.MANIFEST)

    def dep(pkg: str, version: str | None, scope: str, ecosystem: str) -> None:
        pid = f"package:{norm(pkg)}"
        facts.node(pid, "package", pkg, ev.MANIFEST, owner_file=None,
                   attrs={"ecosystem": ecosystem, "external": True})
        facts.edge(mid, pid, "depends_on", ev.MANIFEST, attrs={"version": (version or "").strip(), "scope": scope})

    if name.startswith("requirements") or name == "Pipfile":
        for line in text.splitlines():
            if line.strip().startswith(("#", "-")) or "=" in line and line.strip().startswith("["):
                continue
            m = REQ_RE.match(line)
            if m and m.group(1):
                dep(m.group(1), m.group(3), "dev" if "dev" in name else "runtime", "pypi")
    elif name == "pyproject.toml" and tomllib:
        try:
            data = tomllib.loads(text)
        except Exception as exc:  # noqa: BLE001 - malformed manifests are recorded, not fatal
            facts.errors.append(str(exc))
            return facts
        proj = data.get("project", {})
        for d in proj.get("dependencies", []):
            m = REQ_RE.match(d)
            if m:
                dep(m.group(1), m.group(3), "runtime", "pypi")
        for group, deps in proj.get("optional-dependencies", {}).items():
            for d in deps:
                m = REQ_RE.match(d)
                if m:
                    dep(m.group(1), m.group(3), f"optional:{group}", "pypi")
        for script, target in proj.get("scripts", {}).items():
            _entry(facts, mid, script, target, path)
        poetry = data.get("tool", {}).get("poetry", {})
        for d, v in poetry.get("dependencies", {}).items():
            if d != "python":
                dep(d, str(v), "runtime", "pypi")
    elif name == "package.json":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            facts.errors.append(str(exc))
            return facts
        for scope in ("dependencies", "devDependencies", "peerDependencies"):
            for d, v in (data.get(scope) or {}).items():
                dep(d, v, "dev" if scope == "devDependencies" else "runtime", "npm")
        bins = data.get("bin")
        if isinstance(bins, str):
            bins = {data.get("name", "bin"): bins}
        for script, target in (bins or {}).items():
            _entry(facts, mid, script, target, path)
        if data.get("main"):
            _entry(facts, mid, "main", data["main"], path)
        for script, cmd in (data.get("scripts") or {}).items():
            facts.node(f"script:{path}:{script}", "script", script, ev.MANIFEST, file=path, text=str(cmd)[:300])
            facts.edge(mid, f"script:{path}:{script}", "defines", ev.MANIFEST)
    elif name == "go.mod":
        for m in re.finditer(r"^\s*(?:require\s+)?([\w./-]+\.[\w./-]+)\s+(v[\w.+-]+)", text, re.M):
            dep(m.group(1), m.group(2), "runtime", "go")
    elif name == "Cargo.toml" and tomllib:
        try:
            data = tomllib.loads(text)
        except Exception as exc:  # noqa: BLE001
            facts.errors.append(str(exc))
            return facts
        crate = (data.get("package") or {}).get("name")
        if crate:
            facts.nodes[0].setdefault("attrs", {})["crate"] = crate
        for scope in ("dependencies", "dev-dependencies"):
            for d, v in (data.get(scope) or {}).items():
                dep(d, v if isinstance(v, str) else str((v or {}).get("version", "") if isinstance(v, dict) else v),
                    scope, "cargo")
    elif name == ".env.example":
        for m in re.finditer(r"^\s*([A-Z_][A-Z0-9_]*)\s*=", text, re.M):
            facts.node(f"config:{m.group(1)}", "config", m.group(1), ev.MANIFEST, attrs={"source": "environment"})
            facts.edge(mid, f"config:{m.group(1)}", "configures", ev.MANIFEST)
    elif name.startswith("docker-compose"):
        for m in re.finditer(r"^  ([A-Za-z0-9_-]+):\s*$", text, re.M):
            facts.node(f"service:{m.group(1)}", "service", m.group(1), ev.MANIFEST, file=path,
                       attrs={"source": "docker-compose"})
            facts.edge(mid, f"service:{m.group(1)}", "defines", ev.MANIFEST)
    return facts


def _entry(facts: Facts, mid: str, script: str, target: str, path: str) -> None:
    eid = f"entrypoint:{script}"
    facts.node(eid, "entrypoint", script, ev.MANIFEST, file=path, text=str(target),
               attrs={"target": str(target)})
    facts.edge(mid, eid, "defines", ev.MANIFEST)
    facts.ref(eid, str(target).replace(":", "."), "invokes", None)
