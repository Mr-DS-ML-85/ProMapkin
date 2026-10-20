"""Infrastructure as first-class graph nodes: Dockerfiles, Docker Compose, Kubernetes manifests (including
Helm templates), Terraform, CI workflows and OpenAPI specifications.

Infrastructure is linked to the code it runs and configures, not just listed:
- environment variables set by a container, Compose service, Kubernetes workload or cloud function become
  the same `config:KEY` nodes that code reads (`os.environ`, `process.env`, ...), with `configures` edges;
- container commands (`CMD`, `command:`, CI `run:` steps) are parsed and linked to the code module or file
  they start (`runs` edges);
- Kubernetes Services find their workloads by label selector, Ingresses their Services, Compose services
  their build Dockerfile and dependencies; Terraform blocks link through their references;
- OpenAPI operations link to the code endpoints that implement them (`specifies`).

Facts carry `manifest` evidence (declared configuration, observed). Links that need resolution across files
are emitted as `link` references and resolved by the scanner.
"""

from __future__ import annotations

import re
import shlex
from pathlib import PurePosixPath
from typing import Any

from promapkin import evidence as ev
from promapkin.extract import Facts
from promapkin.extract.yamlish import load

INFRA_LANGUAGES = {"dockerfile", "compose", "kubernetes", "terraform", "ci", "openapi"}
WORKLOADS = {"Deployment", "StatefulSet", "DaemonSet", "Job", "CronJob", "ReplicaSet", "Pod", "Rollout",
             "DeploymentConfig"}
CATEGORIES = [
    ("database", r"rds|db_instance|dynamodb|database|sql|postgres|mysql|aurora|cosmos|spanner|bigtable|firestore|"
                 r"documentdb|neptune|redshift|mongodb|bigquery"),
    ("cache", r"elasticache|redis|memcache|memorystore"),
    ("queue", r"sqs|sns|pubsub|queue|kinesis|eventbridge|event_hub|eventhub|kafka|msk|servicebus|mq_|topic"),
    ("storage", r"s3|bucket|storage|blob|efs|filestore|ebs|disk"),
    ("compute", r"lambda|function|ecs|eks|gke|aks|instance|vm|virtual_machine|container|cloud_run|app_service|"
                r"kubernetes|fargate|autoscaling|batch|beanstalk|app_runner|compute"),
    ("network", r"vpc|subnet|security_group|firewall|lb|load_balancer|alb|elb|dns|route53|cdn|cloudfront|"
                r"gateway|nat|endpoint|ingress|network"),
    ("security", r"iam|role|policy|kms|secret|vault|key_vault|certificate|acm|waf"),
    ("observability", r"cloudwatch|monitor|log|alarm|metric|trace|grafana|datadog"),
]


def category(rtype: str) -> str:
    low = rtype.lower()
    for name, pattern in CATEGORIES:
        if re.search(pattern, low):
            return name
    return "other"


def infra_language(path: str, head: str) -> str | None:
    """Which infrastructure format a file is, by name and (for YAML/JSON) content."""
    p = PurePosixPath(path)
    name, low = p.name, p.name.lower()
    if low in ("dockerfile", "containerfile") or low.endswith(".dockerfile") or low.startswith("dockerfile."):
        return "dockerfile"
    if re.fullmatch(r"(docker-)?compose(\.[\w-]+)?\.ya?ml", low):
        return "compose"
    if p.suffix in (".tf", ".tfvars") or name == "terragrunt.hcl":
        return "terraform"
    if p.suffix in (".yml", ".yaml"):
        if ".github/workflows/" in f"/{path}" or name in (".gitlab-ci.yml",):
            return "ci"
        if re.search(r"^(openapi|swagger)\s*:", head, re.M):
            return "openapi"
        if re.search(r"^apiVersion\s*:", head, re.M) and re.search(r"^kind\s*:", head, re.M):
            return "kubernetes"
        return None
    if p.suffix == ".json" and re.search(r'"(openapi|swagger)"\s*:', head[:2000]):
        return "openapi"
    return None


def extract(path: str, text: str, language: str) -> Facts:
    facts = Facts(path, language)
    mid = f"manifest:{path}"
    facts.node(mid, "manifest", PurePosixPath(path).name, ev.MANIFEST, file=path, line_start=1,
               line_end=text.count("\n") + 1, attrs={"format": language})
    facts.edge(f"file:{path}", mid, "defines", ev.MANIFEST)
    try:
        {"dockerfile": _dockerfile, "compose": _compose, "kubernetes": _kubernetes, "terraform": _terraform,
         "ci": _ci, "openapi": _openapi}[language](facts, path, text, mid)
    except (ValueError, KeyError, TypeError, AttributeError, IndexError) as exc:
        facts.errors.append(f"{language} extraction incomplete: {type(exc).__name__}: {exc}")
    return facts


# ------------------------------------------------------------------ shared helpers
def _line(text: str, pattern: str, default: int = 1) -> int:
    m = re.search(pattern, text, re.M)
    return text.count("\n", 0, m.start()) + 1 if m else default


def _env(facts: Facts, owner: str, keys: list[str], line: int | None, source: str) -> None:
    for key in keys:
        if not key or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]*", key):
            continue
        cid = f"config:{key}"
        facts.node(cid, "config", key, ev.MANIFEST, owner_file=None, attrs={"source": "environment"})
        facts.edge(owner, cid, "configures", ev.MANIFEST, line=line, attrs={"declared_in": source})


RUNNERS = {"uvicorn", "gunicorn", "hypercorn", "daphne", "celery", "flask", "streamlit", "dramatiq", "rq", "faust"}


def command_targets(cmd: Any) -> list[tuple[str, str]]:
    """What a container/CI command starts: [(link target, how)], e.g. ("MODULE:app.main", "python -m")."""
    if cmd is None:
        return []
    if isinstance(cmd, list):
        tokens = [str(t) for t in cmd]
    else:
        s = str(cmd).strip()
        if s.startswith("["):
            tokens = re.findall(r'"([^"]*)"', s)
        else:
            try:
                tokens = shlex.split(s, comments=True)
            except ValueError:
                tokens = s.split()
    out: list[tuple[str, str]] = []
    for i, tok in enumerate(tokens):
        base = PurePosixPath(tok).name
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        if base.startswith("python") and nxt == "-m" and i + 2 < len(tokens):
            mod = tokens[i + 2]
            out.append((f"MODULE:{mod}", "python -m"))
        elif base.startswith("python") and nxt.endswith(".py"):
            out.append((f"FILE:{nxt.lstrip('./')}", "python"))
        elif base in RUNNERS:
            rest = tokens[i + 1:]
            for j, arg in enumerate(rest):
                app = rest[j + 1] if arg in ("-A", "--app") and j + 1 < len(rest) else None
                m = re.match(r"^(?:-A|--app=)([\w.]+)$", arg) or re.match(r"^([\w.]+):[\w.()]+$", arg)
                if app or m:
                    out.append((f"MODULE:{(app or m.group(1)).split(':')[0]}", base))
                    break
        elif base in ("node", "bun", "deno", "ts-node", "tsx") and nxt and not nxt.startswith("-"):
            out.append((f"FILE:{nxt.lstrip('./')}", base))
        elif tok.endswith((".py", ".js", ".ts", ".sh", ".rb")) and i == 0:
            out.append((f"FILE:{tok.lstrip('./')}", "exec"))
        if base in ("sh", "bash") and nxt == "-c" and i + 2 < len(tokens):
            out.extend(command_targets(tokens[i + 2]))
            break
        if tok in ("&&", ";", "||") and i + 1 < len(tokens):
            out.extend(command_targets(tokens[i + 1:]))
            break
    return out


def _runs(facts: Facts, owner: str, cmd: Any, line: int | None, base_dir: str = "") -> None:
    for target, how in command_targets(cmd):
        if target.startswith("FILE:") and base_dir:
            target = f"FILE:{PurePosixPath(base_dir, target[5:]).as_posix()}"
        facts.ref(owner, target, "link", line, edge="runs", evidence=ev.MANIFEST, attrs={"via": how})


# ------------------------------------------------------------------ Dockerfile
def _dockerfile(facts: Facts, path: str, text: str, mid: str) -> None:
    cid = f"container:{path}"
    stages, ports, env_keys, cmd, entry, line_cmd = [], [], [], None, None, None
    joined = re.sub(r"\\\n", " ", text)
    for m in re.finditer(r"^\s*(\w+)\s+(.*)$", joined, re.M):
        instr, arg = m.group(1).upper(), m.group(2).strip()
        if instr == "FROM":
            stages.append(arg.split()[0])
        elif instr == "EXPOSE":
            ports += re.findall(r"\d+", arg)
        elif instr in ("ENV", "ARG"):
            if "=" in arg:
                env_keys += re.findall(r"([A-Za-z_][A-Za-z0-9_]*)=", arg)
            elif instr == "ENV":
                env_keys.append(arg.split()[0])
        elif instr == "CMD":
            cmd, line_cmd = arg, _line(text, r"^\s*CMD\b")
        elif instr == "ENTRYPOINT":
            entry, line_cmd = arg, _line(text, r"^\s*ENTRYPOINT\b")
    facts.node(cid, "container_image", PurePosixPath(path).parent.name or PurePosixPath(path).name, ev.MANIFEST,
               file=path, line_start=1, line_end=text.count("\n") + 1,
               attrs={"base_images": stages, "ports": ports, "env": env_keys, "cmd": cmd, "entrypoint": entry,
                      "multi_stage": len(stages) > 1})
    facts.edge(mid, cid, "defines", ev.MANIFEST)
    _env(facts, cid, env_keys, _line(text, r"^\s*ENV\b"), path)
    for command in (entry, cmd):
        _runs(facts, cid, command, line_cmd)


# ------------------------------------------------------------------ Docker Compose
def _compose(facts: Facts, path: str, text: str, mid: str) -> None:
    docs = load(text)
    data = docs[0] if docs and isinstance(docs[0], dict) else {}
    services = data.get("services") or {}
    if not services and "version" not in data:      # compose v1: services at top level
        services = {k: v for k, v in data.items() if isinstance(v, dict) and ("image" in v or "build" in v)}
    base = PurePosixPath(path).parent
    for name, svc in services.items():
        if not isinstance(svc, dict):
            continue
        sid = f"service:{name}"
        line = _line(text, rf"^\s+{re.escape(str(name))}\s*:")
        env = svc.get("environment") or {}
        keys = list(env) if isinstance(env, dict) else [str(e).split("=", 1)[0] for e in env]
        ports = [str(p) for p in (svc.get("ports") or [])]
        build = svc.get("build")
        facts.node(sid, "service", str(name), ev.MANIFEST, file=path, line_start=line, owner_file=None,
                   attrs={"source": "docker-compose", "image": svc.get("image"), "ports": ports,
                          "env": keys, "build": build if isinstance(build, str) else (build or {}).get("context"),
                          "depends_on": list(svc.get("depends_on") or []), "replicas":
                          ((svc.get("deploy") or {}).get("replicas") if isinstance(svc.get("deploy"), dict) else None)})
        facts.edge(mid, sid, "defines", ev.MANIFEST, line=line)
        for dep in svc.get("depends_on") or []:
            facts.ref(sid, f"ID:service:{dep}", "link", line, edge="depends_on", evidence=ev.MANIFEST)
        _env(facts, sid, keys, line, path)
        for env_file in ([svc["env_file"]] if isinstance(svc.get("env_file"), str) else svc.get("env_file") or []):
            if isinstance(env_file, str):
                facts.ref(sid, f"FILE:{(base / env_file).as_posix().lstrip('./')}", "link", line,
                          edge="configures", evidence=ev.MANIFEST)
        if build is not None:
            ctx_dir = build if isinstance(build, str) else (build or {}).get("context", ".")
            dockerfile = "Dockerfile" if isinstance(build, str) else (build or {}).get("dockerfile", "Dockerfile")
            target = (base / str(ctx_dir) / str(dockerfile)).as_posix()
            target = re.sub(r"^(\./)+", "", re.sub(r"/\./", "/", target))
            facts.ref(sid, f"ID:container:{target}", "link", line, edge="builds", evidence=ev.MANIFEST)
        if svc.get("command") or svc.get("entrypoint"):
            _runs(facts, sid, svc.get("entrypoint") or svc.get("command"), line)
            if svc.get("entrypoint") and svc.get("command"):
                _runs(facts, sid, svc.get("command"), line)


# ------------------------------------------------------------------ Kubernetes
def _kubernetes(facts: Facts, path: str, text: str, mid: str) -> None:
    for doc in load(text):
        if not isinstance(doc, dict) or not doc.get("kind"):
            continue
        kind = str(doc["kind"])
        meta = doc.get("metadata") or {}
        name = str(meta.get("name") or "unnamed")
        ns = str(meta.get("namespace") or "")
        line = getattr(doc, "line", None) or _line(text, rf"^\s*name\s*:\s*{re.escape(name)}\s*$")
        spec = doc.get("spec") or {}
        if kind in WORKLOADS:
            _workload(facts, path, mid, kind, name, ns, spec, meta, line)
        elif kind == "Service":
            sid = f"service:{name}"
            selector = spec.get("selector") or {}
            facts.node(sid, "service", name, ev.MANIFEST, file=path, line_start=line, owner_file=None,
                       attrs={"source": "kubernetes", "namespace": ns, "type": spec.get("type", "ClusterIP"),
                              "ports": [p.get("port") for p in spec.get("ports") or [] if isinstance(p, dict)],
                              "selector": selector})
            facts.edge(mid, sid, "defines", ev.MANIFEST, line=line)
            if isinstance(selector, dict) and selector:
                facts.ref(sid, "LABELS:" + ",".join(f"{k}={v}" for k, v in sorted(selector.items())), "link", line,
                          edge="routes_to", evidence=ev.MANIFEST)
        elif kind in ("Ingress", "HTTPRoute", "IngressRoute"):
            iid = f"k8s:{ns}/{kind.lower()}/{name}"
            hosts, backends = [], set()
            for rule in spec.get("rules") or []:
                if isinstance(rule, dict):
                    hosts.append(rule.get("host"))
                    for p in ((rule.get("http") or {}).get("paths") or []):
                        svc = ((p.get("backend") or {}).get("service") or {}).get("name") or \
                            (p.get("backend") or {}).get("serviceName")
                        if svc:
                            backends.add(str(svc))
                    for ref in rule.get("backendRefs") or []:
                        if isinstance(ref, dict) and ref.get("name"):
                            backends.add(str(ref["name"]))
            facts.node(iid, "ingress", name, ev.MANIFEST, file=path, line_start=line,
                       attrs={"kind": kind, "namespace": ns, "hosts": [h for h in hosts if h]})
            facts.edge(mid, iid, "defines", ev.MANIFEST, line=line)
            for b in sorted(backends):
                facts.ref(iid, f"ID:service:{b}", "link", line, edge="routes_to", evidence=ev.MANIFEST)
        elif kind in ("ConfigMap", "Secret"):
            cid = f"k8s:{ns}/{kind.lower()}/{name}"
            keys = sorted((doc.get("data") or {}) | (doc.get("stringData") or {}))
            facts.node(cid, "config_source", name, ev.MANIFEST, file=path, line_start=line,
                       attrs={"kind": kind, "namespace": ns, "keys": keys[:200]})
            facts.edge(mid, cid, "defines", ev.MANIFEST, line=line)
            _env(facts, cid, [k for k in keys if re.fullmatch(r"[A-Z_][A-Z0-9_]*", k)], line, path)
        elif kind == "HorizontalPodAutoscaler":
            target = (spec.get("scaleTargetRef") or {}).get("name")
            if target:
                facts.ref(mid, f"ID:k8s:{ns}/workload/{target}", "link", line, edge="scales", evidence=ev.MANIFEST,
                          attrs={"min": spec.get("minReplicas"), "max": spec.get("maxReplicas")})


def _workload(facts: Facts, path: str, mid: str, kind: str, name: str, ns: str, spec: dict[str, Any],
              meta: dict[str, Any], line: int | None) -> None:
    wid = f"k8s:{ns}/workload/{name}"
    pod = spec
    if kind == "CronJob":
        pod = ((spec.get("jobTemplate") or {}).get("spec") or {})
    template = pod.get("template") or {}
    labels = dict((template.get("metadata") or {}).get("labels") or meta.get("labels") or {})
    pod_spec = template.get("spec") or (spec if kind == "Pod" else {})
    containers = [c for c in (pod_spec.get("containers") or []) + (pod_spec.get("initContainers") or [])
                  if isinstance(c, dict)]
    facts.node(wid, "deployment", name, ev.MANIFEST, file=path, line_start=line,
               attrs={"kind": kind, "namespace": ns, "replicas": spec.get("replicas"), "labels": labels,
                      "schedule": spec.get("schedule"), "images": [c.get("image") for c in containers],
                      "ports": [p.get("containerPort") for c in containers for p in c.get("ports") or []
                                if isinstance(p, dict)]})
    facts.edge(mid, wid, "defines", ev.MANIFEST, line=line)
    for c in containers:
        keys = [e.get("name") for e in c.get("env") or [] if isinstance(e, dict)]
        _env(facts, wid, keys, line, path)
        for src in c.get("envFrom") or []:
            if not isinstance(src, dict):
                continue
            for key, k8s_kind in (("configMapRef", "configmap"), ("secretRef", "secret")):
                ref_name = (src.get(key) or {}).get("name")
                if ref_name:
                    facts.ref(wid, f"ID:k8s:{ns}/{k8s_kind}/{ref_name}", "link", line, edge="depends_on",
                              evidence=ev.MANIFEST)
        for e in c.get("env") or []:
            vf = (e or {}).get("valueFrom") if isinstance(e, dict) else None
            for key, k8s_kind in (("configMapKeyRef", "configmap"), ("secretKeyRef", "secret")):
                ref_name = ((vf or {}).get(key) or {}).get("name")
                if ref_name:
                    facts.ref(wid, f"ID:k8s:{ns}/{k8s_kind}/{ref_name}", "link", line, edge="depends_on",
                              evidence=ev.MANIFEST)
        command = (c.get("command") or []) + (c.get("args") or []) if isinstance(c.get("command"), list) or \
            isinstance(c.get("args"), list) else c.get("command")
        _runs(facts, wid, command, line)


# ------------------------------------------------------------------ Terraform (HCL)
HCL_REF = re.compile(r"(?<![\w.\"])(data\.[a-z][a-z0-9_]*\.[\w-]+|module\.[\w-]+|var\.[\w-]+|local\.[\w-]+|"
                     r"[a-z][a-z0-9]*_[a-z0-9_]+\.[A-Za-z_][\w-]*)")


def hcl_blocks(text: str) -> list[dict[str, Any]]:
    """Top-level HCL blocks: {type, labels, body, line}. Strings, comments and heredocs are skipped."""
    blocks, i, n = [], 0, len(text)
    clean = _hcl_mask(text)
    head = re.compile(r'^[ \t]*([A-Za-z_][\w-]*)((?:[ \t]+"[^"\n]*"|[ \t]+[A-Za-z_][\w-]*)*)[ \t]*\{', re.M)
    while i < n:
        m = head.search(clean, i)
        if not m:
            break
        depth, j = 1, m.end()
        while j < n and depth:
            if clean[j] == "{":
                depth += 1
            elif clean[j] == "}":
                depth -= 1
            j += 1
        raw = text[m.start(2):m.end(2)]          # labels from the original text: the mask blanks strings
        labels = [lbl.strip('"') for lbl in re.findall(r'"[^"\n]*"|[A-Za-z_][\w-]*', raw)]
        blocks.append({"type": m.group(1), "labels": labels, "body": text[m.end():j - 1],
                       "line": text.count("\n", 0, m.start()) + 1, "end": text.count("\n", 0, j) + 1})
        i = j
    return blocks


def _hcl_mask(text: str) -> str:
    """Same length as `text`, with comments, strings and heredoc bodies blanked (so braces inside them don't count)."""
    out = list(text)
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c == "#" or text.startswith("//", i):
            j = text.find("\n", i)
            j = n if j < 0 else j
            out[i:j] = " " * (j - i)
            i = j
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out[i:j] = [ch if ch == "\n" else " " for ch in text[i:j]]
            i = j
        elif text.startswith("<<", i) and re.match(r"<<-?([A-Z_]+)\n", text[i:]):
            tag = re.match(r"<<-?([A-Z_]+)\n", text[i:]).group(1)
            m = re.search(rf"^\s*{tag}\s*$", text[i:], re.M)
            j = i + (m.end() if m else n - i)
            out[i:j] = [ch if ch == "\n" else " " for ch in text[i:j]]
            i = j
        elif c == '"':
            j = i + 1
            while j < n and text[j] != '"' and text[j] != "\n":
                j += 2 if text[j] == "\\" else 1
            out[i + 1:j] = " " * max(0, j - i - 1)
            i = j + 1
        else:
            i += 1
    return "".join(out)


def _terraform(facts: Facts, path: str, text: str, mid: str) -> None:
    if path.endswith(".tfvars"):
        for m in re.finditer(r"^\s*([\w-]+)\s*=", text, re.M):
            facts.ref(mid, f"ID:tf:{_tfdir(path)}:var.{m.group(1)}", "link", text.count("\n", 0, m.start()) + 1,
                      edge="configures", evidence=ev.MANIFEST)
        return
    d = _tfdir(path)
    for b in hcl_blocks(text):
        typ, labels, body, line = b["type"], b["labels"], b["body"], b["line"]
        if typ == "resource" and len(labels) >= 2:
            addr, kind = f"{labels[0]}.{labels[1]}", "cloud_resource"
            attrs = {"provider": labels[0].split("_")[0], "type": labels[0], "category": category(labels[0])}
        elif typ == "data" and len(labels) >= 2:
            addr, kind = f"data.{labels[0]}.{labels[1]}", "cloud_resource"
            attrs = {"provider": labels[0].split("_")[0], "type": labels[0], "category": category(labels[0]),
                     "data_source": True}
        elif typ == "module" and labels:
            addr, kind = f"module.{labels[0]}", "iac_module"
            src = re.search(r'^\s*source\s*=\s*"([^"]+)"', body, re.M)
            attrs = {"source": src.group(1) if src else None}
        elif typ == "variable" and labels:
            addr, kind = f"var.{labels[0]}", "iac_variable"
            desc = re.search(r'^\s*description\s*=\s*"([^"]*)"', body, re.M)
            attrs = {"description": desc.group(1) if desc else None}
        elif typ == "output" and labels:
            addr, kind = f"output.{labels[0]}", "iac_variable"
            attrs = {"output": True}
        elif typ == "locals":
            for m in re.finditer(r"^\s*([\w-]+)\s*=", body, re.M):
                lid = f"tf:{d}:local.{m.group(1)}"
                facts.node(lid, "iac_variable", f"local.{m.group(1)}", ev.MANIFEST, file=path, line_start=line,
                           attrs={"local": True})
                facts.edge(mid, lid, "defines", ev.MANIFEST, line=line)
            continue
        elif typ == "provider" and labels:
            facts.nodes[0].setdefault("attrs", {}).setdefault("providers", []).append(labels[0])
            continue
        else:
            continue
        nid = f"tf:{d}:{addr}"
        facts.node(nid, kind, addr, ev.MANIFEST, file=path, line_start=line, line_end=b["end"], attrs=attrs)
        facts.edge(mid, nid, "defines", ev.MANIFEST, line=line)
        seen = set()
        for r in HCL_REF.finditer(_hcl_mask(body)):
            target = r.group(1)
            parts = target.split(".")
            if parts[0] in ("data",):
                target = ".".join(parts[:3])
            elif parts[0] in ("module", "var", "local"):
                target = ".".join(parts[:2])
            else:
                target = ".".join(parts[:2])
            if target == addr or target in seen:
                continue
            seen.add(target)
            facts.ref(nid, f"ID:tf:{d}:{target}", "link", line, edge="depends_on", evidence=ev.MANIFEST)
        keys = _tf_env_keys(body)
        _env(facts, nid, keys, line, path)
        if typ == "module" and attrs.get("source", "") and str(attrs["source"]).startswith("."):
            target_dir = PurePosixPath(d, attrs["source"]).as_posix()
            facts.ref(nid, f"ID:manifest:{_normpath(target_dir)}/main.tf", "link", line, edge="uses",
                      evidence=ev.MANIFEST)


def _tf_env_keys(body: str) -> list[str]:
    keys = []
    for m in re.finditer(r"(?:variables|environment_variables|app_settings|env_vars|environment)\s*=?\s*\{([^{}]*)\}",
                         body):
        keys += re.findall(r'^\s*"?([A-Z_][A-Z0-9_]*)"?\s*=', m.group(1), re.M)
    keys += re.findall(r'env\s*\{\s*name\s*=\s*"([A-Z_][A-Z0-9_]*)"', body)
    return keys


def _tfdir(path: str) -> str:
    parent = PurePosixPath(path).parent.as_posix()
    return "" if parent == "." else parent


def _normpath(p: str) -> str:
    parts: list[str] = []
    for seg in p.split("/"):
        if seg in ("", "."):
            continue
        if seg == ".." and parts:
            parts.pop()
        else:
            parts.append(seg)
    return "/".join(parts)


# ------------------------------------------------------------------ CI workflows
def _ci(facts: Facts, path: str, text: str, mid: str) -> None:
    docs = load(text)
    data = docs[0] if docs and isinstance(docs[0], dict) else {}
    name = str(data.get("name") or PurePosixPath(path).stem)
    pid = f"ci:{path}"
    triggers = data.get("on", data.get(True))           # PyYAML reads `on:` as True
    trig = sorted(triggers) if isinstance(triggers, dict) else ([triggers] if isinstance(triggers, str)
                                                                 else list(triggers or []))
    jobs = data.get("jobs") or {}
    if not jobs and path.endswith(".gitlab-ci.yml"):
        jobs = {k: v for k, v in data.items() if isinstance(v, dict) and "script" in v}
    facts.node(pid, "pipeline", name, ev.MANIFEST, file=path, line_start=1,
               attrs={"triggers": [str(t) for t in trig], "jobs": sorted(map(str, jobs))})
    facts.edge(mid, pid, "defines", ev.MANIFEST)
    for job_name, job in jobs.items():
        if not isinstance(job, dict):
            continue
        line = _line(text, rf"^\s+{re.escape(str(job_name))}\s*:")
        commands = [s.get("run") for s in job.get("steps") or [] if isinstance(s, dict) and s.get("run")]
        commands += job.get("script") or [] if isinstance(job.get("script"), list) else []
        env = job.get("env") or {}
        _env(facts, pid, list(env) if isinstance(env, dict) else [], line, path)
        for cmd in commands:
            for one in str(cmd).splitlines():
                _runs(facts, pid, one, line)


# ------------------------------------------------------------------ OpenAPI
METHODS = ("get", "put", "post", "delete", "patch", "head", "options")


def _openapi(facts: Facts, path: str, text: str, mid: str) -> None:
    if path.endswith(".json"):
        import json
        data = json.loads(text)
    else:
        docs = load(text)
        data = docs[0] if docs else {}
    info = data.get("info") or {}
    facts.nodes[0]["attrs"].update({"title": info.get("title"), "version": info.get("version"),
                                    "openapi": data.get("openapi") or data.get("swagger")})
    for route, ops in (data.get("paths") or {}).items():
        if not isinstance(ops, dict):
            continue
        for method, op in ops.items():
            if method.lower() not in METHODS or not isinstance(op, dict):
                continue
            label = f"{method.upper()} {route}"
            oid = f"apispec:{path}#{label}"
            line = _line(text, rf"^\s*[\"']?{re.escape(str(route))}[\"']?\s*:")
            facts.node(oid, "api_operation", label, ev.MANIFEST, file=path, line_start=line,
                       text=str(op.get("summary") or op.get("description") or "")[:500],
                       attrs={"method": method.upper(), "path": route, "operation_id": op.get("operationId"),
                              "tags": op.get("tags") or [], "deprecated": bool(op.get("deprecated")),
                              "security": bool(op.get("security", data.get("security")))})
            facts.edge(mid, oid, "defines", ev.MANIFEST, line=line)
            facts.ref(oid, f"API:{method.upper()} {route}", "link", line, edge="specifies", evidence=ev.MANIFEST)
            if op.get("operationId"):
                facts.ref(oid, f"SYMBOL:{op['operationId']}", "link", line, edge="specifies",
                          evidence=ev.NAME_MATCH, confidence=0.5)
    schemas = ((data.get("components") or {}).get("schemas") or data.get("definitions") or {})
    for sname, schema in schemas.items():
        sid = f"apispec:{path}#schema:{sname}"
        props = sorted((schema or {}).get("properties") or {}) if isinstance(schema, dict) else []
        facts.node(sid, "schema_type", str(sname), ev.MANIFEST, file=path,
                   line_start=_line(text, rf"^\s*[\"']?{re.escape(str(sname))}[\"']?\s*:"),
                   attrs={"fields": props[:100], "format": "openapi"})
        facts.edge(mid, sid, "defines", ev.MANIFEST)
