"""Heuristic classification of calls that touch external resources (data-flow sinks and sources)."""

from __future__ import annotations

import re

# (resource, pattern over the dotted call expression or imported module)
_CALL_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("database", re.compile(r"(^|\.)(execute|executemany|executescript|cursor|commit|rollback|query|"
                            r"bulk_create|bulk_insert|find_one|insert_one|insert_many|update_one|"
                            r"delete_one|aggregate)$|^(sqlite3|psycopg2?|pymysql|sqlalchemy|pymongo|"
                            r"mongoose|knex|prisma|sequelize|typeorm|pg|mysql2?)\b|\.objects\.|^(db|session)\.")),
    ("http", re.compile(r"^(requests|httpx|aiohttp|urllib\.request|urllib3|axios|got|superagent)\.|"
                        r"(^|\.)(urlopen|fetch)$")),
    ("queue", re.compile(r"^(kafka|pika|celery|nats|amqp|bullmq|sqs)\b|(^|\.)(publish|enqueue|send_message|"
                         r"produce|delay|apply_async)$")),
    ("cache", re.compile(r"^(redis|memcache|pymemcache|ioredis)\b|^cache\.")),
    ("filesystem", re.compile(r"^(open|io\.open|shutil\.\w+|os\.(remove|unlink|makedirs|mkdir|rename|listdir)|"
                              r"fs\.\w+|json\.(load|dump)|pickle\.(load|dump)|csv\.(reader|writer))$|"
                              r"(^|\.)(read_text|write_text|read_bytes|write_bytes|readFile|writeFile|"
                              r"readFileSync|writeFileSync)$")),
    ("subprocess", re.compile(r"^(subprocess\.\w+|os\.system|os\.popen|child_process\.\w+|exec|execSync|spawn)$")),
]

_WRITE_VERBS = re.compile(
    r"(write|save|insert|add|commit|put|post|patch|publish|send|update|delete|remove|unlink|dump|create|"
    r"enqueue|produce|set|push|delay|apply_async|executemany|executescript|mkdir|makedirs|rename)",
    re.IGNORECASE,
)


def classify_call(dotted: str, module_of_root: str | None = None) -> tuple[str, str] | None:
    """Return (resource, 'reads'|'writes'|'invokes') for a call expression, or None."""
    probe = dotted
    if module_of_root:
        head, _, rest = dotted.partition(".")
        probe = f"{module_of_root}.{rest}" if rest else module_of_root
    for resource, pattern in _CALL_RULES:
        if pattern.search(probe) or (probe != dotted and pattern.search(dotted)):
            last = dotted.rsplit(".", 1)[-1]
            if resource == "subprocess":
                return resource, "invokes"
            if resource == "http":
                verb = "writes" if last.lower() in {"post", "put", "patch", "delete"} else "reads"
                return resource, verb
            if dotted.endswith("open") and resource == "filesystem":
                return resource, "reads"
            return resource, "writes" if _WRITE_VERBS.search(last) else "reads"
    return None


AUTH_HINT = re.compile(r"(login_required|auth|permission|jwt|token|requires_|guard|protect|oauth|api_key|"
                       r"current_user|is_authenticated)", re.IGNORECASE)
