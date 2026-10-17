"""Workspace layout and configuration (.promapkin/ inside the analyzed repository)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from promapkin import gitutil
from promapkin.store import Store

DIRNAME = ".promapkin"


@dataclass
class Config:
    # Agent provider: "none" (deterministic specialists only), "command", or "anthropic".
    provider: str = "none"
    provider_model: str = ""
    provider_command: str = ""
    # Remote providers receive source excerpts; this must be enabled explicitly.
    allow_remote: bool = False
    include: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)
    default_token_budget: int = 4000
    # Classification / embeddings. ml_mode: "ask" (prompt before downloading a model), "auto"
    # (download without asking), "never" (heuristic + built-in classifiers only, nothing downloaded).
    ml_mode: str = "ask"
    ml_backend: str = "auto"          # auto | fastembed | sentence-transformers | builtin
    ml_model: str = ""                # override the embedding model name
    ml_consent: str = ""              # "", "granted" or "declined" (remembered answer to the download prompt)
    layers: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "Config":
        if not path.exists():
            return cls()
        data = json.loads(path.read_text())
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")


class Workspace:
    """A repository plus its Promapkin state."""

    def __init__(self, root: Path | str):
        self.root = gitutil.find_root(Path(root).resolve())
        self.dir = self.root / DIRNAME
        self.config_path = self.dir / "config.json"
        self.db_path = self.dir / "index.db"
        self.blueprint_dir = self.dir / "blueprints"
        self._store: Store | None = None
        self.config = Config.load(self.config_path)

    @property
    def initialized(self) -> bool:
        return self.config_path.exists()

    def init(self) -> bool:
        created = not self.initialized
        self.dir.mkdir(exist_ok=True)
        self.blueprint_dir.mkdir(exist_ok=True)
        if created:
            self.config.save(self.config_path)
        gi = self.dir / ".gitignore"
        if not gi.exists():
            # The index is a cache; blueprints and handoff exports may be committed.
            gi.write_text("index.db\nindex.db-*\negress.log\n")
        return created

    @property
    def store(self) -> Store:
        if self._store is None:
            if not self.dir.exists():
                self.init()
            self._store = Store(self.db_path)
        return self._store

    def revision(self) -> str:
        return gitutil.revision_label(self.root)

    def close(self) -> None:
        if self._store is not None:
            self._store.close()
            self._store = None

    def info(self) -> dict[str, Any]:
        return {"root": str(self.root), "initialized": self.initialized, "state_dir": str(self.dir)}
