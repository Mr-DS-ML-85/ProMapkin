"""Model availability, consent-gated downloads and embedder selection.

Resolution order:
1. A neural embedding model that is installed *and* cached locally is used directly.
2. If none is cached, Promapkin asks before downloading one (``ml_mode="ask"``), downloads
   without asking (``"auto"``), or never downloads (``"never"``). Non-interactive callers
   such as the MCP server never prompt: they fall back and report that a download is available.
3. When the user declines (or no model is available), heuristic rules plus the built-in
   pure-Python classifier/embedder are used. Nothing is downloaded and no source leaves the machine.

Downloads fetch model weights only; repository content is never uploaded.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from promapkin.ml.embedders import (
    Embedder,
    FastEmbedEmbedder,
    HashingEmbedder,
    SentenceTransformerEmbedder,
)

BACKENDS = {
    "fastembed": ("fastembed", FastEmbedEmbedder),
    "sentence-transformers": ("sentence_transformers", SentenceTransformerEmbedder),
}
PIP_PACKAGES = {"fastembed": "fastembed", "sentence-transformers": "sentence-transformers"}


def model_dir() -> Path:
    base = os.environ.get("PROMAPKIN_MODEL_DIR") or os.path.join(
        os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")), "promapkin", "models")
    return Path(base)


@dataclass
class Decision:
    embedder: Embedder
    tier: str              # "neural" or "builtin"
    reason: str
    download_available: bool = False


class ModelManager:
    def __init__(self, mode: str = "ask", backend: str = "auto", model: str = "", consent: str = "",
                 on_consent: Callable[[str], None] | None = None):
        self.mode = os.environ.get("PROMAPKIN_ML_MODE", mode or "ask")
        self.backend = os.environ.get("PROMAPKIN_ML_BACKEND", backend or "auto")
        self.model = model
        self.consent = consent
        self.on_consent = on_consent
        self.dir = model_dir()

    # ---------------------------------------------------------------- inspection
    @staticmethod
    def installed(backend: str) -> bool:
        module = BACKENDS[backend][0]
        return importlib.util.find_spec(module) is not None

    def _marker(self, backend: str) -> Path:
        cls = BACKENDS[backend][1]
        name = (self.model or cls.default_model).replace("/", "__")
        return self.dir / f"{backend}__{name}.ready"

    def cached(self, backend: str) -> bool:
        return self._marker(backend).exists()

    def candidates(self) -> list[str]:
        if self.backend in ("builtin", "heuristic"):
            return []
        if self.backend in BACKENDS:
            return [self.backend]
        return list(BACKENDS)

    def status(self) -> dict:
        rows = []
        for b in BACKENDS:
            cls = BACKENDS[b][1]
            rows.append({"backend": b, "model": self.model or cls.default_model, "installed": self.installed(b),
                         "cached": self.cached(b)})
        return {"mode": self.mode, "backend": self.backend, "consent": self.consent or None,
                "model_dir": str(self.dir), "backends": rows,
                "builtin": {"embedder": HashingEmbedder().name, "classifier": "naive-bayes + heuristic rules"}}

    # ---------------------------------------------------------------- selection
    def resolve(self, interactive: bool = False, ask: Callable[[str], bool] | None = None) -> Decision:
        """Pick the best embedder permitted by policy, prompting for a download only if allowed."""
        candidates = self.candidates()
        for b in candidates:
            if self.installed(b) and self.cached(b):
                emb = self._load(b)
                if emb:
                    return Decision(emb, "neural", f"{b} model cached locally")
        if not candidates or self.mode == "never" or os.environ.get("PROMAPKIN_OFFLINE"):
            return Decision(HashingEmbedder(), "builtin", "ML downloads disabled; heuristic + built-in classifier")
        if self.consent == "declined":
            return Decision(HashingEmbedder(), "builtin", "model download declined; heuristic + built-in classifier",
                            download_available=True)
        want = self.mode == "auto" or self.consent == "granted"
        if not want and interactive and ask is not None:
            b = candidates[0]
            what = (f"the {b} package and a small embedding model (~70-130 MB)" if not self.installed(b)
                    else f"a small {b} embedding model (~70-130 MB)")
            want = ask(f"Promapkin can download {what} for ML classification and semantic search.\n"
                       f"Only model weights are downloaded; your source code is not uploaded. Download now?")
            self.consent = "granted" if want else "declined"
            if self.on_consent:
                self.on_consent(self.consent)
        if want:
            for b in candidates:
                ok, msg = self.download(b, install_package=True)
                if ok:
                    emb = self._load(b)
                    if emb:
                        return Decision(emb, "neural", f"downloaded {b} model")
            return Decision(HashingEmbedder(), "builtin", "model download failed; heuristic + built-in classifier",
                            download_available=True)
        return Decision(HashingEmbedder(), "builtin",
                        "no local model; run `promapkin models download` to enable neural classification",
                        download_available=True)

    def _load(self, backend: str) -> Embedder | None:
        try:
            cls = BACKENDS[backend][1]
            return cls(self.model or None, str(self.dir))
        except Exception:  # noqa: BLE001 - any model failure degrades to the built-in tier
            return None

    # ---------------------------------------------------------------- download / removal
    def download(self, backend: str | None = None, install_package: bool = False) -> tuple[bool, str]:
        backend = backend or (self.candidates() or ["fastembed"])[0]
        if backend not in BACKENDS:
            return False, f"unknown backend {backend}"
        if not self.installed(backend):
            if not install_package:
                return False, f"python package for {backend} is not installed (pip install {PIP_PACKAGES[backend]})"
            proc = subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", PIP_PACKAGES[backend]],
                                  capture_output=True, text=True, check=False)
            importlib.invalidate_caches()
            if proc.returncode != 0 or not self.installed(backend):
                return False, f"pip install {PIP_PACKAGES[backend]} failed: {proc.stderr.strip()[-400:]}"
        self.dir.mkdir(parents=True, exist_ok=True)
        try:
            emb = BACKENDS[backend][1](self.model or None, str(self.dir))
        except Exception as exc:  # noqa: BLE001
            return False, f"model download failed: {exc}"
        self._marker(backend).write_text(f"{emb.name} dim={emb.dim}\n")
        return True, f"{emb.name} ready (dim {emb.dim}) in {self.dir}"

    def remove(self) -> list[str]:
        removed = []
        if self.dir.exists():
            for child in self.dir.iterdir():
                removed.append(child.name)
                shutil.rmtree(child) if child.is_dir() else child.unlink()
        return removed
