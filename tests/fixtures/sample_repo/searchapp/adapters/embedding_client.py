"""Adapter around the external embedding HTTP service."""
import requests


class EmbeddingAdapter:
    """Adapts the remote embedding API to the local embedder interface."""

    def __init__(self, base_url: str):
        self.base_url = base_url

    def embed(self, text: str) -> list:
        response = requests.post(f"{self.base_url}/embed", json={"text": text}, timeout=10)
        response.raise_for_status()
        return response.json()["vector"]
