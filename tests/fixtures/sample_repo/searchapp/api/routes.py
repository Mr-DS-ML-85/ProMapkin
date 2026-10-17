"""HTTP API for search."""
from fastapi import Depends, FastAPI

from searchapp.config import load_settings
from searchapp.domain.ranking import BM25Ranker
from searchapp.services.search_service import SearchService
from searchapp.storage.index_repository import IndexRepository

app = FastAPI()


def require_token(token: str = "") -> str:
    return token


def build_service() -> SearchService:
    settings = load_settings()
    return SearchService(IndexRepository(settings.db_path), BM25Ranker())


@app.get("/search")
def search_endpoint(q: str, limit: int = 10, token: str = Depends(require_token)) -> dict:
    """Search indexed documents."""
    service = build_service()
    return {"results": service.search(q, limit)}


@app.post("/documents")
def add_document(doc_id: str, text: str) -> dict:
    service = build_service()
    service.index_document(doc_id, text)
    return {"ok": True}
