"""Application service coordinating storage and ranking."""
from searchapp.domain.ranking import Ranker, select_candidates
from searchapp.storage.index_repository import IndexRepository


class SearchService:
    """Executes searches against the index."""

    def __init__(self, repository: IndexRepository, ranker: Ranker):
        self.repository = repository
        self.ranker = ranker

    def search(self, query: str, limit: int = 10) -> list:
        documents = self.repository.find_all()
        candidates = select_candidates(documents, limit)
        scored = [(self.ranker.score(query, d), d) for d in candidates]
        scored.sort(key=lambda pair: -pair[0])
        return [d for _, d in scored[:limit]]

    def index_document(self, doc_id: str, text: str) -> None:
        # TODO: batch inserts once the repository supports transactions.
        self.repository.add(doc_id, text)
