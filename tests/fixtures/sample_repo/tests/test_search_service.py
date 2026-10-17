from searchapp.domain.ranking import BM25Ranker
from searchapp.services.search_service import SearchService


class FakeRepo:
    def find_all(self):
        return [{"id": "a", "text": "vector search"}, {"id": "b", "text": "cooking"}]


def test_search_ranks_matching_documents_first():
    service = SearchService(FakeRepo(), BM25Ranker())
    assert service.search("search")[0]["id"] == "a"
