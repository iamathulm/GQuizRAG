from pathlib import Path

from placerag.embeddings import EmbeddingModel
from placerag.models import Chunk, SearchResult
from placerag.repository import DocumentRepository


class SearchEngine:
    def __init__(self, vectorstore_dir: Path):
        self.embedding_model = EmbeddingModel()

        self.repository = DocumentRepository(vectorstore_dir)
        self.repository.load_all()

    def search(
        self,
        question: str,
        k: int = 5,
        documents: list[str] | None = None,
    ) -> list[SearchResult]:
        query_embedding = self.embedding_model.embed_query(question)

        return self.repository.search(
            query_embedding,
            k=k,
            documents=documents,
        )