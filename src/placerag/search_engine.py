from pathlib import Path

from placerag.config import config
from placerag.embeddings import EmbeddingModel
from placerag.repository import DocumentRepository


class SearchEngine:
    def __init__(self, vectorstore_dir: Path):
        print("SearchEngine initialized")
        self.embedding_model = EmbeddingModel()

        self.repository = DocumentRepository(vectorstore_dir)
        self.repository.load_all()

    def search(self, query: str):
        embedding = self.embedding_model.embed_query(query)

        return self.repository.search(
            embedding,
            k=config.top_k,
        )