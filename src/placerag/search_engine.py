from pathlib import Path

from placerag.config import config
from placerag.embeddings import EmbeddingModel
from placerag.vector_store import VectorStore


class SearchEngine:
    def __init__(self, index_dir: Path):
        self.embedding_model = EmbeddingModel()

        self.vector_store = VectorStore()
        self.vector_store.load(index_dir)

    def search(self, query: str):
        embedding = self.embedding_model.embed_query(query)

        return self.vector_store.search(
            embedding,
            k=config.top_k,
        )