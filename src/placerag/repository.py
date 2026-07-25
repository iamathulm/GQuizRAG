from pathlib import Path

from placerag.models import Chunk, SearchResult
from placerag.vector_store import VectorStore


class DocumentRepository:
    def __init__(self, vectorstore_dir: Path):
        self.vectorstore_dir = Path(vectorstore_dir)
        self.vector_stores: dict[str, VectorStore] = {}

    def __len__(self) -> int:
        return len(self.vector_stores)

    def list_documents(self) -> list[str]:
        """Return the names of all indexed documents."""
        return sorted(self.vector_stores.keys())

    def search( self, query_embedding: list[float],  k: int = 5, documents: list[str] | None = None, ) -> list[SearchResult]:
        """Search all vector stores and return the best matching chunks."""

        results = []

        stores = self.vector_stores.items()

        if documents is not None:
            stores = (
                (name, store)
                for name, store in stores
                if name in documents
            )

        for document_name, store in stores:
            store_results = store.search(query_embedding, k)
            results.extend(store_results)

        # Lower L2 distance = better match
        results.sort(key=lambda item: item[1])

        final_results = []

        for chunk, score in results[:k]:
            final_results.append(
                SearchResult(
                    chunk=chunk,
                    score=score,
                    document=chunk.source.stem,
                )
            )

        return final_results

    
    def load_all(self):
        self.vector_stores.clear()

        if not self.vectorstore_dir.exists():
            return

        for index_dir in self.vectorstore_dir.iterdir():
            if not index_dir.is_dir():
                continue

            store = VectorStore()
            store.load(index_dir)
            self.vector_stores[index_dir.name] = store

        print(f"Loaded {len(self.vector_stores)} vector stores")