from pathlib import Path

from placerag.models import Chunk
from placerag.vector_store import VectorStore


class DocumentRepository:
    def __init__(self, vectorstore_dir: Path):
        self.vectorstore_dir = Path(vectorstore_dir)
        self.vector_stores: list[VectorStore] = []

    def __len__(self) -> int:
        return len(self.vector_stores)

    def search(
        self,
        query_embedding: list[float],
        k: int = 5,
    ) -> list[Chunk]:
        """Search all vector stores and return the best matching chunks."""

        results: list[tuple[Chunk, float]] = []

        for i, store in enumerate(self.vector_stores, start=1):
            store_results = store.search(query_embedding, k)
            print(f"Store {i}: {len(store_results)} matches")
            results.extend(store_results)

        # Lower L2 distance = better match
        results.sort(key=lambda item: item[1])

        return [chunk for chunk, _ in results[:k]]

    
    def load_all(self):
        self.vector_stores.clear()

        if not self.vectorstore_dir.exists():
            return

        for index_dir in self.vectorstore_dir.iterdir():
            if not index_dir.is_dir():
                continue

            store = VectorStore()
            store.load(index_dir)
            self.vector_stores.append(store)

        print(f"Loaded {len(self.vector_stores)} vector stores")