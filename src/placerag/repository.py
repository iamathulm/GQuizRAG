from pathlib import Path
from collections import defaultdict
from placerag.models import Chunk, SearchResult
from placerag.vector_store import VectorStore
from placerag.bm25_store import BM25Store


class DocumentRepository:
    def __init__(self, vectorstore_dir: Path):
        self.vectorstore_dir = Path(vectorstore_dir)

        self.vector_stores: dict[str, VectorStore] = {}
        self.bm25_stores: dict[str, BM25Store] = {}

    def __len__(self) -> int:
        return len(self.vector_stores)

    def list_documents(self) -> list[str]:
        """Return the names of all indexed documents."""
        return sorted(self.vector_stores.keys())

    def search(self,query_embedding: list[float],query: str,k: int = 5,documents: list[str] | None = None,) -> list[SearchResult]:
        """Search using hybrid retrieval (FAISS + BM25)."""

        RRF_K = 60

        fused_scores = defaultdict(float)
        chunk_lookup = {}

        stores = self.vector_stores.items()

        if documents is not None:
            stores = (
                (name, store)
                for name, store in stores
                if name in documents
            )

        for document_name, vector_store in stores:

            bm25_store = self.bm25_stores[document_name]

            semantic_results = vector_store.search(query_embedding, k)
            lexical_results = bm25_store.search(query, k)

            # FAISS contribution
            for rank, (chunk, _) in enumerate(semantic_results, start=1):
                key = (chunk.source, chunk.text)
                fused_scores[key] += 1 / (RRF_K + rank)
                chunk_lookup[key] = chunk

            # BM25 contribution
            for rank, (chunk, _) in enumerate(lexical_results, start=1):
                key = (chunk.source, chunk.text)
                fused_scores[key] += 1 / (RRF_K + rank)
                chunk_lookup[key] = chunk

        ranked = sorted(
            fused_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        return [
            SearchResult(
                chunk=chunk_lookup[key],
                score=score,
                document=chunk_lookup[key].source.stem,
            )
            for key, score in ranked[:k]
        ]

    
    def load_all(self):
        self.vector_stores.clear()
        self.bm25_stores.clear()

        if not self.vectorstore_dir.exists():
            return

        for index_dir in self.vectorstore_dir.iterdir():
            if not index_dir.is_dir():
                continue

            store = VectorStore()
            store.load(index_dir)
            self.vector_stores[index_dir.name] = store

            bm25 = BM25Store()
            bm25.build(store.chunks)
            self.bm25_stores[index_dir.name] = bm25

        print(f"Loaded {len(self.vector_stores)} vector stores")