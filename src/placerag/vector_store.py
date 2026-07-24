import faiss
import numpy as np

from placerag.models import Chunk


class VectorStore:
    def __init__(self):
        self.index = None
        self.chunks: list[Chunk] = []

    def build(self, embeddings: list[list[float]], chunks: list[Chunk]) -> None:
        """Build a FAISS index from embeddings."""

        vectors = np.array(embeddings, dtype=np.float32)

        dimension = vectors.shape[1]

        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(vectors)

        self.chunks = chunks

    def search(self, embedding: list[float], k: int = 3) -> list[Chunk]:
        """Return the k most similar chunks."""

        if self.index is None:
            raise RuntimeError("Vector store has not been built.")

        query = np.array([embedding], dtype=np.float32)

        _, indices = self.index.search(query, k)

        return [self.chunks[i] for i in indices[0]]