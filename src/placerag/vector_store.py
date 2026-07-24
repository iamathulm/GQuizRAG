import faiss
import numpy as np
import json
from pathlib import Path
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

    def search(    self,    embedding: list[float],    k: int = 3,) -> list[tuple[Chunk, float]]:
        """Return the k most similar chunks and their scores."""

        if self.index is None:
            raise RuntimeError("Vector store has not been built.")

        query = np.array([embedding], dtype=np.float32)

        scores, indices = self.index.search(query, k)

        results = []

        for score, index in zip(scores[0], indices[0]):
            if index == -1:
                continue

            results.append((self.chunks[index], float(score)))

        return results

    def load(self, directory: Path) -> None:
        """Load the FAISS index and chunk metadata."""

        self.index = faiss.read_index(str(directory / "index.faiss"))

        with open(directory / "chunks.json", encoding="utf-8") as f:
            metadata = json.load(f)

        self.chunks = [
            Chunk(
                text=item["text"],
                source=Path(item["source"]),
                page=item["page"],
                slide=item["slide"],
            )
            for item in metadata
        ]

    def save(self, directory: Path) -> None:
        """Save the FAISS index and chunk metadata."""

        if self.index is None:
            raise RuntimeError("Vector store has not been built.")

        directory.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(directory / "index.faiss"))

        metadata = []

        for chunk in self.chunks:
            metadata.append(
                {
                    "text": chunk.text,
                    "source": str(chunk.source),
                    "page": chunk.page,
                    "slide": chunk.slide,
                }
            )

        with open(directory / "chunks.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)