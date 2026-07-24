from sentence_transformers import SentenceTransformer

from placerag.models import Chunk


class EmbeddingModel:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed_chunks(self, chunks: list[Chunk]) -> list[list[float]]:
        texts = [chunk.text for chunk in chunks]
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        embedding = self.model.encode(query, convert_to_numpy=True)
        return embedding.tolist()