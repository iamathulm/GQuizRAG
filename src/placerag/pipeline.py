from pathlib import Path

from placerag.config import config
from placerag.embeddings import EmbeddingModel
from placerag.llm import LLM
from placerag.vector_store import VectorStore


class RAGPipeline:
    def __init__(self, index_dir: Path):
        self.embedding_model = EmbeddingModel()
        self.llm = LLM()

        self.vector_store = VectorStore()
        self.vector_store.load(index_dir)

    def ask(self, question: str):
        embedding = self.embedding_model.embed_query(question)

        chunks = self.vector_store.search(
            embedding,
            k=config.top_k,
        )

        context = "\n\n".join(
            chunk.text
            for chunk in chunks
        )

        answer = self.llm.generate(question, context)

        return answer, chunks