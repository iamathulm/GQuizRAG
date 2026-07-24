from pathlib import Path

from placerag.chunker import chunk_document
from placerag.embeddings import EmbeddingModel
from placerag.llm import LLM
from placerag.pdf_loader import load_pdf
from placerag.vector_store import VectorStore
from placerag.config import config



class RAGPipeline:
    def __init__(self, pdf_path: Path):
        self.embedding_model = EmbeddingModel()
        self.llm = LLM()
        self.vector_store = VectorStore()

        document = load_pdf(pdf_path)

        chunks = chunk_document(document)

        embeddings = self.embedding_model.embed_chunks(chunks)

        self.vector_store.build(embeddings, chunks)

    def ask(self, question: str, k: int = config.top_k) -> str:
        query_embedding = self.embedding_model.embed_query(question)

        chunks = self.vector_store.search(query_embedding, k=k)

        context = "\n\n".join(
            f"Source: {chunk.source.name}\n\n{chunk.text}"
            for chunk in chunks
        )

        return self.llm.generate(question, context)