from pathlib import Path

from placerag.chunker import chunk_document
from placerag.embeddings import EmbeddingModel
from placerag.pdf_loader import load_pdf
from placerag.vector_store import VectorStore


class DocumentIndexer:
    def __init__(self):
        self.embedding_model = EmbeddingModel()

    def build(self, pdf_path: Path, index_dir: Path):
        document = load_pdf(pdf_path)

        chunks = chunk_document(document)

        embeddings = self.embedding_model.embed_chunks(chunks)

        store = VectorStore()

        store.build(embeddings, chunks)

        store.save(index_dir)