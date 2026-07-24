from pathlib import Path

from placerag.chunker import chunk_document
from placerag.config import config
from placerag.embeddings import EmbeddingModel
from placerag.llm import LLM
from placerag.pdf_loader import load_pdf
from placerag.vector_store import VectorStore


class RAGPipeline:

    def __init__(self):

        self.embedding_model = EmbeddingModel()
        self.llm = LLM()

        self.vector_store = VectorStore()

    def index_document(self, pdf_path: Path):
        """Build the index if needed, otherwise load it."""

        index_dir = Path("data/vectorstore")

        index_file = index_dir / "index.faiss"
        metadata_file = index_dir / "chunks.json"

        if index_file.exists() and metadata_file.exists():
            print("Loading existing index...")
            self.vector_store.load(index_dir)
            return

        print("Building new index...")

        document = load_pdf(pdf_path)

        chunks = chunk_document(document)

        embeddings = self.embedding_model.embed_chunks(chunks)

        self.vector_store.build(embeddings, chunks)

        self.vector_store.save(index_dir)

    def load_index(self):

        self.vector_store.load(Path("data/vectorstore"))

    def ask(self, question: str):

        embedding = self.embedding_model.embed_query(question)

        chunks = self.vector_store.search(
            embedding,
            k=config.top_k,
        )

        context = "\n\n".join(
            chunk.text for chunk in chunks
        )

        return self.llm.generate(
            question,
            context,
        )