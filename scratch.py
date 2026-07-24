from pathlib import Path

from placerag.chunker import chunk_document
from placerag.embeddings import EmbeddingModel
from placerag.pdf_loader import load_pdf
from placerag.vector_store import VectorStore

doc = load_pdf(Path("data/uploads/UNIT-2.pdf"))

chunks = chunk_document(doc)

model = EmbeddingModel()

embeddings = model.embed_chunks(chunks)

store = VectorStore()
store.build(embeddings, chunks)

store.save(Path("data/vectorstore"))

print("Saved!")

loaded = VectorStore()
loaded.load(Path("data/vectorstore"))

print(f"Loaded {len(loaded.chunks)} chunks.")