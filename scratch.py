from pathlib import Path

from placerag.chunker import chunk_document
from placerag.embeddings import EmbeddingModel
from placerag.pdf_loader import load_pdf

doc = load_pdf(Path("data/uploads/UNIT-2.pdf"))

chunks = chunk_document(doc)

model = EmbeddingModel()

embeddings = model.embed_chunks(chunks)

print(repr(doc.content[:1000]))
print(f"Chunks: {len(chunks)}")
print(f"Embedding dimension: {len(embeddings[0])}")