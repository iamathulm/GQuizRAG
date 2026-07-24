from pathlib import Path

from placerag.chunker import chunk_document
from placerag.embeddings import EmbeddingModel
from placerag.pdf_loader import load_pdf
from placerag.vector_store import VectorStore

doc = load_pdf(Path("data/uploads/UNIT-2.pdf"))

chunks = chunk_document(doc)

embedding_model = EmbeddingModel()

embeddings = embedding_model.embed_chunks(chunks)

store = VectorStore()
store.build(embeddings, chunks)

query = "What is Round Robin scheduling?"

query_embedding = embedding_model.embed_query(query)

results = store.search(query_embedding, k=3)

print(f"Query: {query}\n")

for i, chunk in enumerate(results, start=1):
    print("=" * 60)
    print(f"Result {i}")
    print("-" * 60)
    print(chunk.text)
    print()