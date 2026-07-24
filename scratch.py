from pathlib import Path

from placerag.chunker import chunk_document
from placerag.pdf_loader import load_pdf

doc = load_pdf(Path("data/uploads/UNIT-2.pdf"))

chunks = chunk_document(doc)

print(f"Chunks: {len(chunks)}")

print()
print("=" * 60)
print(chunks[0].text)