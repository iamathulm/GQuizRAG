from pathlib import Path

from placerag.pdf_loader import load_pdf

doc = load_pdf(Path("data/uploads/UNIT-2.pdf"))

print(doc.source)
print()
print(doc.content[:1000])