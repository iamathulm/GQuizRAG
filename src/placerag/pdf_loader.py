from pathlib import Path

import fitz

from placerag.models import Document


def load_pdf(pdf_path: Path) -> Document:
    """Load a PDF and extract all text."""

    with fitz.open(pdf_path) as pdf:
        text = ""

        for page in pdf:
            text += page.get_text()

    return Document(
        source=pdf_path,
        content=text,
    )