from placerag.models import Chunk, Document


def chunk_document(
    document: Document,
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[Chunk]:
    """Split a document into overlapping character chunks."""

    text = document.content.strip()

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        chunk_text = text[start:end].strip()

        if chunk_text:
            chunks.append(
                Chunk(
                    text=chunk_text,
                    source=document.source,
                )
            )

        start += chunk_size - overlap

    return chunks