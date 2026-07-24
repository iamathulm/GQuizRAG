from placerag.models import Chunk, Document


def chunk_document(document: Document) -> list[Chunk]:
    """Split a document into paragraph-sized chunks."""

    paragraphs = document.content.split("\n\n")

    chunks = []

    for paragraph in paragraphs:
        paragraph = paragraph.strip()

        if not paragraph:
            continue

        chunks.append(
            Chunk(
                text=paragraph,
                source=document.source,
            )
        )

    return chunks