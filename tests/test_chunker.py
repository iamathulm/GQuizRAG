from pathlib import Path

from placerag.chunker import chunk_document
from placerag.models import Document
from placerag.config import config



def test_chunk_document_returns_chunks():
    text = "A" * 1200

    document = Document(
        source=Path("sample.pdf"),
        content=text,
    )

    chunks = chunk_document(document)

    assert len(chunks) > 1


def test_chunk_overlap():
    text = "0123456789" * 100

    document = Document(
        source=Path("sample.pdf"),
        content=text,
    )

    chunks = chunk_document(document)

    assert (
        chunks[0].text[-config.chunk_overlap:]
        == chunks[1].text[:config.chunk_overlap]
    )

    

def test_chunk_source_is_preserved():
    document = Document(
        source=Path("sample.pdf"),
        content="Hello World" * 100,
    )

    chunks = chunk_document(document)

    assert all(chunk.source == document.source for chunk in chunks)