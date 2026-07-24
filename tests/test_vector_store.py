import numpy as np
from pathlib import Path

from placerag.models import Chunk
from placerag.vector_store import VectorStore


def test_build_vector_store():
    chunks = [
        Chunk(text="First chunk", source=Path("doc.pdf")),
        Chunk(text="Second chunk", source=Path("doc.pdf")),
    ]

    embeddings = np.random.rand(2, 384).astype("float32")

    store = VectorStore()
    store.build(embeddings, chunks)

    assert store.index.ntotal == 2

def test_search_returns_chunks():
    chunks = [
        Chunk(text="A", source=Path("doc.pdf")),
        Chunk(text="B", source=Path("doc.pdf")),
    ]

    embeddings = np.random.rand(2, 384).astype("float32")

    store = VectorStore()
    store.build(embeddings, chunks)

    query = embeddings[0]

    results = store.search(query, k=1)

    assert len(results) == 1

def test_save_and_load(tmp_path):
    chunks = [
        Chunk(text="Hello", source=Path("doc.pdf")),
    ]

    embeddings = np.random.rand(1, 384).astype("float32")

    store = VectorStore()
    store.build(embeddings, chunks)

    store.save(tmp_path)

    loaded = VectorStore()
    loaded.load(tmp_path)

    assert loaded.index.ntotal == 1
    assert loaded.chunks[0].text == "Hello"