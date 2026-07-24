from pathlib import Path

import numpy as np

from placerag.models import Chunk
from placerag.repository import DocumentRepository
from placerag.vector_store import VectorStore

def __len__(self):
    return len(self.vector_stores)

def test_load_all(tmp_path):
    # Create first vector store
    store1 = VectorStore()
    chunks1 = [Chunk(text="AI", source=Path("ai.pdf"))]
    embeddings1 = np.random.rand(1, 384).astype("float32")

    store1.build(embeddings1, chunks1)
    store1.save(tmp_path / "AI")

    # Create second vector store
    store2 = VectorStore()
    chunks2 = [Chunk(text="Networks", source=Path("net.pdf"))]
    embeddings2 = np.random.rand(1, 384).astype("float32")

    store2.build(embeddings2, chunks2)
    store2.save(tmp_path / "Networks")

    # Load both through the repository
    repo = DocumentRepository(tmp_path)
    repo.load_all()

    assert len(repo) == 2