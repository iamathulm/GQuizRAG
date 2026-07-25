from pathlib import Path

from placerag.bm25_store import BM25Store
from placerag.models import Chunk


def test_bm25_returns_best_match():
    chunks = [
        Chunk(
            text="Round Robin scheduling algorithm",
            source=Path("UNIT-2.pdf"),
        ),
        Chunk(
            text="Operating systems critical section",
            source=Path("UNIT-3.pdf"),
        ),
        Chunk(
            text="Virtual memory paging",
            source=Path("UNIT-4.pdf"),
        ),
    ]

    store = BM25Store()
    store.build(chunks)

    results = store.search("critical section")

    assert results[0][0].text == "Operating systems critical section"