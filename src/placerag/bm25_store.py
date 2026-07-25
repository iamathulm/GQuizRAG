from rank_bm25 import BM25Okapi

from placerag.models import Chunk


class BM25Store:
    def __init__(self):
        self.index: BM25Okapi | None = None
        self.chunks: list[Chunk] = []

    def _tokenize(self, text: str) -> list[str]:
        return text.lower().split()

    def build(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks

        corpus = [
            self._tokenize(chunk.text)
            for chunk in chunks
        ]

        self.index = BM25Okapi(corpus)

    def search(
        self,
        query: str,
        k: int = 3,
    ) -> list[tuple[Chunk, float]]:

        if self.index is None:
            raise RuntimeError("BM25 index has not been built.")

        scores = self.index.get_scores(
            self._tokenize(query)
        )

        ranked = sorted(
            zip(self.chunks, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        return ranked[:k]