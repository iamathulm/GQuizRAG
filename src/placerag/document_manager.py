from pathlib import Path

from placerag.indexer import DocumentIndexer
from placerag.pipeline import RAGPipeline
from placerag.search_engine import SearchEngine


class DocumentManager:
    def __init__(self):
        self.index_root = Path("data/vectorstore")
        self.index_root.mkdir(parents=True, exist_ok=True)

    def open_repository(self, pdf_path: Path | None = None) -> RAGPipeline:
        """Index a PDF if provided, then open the document repository."""

        if pdf_path is not None:
            index_dir = self.index_root / pdf_path.stem
            index_file = index_dir / "index.faiss"

            if not index_file.exists():
                DocumentIndexer().build(pdf_path, index_dir)

        engine = SearchEngine(self.index_root)
        return RAGPipeline(engine)