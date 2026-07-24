from pathlib import Path

from placerag.indexer import DocumentIndexer
from placerag.pipeline import RAGPipeline


class DocumentManager:
    def __init__(self):
        self.index_root = Path("data/vectorstore")
        self.index_root.mkdir(parents=True, exist_ok=True)

    def open_document(self, pdf_path: Path) -> RAGPipeline:
        index_dir = self.index_root / pdf_path.stem
        index_file = index_dir / "index.faiss"

        if not index_file.exists():
            DocumentIndexer().build(pdf_path, index_dir)

        return RAGPipeline(index_dir)