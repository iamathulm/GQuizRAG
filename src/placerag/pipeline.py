from placerag.llm import LLM
from placerag.search_engine import SearchEngine
from placerag.models import SearchResult

class RAGPipeline:
    def __init__(self, search_engine: SearchEngine):
        self.search_engine = search_engine
        self.llm = LLM()

    def _retrieve(
        self,
        question: str,
        documents: list[str] | None = None,
    )-> tuple[str, list[SearchResult]]:
        results = self.search_engine.search(
            question,
            documents=documents,
        )

        context = "\n\n".join(
            result.chunk.text
            for result in results
        )

        return context, results

    def ask(
        self,
        question: str,
        documents: list[str] | None = None,
    ) -> tuple[str, list[SearchResult]]:
        context, chunks = self._retrieve(
            question,
            documents,
        )

        answer = self.llm.generate(
            question,
            context,
        )

        return answer, chunks

    def ask_stream(
        self,
        question: str,
        documents: list[str] | None = None,
    ):
        context, chunks = self._retrieve(
            question,
            documents,
        )

        stream = self.llm.generate_stream(
            question,
            context,
        )

        return stream, chunks

    def list_documents(self) -> list[str]:
        return self.search_engine.repository.list_documents()