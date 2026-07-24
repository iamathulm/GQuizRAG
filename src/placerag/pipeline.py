from placerag.llm import LLM
from placerag.search_engine import SearchEngine


class RAGPipeline:
    def __init__(self, search_engine: SearchEngine):
        self.search_engine = search_engine
        self.llm = LLM()

    def _retrieve(self, question: str):
        chunks = self.search_engine.search(question)

        context = "\n\n".join(
            chunk.text
            for chunk in chunks
        )

        return context, chunks

    def ask(self, question: str):
        context, chunks = self._retrieve(question)

        answer = self.llm.generate(
            question,
            context,
        )

        return answer, chunks

    def ask_stream(self, question: str):
        context, chunks = self._retrieve(question)

        stream = self.llm.generate_stream(
            question,
            context,
        )

        return stream, chunks