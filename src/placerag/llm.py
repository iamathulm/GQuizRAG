import ollama
from placerag.config import config


class LLM:
    def __init__(self, model: str = "gemma3:4b"):
        self.model = model or config.llm_model

    def _build_prompt(self, question: str, context: str) -> str:
        return f"""You are a helpful teaching assistant.

            Use ONLY the provided context to answer the question.

            Whenever you use information from a source, cite it inline using its
            source number, for example [Source 1] or [Source 2].

            If the context does not contain the answer, reply exactly:
            "I couldn't find the answer in the provided document."

            Write a clear, complete answer in your own words.
            Do not simply copy the context unless necessary.

            Context:
            {context}

            Question:
            {question}

            Answer:
            """

    def generate(self, question: str, context: str) -> str:
        prompt = self._build_prompt(question, context)
        print("=" * 60)
        print(f"Question: {question}")
        print(f"Context length: {len(context)}")
        print(f"Prompt length: {len(prompt)}")
        print("=" * 60)
        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
            )
        except ollama.ResponseError as e:
            return f"Ollama error: {e}"

        return response["message"]["content"]

    def generate_stream(self, question: str, context: str):
        prompt = self._build_prompt(question, context)

        stream = ollama.chat(
            model=config.llm_model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            stream=True,
        )

        for chunk in stream:
            yield chunk["message"]["content"]
    