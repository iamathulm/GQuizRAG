import ollama
from placerag.config import config


class LLM:
    def __init__(self, model: str = "gemma3:4b"):
        self.model = model or config.llm_model

    def generate(self, question: str, context: str) -> str:
        prompt = f"""You are a helpful teaching assistant.

        Use ONLY the provided context to answer the question.

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
        messages = [
            {
                "role": "system",
                "content": (
                    "Use only the provided context to answer the user's question. "
                    "If the answer is not in the context, say you don't know."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            },
        ]

        stream = ollama.chat(
            model=config.llm_model,
            messages=messages,
            stream=True,
        )

        for chunk in stream:
            yield chunk["message"]["content"]