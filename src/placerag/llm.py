import ollama


class LLM:
    def __init__(self, model: str = "gemma3:4b"):
        self.model = model

    def generate(self, question: str, context: str) -> str:
        prompt = f"""You are a helpful assistant.

Answer the user's question using ONLY the context below.

If the answer cannot be found in the context, say:
"I couldn't find the answer in the provided document."

Context:
{context}

Question:
{question}

Answer:
"""

        response = ollama.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        return response["message"]["content"]