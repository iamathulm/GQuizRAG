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