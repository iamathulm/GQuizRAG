from google import genai
from google.genai import types
from placerag.config import config


class LLM:
    def __init__(self, model: str | None = None):
        if not config.gemini_api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. "
                "Add it to the .env file."
            )

        self.model = model or config.llm_model

        self.client = genai.Client(
    api_key=config.gemini_api_key,
    http_options=types.HttpOptions(
        timeout=30_000,
    ),
)

    def _build_prompt(self, question: str, context: str) -> str:
        return f"""You are a helpful teaching assistant.

Your task is to answer the user's question using ONLY the provided sources.

Rules:
- Use only information contained in the provided context.
- Do not add facts from your own knowledge.
- Cite every factual claim using the corresponding source number.
- Use citations in the format [Source 1].
- When multiple sources support the same claim, cite them separately, for example [Source 1] [Source 3].
- Never invent a source number that is not present in the provided context.
- If the context does not contain enough information to answer the question, reply exactly:
  "I couldn't find the answer in the provided document."
- Explain the answer clearly and naturally in your own words.
- Prefer a concise answer, but include enough detail to fully answer the question.
- Use bullets or short sections when they improve readability.
- Do not simply copy passages from the context.

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
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
            )

            return response.text or ""

        except Exception as e:
            return f"Gemini API error: {e}"

    def generate_stream(self, question: str, context: str):
        prompt = self._build_prompt(question, context)

        try:
            stream = self.client.models.generate_content_stream(
                model=self.model,
                contents=prompt,
            )

            for chunk in stream:
                if chunk.text:
                    yield chunk.text

        except Exception as e:
            yield f"Gemini API error: {e}"