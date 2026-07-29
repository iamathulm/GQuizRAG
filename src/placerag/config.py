import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Config:
    llm_model: str = "gemini-3.6-flash"
    gemini_api_key: str | None = os.getenv("GEMINI_API_KEY")

    embedding_model: str = "all-MiniLM-L6-v2"

    chunk_size: int = 500
    chunk_overlap: int = 100

    top_k: int = 6


config = Config()