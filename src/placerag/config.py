from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    llm_model: str = "gemma3:4b"
    embedding_model: str = "all-MiniLM-L6-v2"

    chunk_size: int = 500
    chunk_overlap: int = 100

    top_k: int = 6


config = Config()