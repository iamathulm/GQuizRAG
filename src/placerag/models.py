from dataclasses import dataclass
from pathlib import Path


@dataclass
class Document:
    source: Path
    content: str


@dataclass
class Chunk:
    text: str
    source: Path
    page: int | None = None
    slide: int | None = None


@dataclass
class SearchResult:
    chunk: Chunk
    score: float
    document: str