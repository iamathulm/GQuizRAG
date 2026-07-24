from pathlib import Path

from placerag.pipeline import RAGPipeline

pipeline = RAGPipeline(
    Path("data/uploads/UNIT-2.pdf")
)

question = "Explain Round Robin scheduling."

answer = pipeline.ask(question)

print(answer)