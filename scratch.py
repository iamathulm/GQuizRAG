from pathlib import Path

from placerag.pipeline import RAGPipeline

pipeline = RAGPipeline()

pipeline.index_document(
    Path("data/uploads/UNIT-2.pdf")
)

while True:
    question = input("\nYou: ")

    if question.lower() in ("quit", "exit"):
        break

    print()
    print(pipeline.ask(question))