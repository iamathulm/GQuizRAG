# GQuizRAG

GQuizRAG is a Retrieval-Augmented Generation (RAG) application for querying PDF documents using natural language. It combines semantic and keyword-based retrieval with Google's Gemini model to generate grounded answers with inline source citations.

The project was built to explore practical RAG architecture, hybrid information retrieval, LLM integration, document processing, and containerized deployment.

## Features

- Upload and query PDF documents through a Streamlit interface
- Extract and chunk document content for retrieval
- Generate semantic embeddings using Sentence Transformers
- Perform vector similarity search using FAISS
- Perform keyword-based retrieval using BM25
- Combine dense and sparse retrieval using Reciprocal Rank Fusion (RRF)
- Generate grounded responses using Gemini
- Provide inline source citations for generated answers
- Filter retrieval across indexed documents
- Persist document indexes for reuse
- Run automated tests with pytest
- Containerized deployment using Docker

## Architecture

GQuizRAG uses a hybrid RAG pipeline:

```text
                    PDF Documents
                          │
                          ▼
                    PDF Extraction
                          │
                          ▼
                       Chunking
                          │
             ┌────────────┴────────────┐
             ▼                         ▼
      Sentence Transformer         Tokenization
          Embeddings                   │
             │                         ▼
             ▼                        BM25
            FAISS                      │
             │                         │
             └────────────┬────────────┘
                          ▼
               Reciprocal Rank Fusion
                          │
                          ▼
                  Top Ranked Chunks
                          │
                          ▼
                  Context Construction
                          │
                          ▼
                     Gemini API
                          │
                          ▼
              Grounded Answer + Citations
```

### Retrieval Pipeline

GQuizRAG combines two complementary retrieval strategies.

**Dense retrieval** uses Sentence Transformers to convert document chunks and user queries into vector embeddings. FAISS performs similarity search over these vectors to retrieve semantically related passages.

**Sparse retrieval** uses BM25 to identify passages with strong lexical overlap with the query.

The results from both retrieval methods are combined using **Reciprocal Rank Fusion (RRF)**, allowing the system to benefit from both semantic similarity and keyword matching.

The highest-ranked chunks are then supplied to Gemini as context for answer generation.

## Tech Stack

| Component | Technology |
| --- | --- |
| Language | Python |
| UI | Streamlit |
| LLM | Google Gemini |
| Embeddings | Sentence Transformers |
| Vector Search | FAISS |
| Keyword Search | BM25 |
| Rank Fusion | Reciprocal Rank Fusion |
| PDF Processing | PyMuPDF |
| Package Management | uv |
| Testing | pytest |
| Containerization | Docker |

## Project Structure

```text
GQuizRAG/
├── src/
│   └── placerag/
│       ├── bm25_store.py
│       ├── chunker.py
│       ├── config.py
│       ├── document_manager.py
│       ├── embeddings.py
│       ├── indexer.py
│       ├── llm.py
│       ├── models.py
│       ├── pdf_loader.py
│       ├── pipeline.py
│       ├── repository.py
│       ├── search_engine.py
│       └── vector_store.py
│
├── tests/
├── app.py
├── Dockerfile
├── .dockerignore
├── .env.example
├── pyproject.toml
├── uv.lock
└── README.md
```

## Getting Started

### Prerequisites

You will need:

- Python 3.14+
- uv
- A Gemini API key

### 1. Clone the repository

```bash
git clone https://github.com/iamathulm/GQuizRAG
cd GQuizRAG
```

### 2. Install dependencies

```bash
uv sync
```

### 3. Configure Gemini

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_api_key_here
```

Do not commit this file. The repository includes `.env.example` as a configuration template.

### 4. Start the application

```bash
uv run python -m streamlit run app.py
```

Open the Streamlit URL displayed in the terminal, upload a PDF, and begin querying the document.

## Running with Docker

The application can also run inside a Docker container, providing a reproducible environment without requiring the Python dependencies to be installed directly on the host.

### Build the image

```bash
docker build -t gquizrag .
```

### Run the container

```bash
docker run --env-file .env -p 8501:8501 gquizrag
```

Then open:

```text
http://localhost:8501
```

The Gemini API key is provided to the container at runtime rather than being stored inside the Docker image.

## Testing

Run the automated test suite with:

```bash
uv run python -m pytest
```

The tests cover core components of the retrieval pipeline, including document chunking, vector search, BM25 retrieval, and repository behavior.

## How It Works

When a PDF is uploaded, GQuizRAG extracts its text and divides the content into overlapping chunks.

Each chunk is converted into a dense embedding and stored in a FAISS index. The same chunks are also indexed for BM25 keyword retrieval.

When a user asks a question:

1. The query is embedded using the same Sentence Transformer model.
2. FAISS retrieves semantically similar chunks.
3. BM25 retrieves lexically relevant chunks.
4. Reciprocal Rank Fusion combines both rankings.
5. The highest-ranked chunks are formatted as numbered sources.
6. Gemini receives the retrieved context and the user's question.
7. The model generates a grounded response with inline source citations.

If the retrieved context does not contain enough information to answer the question, the model is instructed not to generate an unsupported answer.

## Design Decisions

### Why Hybrid Retrieval?

Vector search is effective at identifying semantically similar content even when the wording differs between the query and document.

BM25 performs well when exact terminology, technical terms, or keywords matter.

Combining both approaches with Reciprocal Rank Fusion reduces dependence on a single retrieval strategy.

### Why Gemini?

Gemini provides fast hosted inference without requiring local GPU resources. This keeps the generation layer independent of local CUDA configuration while allowing document processing and retrieval to remain within the application.

The LLM is used for **generation**, while document retrieval is handled independently by FAISS, BM25, and the embedding model.

### Why Docker?

Docker packages the application and its runtime dependencies into a reproducible environment, reducing differences between development and deployment environments.

Secrets such as the Gemini API key are supplied at runtime and are not included in the Docker image.

## Future Improvements

Potential extensions include:

- Cross-encoder reranking
- OCR support for scanned PDFs
- Improved document metadata and page-level citations
- Conversation-aware retrieval
- Quiz generation and evaluation
- Configurable LLM providers
- Persistent deployment to a cloud platform

## License

This project is intended for educational and portfolio purposes.