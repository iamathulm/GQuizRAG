from pathlib import Path

import streamlit as st

from placerag.pipeline import RAGPipeline

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="placeRAG",
    page_icon="📄",
)

st.title("📄 placeRAG")

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type="pdf",
)

if uploaded_file:

    pdf_path = UPLOAD_DIR / uploaded_file.name

    if not pdf_path.exists():
        pdf_path.write_bytes(uploaded_file.getbuffer())

    @st.cache_resource
    def load_pipeline(pdf_file: str):
        pipeline = RAGPipeline()
        pipeline.index_document(Path(pdf_file))
        return pipeline

    pipeline = load_pipeline(str(pdf_path))

    question = st.text_input(
        "Ask a question about the document:"
    )

    if st.button("Ask") and question:

        with st.spinner("Thinking..."):

            answer = pipeline.ask(question)

        st.markdown("### Answer")

        st.write(answer)