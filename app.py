from pathlib import Path

import streamlit as st

from placerag.pipeline import RAGPipeline

st.set_page_config(
    page_title="placeRAG",
    page_icon="📄",
)

st.title("📄 placeRAG")

pdf_path = Path("data/uploads/sample.pdf")

@st.cache_resource
def load_pipeline():
    pipeline = RAGPipeline()
    pipeline.index_document(pdf_path)
    return pipeline

pipeline = load_pipeline()

question = st.text_input(
    "Ask a question about the document:"
)

if st.button("Ask") and question:

    with st.spinner("Thinking..."):

        answer = pipeline.ask(question)

    st.markdown("### Answer")

    st.write(answer)