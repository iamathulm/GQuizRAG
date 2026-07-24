from pathlib import Path

import streamlit as st

from placerag.document_manager import DocumentManager

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
    def load_pipeline(pdf_name: str):
        manager = DocumentManager()
        return manager.open_document(UPLOAD_DIR / pdf_name)

    pipeline = load_pipeline(uploaded_file.name)
    current_doc = uploaded_file.name

    if st.session_state.get("current_doc") != current_doc:
        st.session_state.current_doc = current_doc
        st.session_state.messages = []

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    question = st.chat_input("Ask about the document...")

    if question:
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, chunks = pipeline.ask(question)

            st.write(answer)
            with st.expander("Retrieved Sources"):

                for i, chunk in enumerate(chunks, start=1):

                    st.markdown(f"### Chunk {i}")

                    st.caption(chunk.source.name)

                    st.write(chunk.text)

                    st.divider()
                

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )