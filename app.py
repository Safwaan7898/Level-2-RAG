import os
import streamlit as st

from loaders.pdf_loader import PDFLoader
from services.vector_db import VectorDB
from services.rag import RAGService

# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------
st.set_page_config(
    page_title="Multi-PDF RAG Chatbot",
    page_icon="📄",
    layout="wide"
)

# ---------------------------------------------------
# Constants & Directory Setup
# ---------------------------------------------------
TEMP_DIR = "temp_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)

# ---------------------------------------------------
# Session State Initialization
# ---------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "processed_files_hash" not in st.session_state:
    st.session_state.processed_files_hash = None

if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0


# ---------------------------------------------------
# Helper Functions
# ---------------------------------------------------
def extract_citations(context_docs):
    """Extract unique source file names and page numbers from retrieved documents."""
    citations = []
    seen = set()
    for doc in context_docs:
        source_path = doc.metadata.get("source", "Unknown Document")
        file_name = os.path.basename(source_path)
        raw_page = doc.metadata.get("page", None)
        page_num = raw_page + 1 if (raw_page is not None and isinstance(raw_page, int)) else None

        key = (file_name, page_num)
        if key not in seen:
            seen.add(key)
            snippet = doc.page_content.strip()
            if len(snippet) > 200:
                snippet = snippet[:200] + "..."
            citations.append({
                "file_name": file_name,
                "page": page_num,
                "snippet": snippet
            })
    return citations


def process_uploaded_files(uploaded_files):
    """Save uploaded files to temp folder, load PDFs, and create Vector DB."""
    # Clean temp directory
    for f in os.listdir(TEMP_DIR):
        file_path = os.path.join(TEMP_DIR, f)
        if os.path.isfile(file_path):
            os.remove(file_path)

    saved_paths = []
    for uploaded_file in uploaded_files:
        temp_path = os.path.join(TEMP_DIR, uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_paths.append(temp_path)

    # Load documents from all uploaded PDFs
    documents = PDFLoader.load_multiple_pdfs(saved_paths)

    if not documents:
        return None, 0

    # Create vector database
    vector_db, num_chunks = VectorDB.create(documents)
    return vector_db, num_chunks


# ---------------------------------------------------
# Sidebar Configuration
# ---------------------------------------------------
with st.sidebar:
    st.title("📄 PDF RAG Assistant")
    st.markdown("---")

    # Task 1 & Task 2: Allow user to upload multiple PDFs
    st.subheader("📁 PDF Upload")
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or multiple PDF documents to ask questions about."
    )

    st.markdown("---")

    # Task 4: Retrieval Configuration (Top-K Slider)
    st.subheader("⚙️ Retrieval Settings")
    top_k = st.slider(
        "Top-K Chunks to Retrieve",
        min_value=1,
        max_value=10,
        value=3,
        step=1,
        help="Select the number of relevant document chunks (Top-K) to use for generating answers."
    )

    st.markdown("---")

    # Clear Chat Button
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    # Automatically process uploaded files when uploaded or changed
    if uploaded_files:
        files_hash = "_".join(sorted([f"{f.name}_{f.size}" for f in uploaded_files]))
        if files_hash != st.session_state.processed_files_hash:
            with st.spinner("Processing PDF documents & building vector index..."):
                vector_db, num_chunks = process_uploaded_files(uploaded_files)
                st.session_state.vector_db = vector_db
                st.session_state.chunk_count = num_chunks
                st.session_state.processed_files_hash = files_hash
                st.session_state.messages = []

    if st.session_state.vector_db is not None and uploaded_files:
        st.success(f"Database Ready! ({len(uploaded_files)} PDF(s), {st.session_state.chunk_count} Chunks)")
    else:
        st.info("Upload PDF(s) to activate vector search.")


# ---------------------------------------------------
# Main UI
# ---------------------------------------------------
st.title("📄 PDF Question Answering System")
st.caption("Upload multiple PDFs, ask questions, and view exact source citations.")

# ---------------------------------------------------
# Display Chat History with Source Citations (Task 3)
# ---------------------------------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("citations"):
            with st.expander("📚 Source Citations"):
                for cit in message["citations"]:
                    page_text = f" (Page {cit['page']})" if cit["page"] is not None else ""
                    st.markdown(f"• **{cit['file_name']}**{page_text}")
                    if cit.get("snippet"):
                        st.caption(f'"{cit["snippet"]}"')

# ---------------------------------------------------
# User Input Handling
# ---------------------------------------------------
if st.session_state.vector_db is None:
    st.info("👈 Please upload one or more PDF files in the sidebar to start asking questions.")
    st.stop()

question = st.chat_input("Ask a question about your uploaded PDFs...")

if question:
    # 1. User Message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # 2. Assistant Message with Citations & Dynamic Top-K
    with st.chat_message("assistant"):
        with st.spinner(f"Retrieving top {top_k} chunk(s) & generating answer..."):
            try:
                # Task 4: Dynamic chain with top_k
                chain = RAGService.create_chain(st.session_state.vector_db, top_k=top_k)
                response = chain.invoke({"input": question})

                answer = response.get("answer", "")
                context_docs = response.get("context", [])

                # Task 3: Source Citations
                citations = extract_citations(context_docs)

                st.markdown(answer)

                if citations:
                    with st.expander("📚 Source Citations"):
                        for cit in citations:
                            page_text = f" (Page {cit['page']})" if cit["page"] is not None else ""
                            st.markdown(f"• **{cit['file_name']}**{page_text}")
                            if cit.get("snippet"):
                                st.caption(f'"{cit["snippet"]}"')

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "citations": citations
                })

            except Exception as e:
                st.error(f"Error processing your query: {e}")