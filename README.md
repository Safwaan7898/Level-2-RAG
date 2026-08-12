# Multi-PDF RAG Chatbot

A Streamlit-based Retrieval-Augmented Generation (RAG) application for uploading multiple PDF documents, building a vector search index, and asking questions with source citations.

## Features

- Upload multiple PDF files
- Extract and index PDF text using a vector database
- Answer questions using a RAG pipeline
- Display source citations and document snippets
- Configurable Top-K retrieval setting

## Requirements

- Python 3.11+ recommended
- Dependencies listed in `requirements.txt`

## Installation

1. Create and activate a Python virtual environment.

```bash
python -m venv venv
venv\Scripts\activate
```

2. Install dependencies.

```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

Then open the local URL displayed by Streamlit in your browser.

## Project Structure

- `app.py` - main Streamlit application
- `loaders/pdf_loader.py` - PDF loading utilities
- `services/vector_db.py` - vector database creation and queries
- `services/rag.py` - RAG chain and answer generation
- `requirements.txt` - Python dependencies

## Notes

- Uploaded PDFs are temporarily stored in `temp_uploads/`
- Existing vector DB files are present in `chroma_db/`
- Keep sensitive API keys out of source control
