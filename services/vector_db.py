import os
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from services.embeddings import EmbeddingModel
from config import DB_DIRECTORY


class VectorDB:

    @staticmethod
    def load(persist_directory: str = DB_DIRECTORY, collection_name: str = "pdf_rag"):
        embeddings = EmbeddingModel.load_embeddings()
        db = Chroma(
            collection_name=collection_name,
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
        return db

    @staticmethod
    def create(documents, persist_directory: str = DB_DIRECTORY, collection_name: str = "pdf_rag"):
        embeddings = EmbeddingModel.load_embeddings()

        # Safely reset existing collection if it exists
        try:
            existing_db = Chroma(
                collection_name=collection_name,
                persist_directory=persist_directory,
                embedding_function=embeddings
            )
            existing_db.delete_collection()
        except Exception as e:
            print(f"Notice: Resetting collection: {e}")

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(documents)

        db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=collection_name,
            persist_directory=persist_directory
        )

        print(f"Created Vector Database with {len(chunks)} chunks.")
        return db, len(chunks)