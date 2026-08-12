from loaders.pdf_loader import PDFLoader
from services.vector_db import VectorDB

docs = PDFLoader.load_pdf(r"C:\Users\mmoha\RAG Projects\RAG_BASE-main\RAG_BASE-main\DATA\KIT_Mohammed Safwaan_CSE_Net.pdf")

db = VectorDB.create(docs)

print("Database Created Successfully")