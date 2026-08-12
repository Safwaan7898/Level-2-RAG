from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from loaders.pdf_loader import PDFLoader
from services.embeddings import EmbeddingModel
from config import DB_DIRECTORY

PDF_PATH = "DATA\KIT_Mohammed Safwaan_CSE_Net.pdf"

documents = PDFLoader.load_pdf(PDF_PATH)

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(documents)

embeddings = EmbeddingModel.load_embeddings()

db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=DB_DIRECTORY
)

print(f"Ingested {len(chunks)} chunks into vector database.")
