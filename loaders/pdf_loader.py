from langchain_community.document_loaders import PyPDFLoader


class PDFLoader:

    @staticmethod
    def load_pdf(pdf_path: str):
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        return documents

    @staticmethod
    def load_multiple_pdfs(pdf_paths: list[str]):
        all_documents = []
        for pdf_path in pdf_paths:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            all_documents.extend(docs)
        return all_documents