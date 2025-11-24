import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma


def create_vector_store(pdf_path: str, persist_directory: str):
    """
    Creates a Chroma vector store from a PDF document.
    Automatically persists the embeddings in a directory.
    """

    # 1. Load PDF pages
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()

    # 2. Embedding function
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")

    # 3. Build Chroma DB
    vectordb = Chroma.from_documents(
        documents=pages,
        embedding=embeddings,
        persist_directory=persist_directory
    )

    # vectordb.persist()  # ensure data is saved to disk

    return vectordb
