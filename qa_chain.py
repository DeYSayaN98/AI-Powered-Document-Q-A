from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_ollama.embeddings import OllamaEmbeddings


def load_qa_chain(persist_directory: str):
    """
    Loads a Chroma vector store and builds a RetrievalQA chain using Ollama.
    """

    # Load LLaMA model
    llm = OllamaLLM(model="llama3.2")

    # Embeddings model
    embedding_function = OllamaEmbeddings(model="mxbai-embed-large")

    # Load vector database
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_function
    )

    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    # Create QA Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    return qa_chain
