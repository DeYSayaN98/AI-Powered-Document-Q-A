import os
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings


def load_qa_chain(persist_directory: str):
    """
    Loads a Chroma vector store and builds a RetrievalQA chain using Groq LLM.
    """

    # 1. Load Groq LLM
    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="mixtral-8x7b-32768"  # You can also use llama3-8b
    )

    # 2. Embeddings model
    embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 3. Load vector database
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_function
    )

    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    # 4. Create QA Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    return qa_chain
