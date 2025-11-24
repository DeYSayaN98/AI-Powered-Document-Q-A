import streamlit as st
import os
import shutil
from vector_store import create_vector_store
from qa_chain import load_qa_chain

# --------------------------
# Streamlit Page Config
# --------------------------
st.set_page_config(page_title="AI Powered Document Q&A Bot")
st.title("🧠 AI Powered Document Q&A (Groq Mixtral)")

DB_PATH = "./chroma_langchain_db"


# --------------------------
# Utility Functions
# --------------------------
def cleanup_session():
    """Safely remove vectordb and QA chain from session and delete DB folder."""
    if "vectordb" in st.session_state:
        try:
            st.session_state.vectordb._client.reset()
        except:
            pass
        del st.session_state["vectordb"]

    if "qa_chain" in st.session_state:
        del st.session_state["qa_chain"]

    if os.path.exists(DB_PATH):
        try:
            shutil.rmtree(DB_PATH, ignore_errors=True)
        except:
            pass


# --------------------------
# Clear Session Button
# --------------------------
if st.button("🧹 Clear Previous Session"):
    cleanup_session()
    st.success("Session cleared!")


# --------------------------
# PDF Upload
# --------------------------
uploaded_file = st.file_uploader("📄 Upload PDF", type=["pdf"])

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())
    st.success("✅ PDF uploaded!")

    if st.button("📌 Process PDF"):
        with st.spinner("Indexing document..."):
            cleanup_session()
            os.makedirs(DB_PATH, exist_ok=True)

            vectordb = create_vector_store("temp.pdf", DB_PATH)
            st.session_state.vectordb = vectordb

            st.session_state.qa_chain = load_qa_chain(DB_PATH)

            st.success("✅ Done! You can now ask questions.")


# --------------------------
# Q&A Interface
# --------------------------
if "qa_chain" in st.session_state:
    query = st.text_input("💬 Ask a question:")
    if query:
        with st.spinner("Thinking..."):
            result = st.session_state.qa_chain.invoke(query)
            st.write("### ✅ Answer:")
            st.write(result["result"])
