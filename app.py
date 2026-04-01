import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from dotenv import load_dotenv
import os

load_dotenv()

st.header("My Chatbot")

# ✅ Chat memory
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.title("Your Documents")
    files = st.file_uploader("Upload your files", type="pdf", accept_multiple_files=True)

# ✅ Show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

documents = []

if files:
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"],
        chunk_size=1000,
        chunk_overlap=150
    )

    # ✅ Create documents with metadata (file + page)
    for file in files:
        pdf_reader = PdfReader(file)
        for i, page in enumerate(pdf_reader.pages):
            content = page.extract_text() or ""
            chunks = text_splitter.split_text(content)

            for chunk in chunks:
                documents.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "source": file.name,
                            "page": i + 1
                        }
                    )
                )

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_documents(documents, embedding=embeddings)

    user_question = st.chat_input("Ask something...")

    llm = ChatGroq(
       model="llama-3.1-8b-instant"
       
    )

    if user_question:
        # ✅ store user message
        st.session_state.messages.append({
            "role": "user",
            "content": user_question
        })

        with st.chat_message("user"):
            st.write(user_question)

        # ✅ retrieve docs
        docs = vector_store.similarity_search(user_question, k=3)

        context = "\n".join([doc.page_content for doc in docs])

        prompt = f"""
        Answer the question based on the context below:

        Context:
        {context}

        Question:
        {user_question}
        """

        with st.spinner("Thinking..."):
            response = llm.invoke(prompt)

        # ✅ show answer
        with st.chat_message("assistant"):
            st.write(response.content)

            # 🔥 SHOW CITATIONS
            st.markdown("**Sources:**")
            for doc in docs:
                st.write(f"📄 {doc.metadata['source']} (Page {doc.metadata['page']})")

        # ✅ save bot message
        st.session_state.messages.append({
            "role": "assistant",
            "content": response.content
        })