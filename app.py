import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from dotenv import load_dotenv
import os
import zipfile

load_dotenv()

with st.sidebar:
    st.title("Your Documents")
    files = st.file_uploader("Upload your files", accept_multiple_files=True, key="sidebar_uploader")

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("<h1 style='text-align: center; color: white;'>Hello Coders!!</h1>", unsafe_allow_html=True)

#  Centered collapsible file upload
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    with st.expander(" Upload Your code files", expanded=True):
        files_main = st.file_uploader("Upload your files", accept_multiple_files=True, key="main_uploader")

#  Combine files from both sidebar and main uploader
files = list(set(list(files) + list(files_main))) if files or files_main else []

#  Chat memory
if "messages" not in st.session_state:
    st.session_state.messages = []

#  Show chat history
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

    for file in files:

        #  ZIP HANDLING (FIXED INDENTATION)
        if file.name.endswith('.zip'):
            with zipfile.ZipFile(file, 'r') as zip_ref:

                #  Add file list
                file_list = "\n".join(zip_ref.namelist())

                documents.append(
                    Document(
                        page_content=f"Files inside zip:\n{file_list}",
                        metadata={
                            "source_file": file.name,
                            "internal_file": "ZIP_MANIFEST",
                            "page": "Index",
                            "chunk": 0,
                            "type": "zip_manifest",
                            "file_path": file.name
                        }
                    )
                )

                #  Read code files inside zip
                for info in zip_ref.infolist():
                    if info.filename.endswith(('.py', '.js', '.ts', '.java', '.cpp')):
                        try:
                            content = zip_ref.read(info).decode('utf-8', errors='ignore')
                        except:
                            continue

                        chunks = text_splitter.split_text(content)

                        for chunk_idx, chunk in enumerate(chunks, 1):
                            documents.append(
                                Document(
                                    page_content=chunk,
                                    metadata={
                                        "source_file": file.name,
                                        "internal_file": info.filename,
                                        "page": f"Chunk {chunk_idx}",
                                        "chunk": chunk_idx,
                                        "type": "code",
                                        "file_path": f"{file.name} → {info.filename}",
                                        "total_chunks": len(chunks)
                                    }
                                )
                            )

        #  PDF HANDLING
        elif file.name.endswith('.pdf'):
            pdf_reader = PdfReader(file)
            for i, page in enumerate(pdf_reader.pages):
                content = page.extract_text() or ""
                chunks = text_splitter.split_text(content)

                for chunk_idx, chunk in enumerate(chunks, 1):
                    documents.append(
                        Document(
                            page_content=chunk,
                            metadata={
                                "source_file": file.name,
                                "internal_file": None,
                                "page": i + 1,
                                "chunk": chunk_idx,
                                "type": "pdf",
                                "file_path": file.name,
                                "total_chunks": len(chunks)
                            }
                        )
                    )

        #  NORMAL FILES
        else:
            try:
                content = file.read().decode('utf-8', errors='ignore')
            except:
                continue

            chunks = text_splitter.split_text(content)

            for chunk_idx, chunk in enumerate(chunks, 1):
                documents.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "source_file": file.name,
                            "internal_file": None,
                            "page": f"Chunk {chunk_idx}",
                            "chunk": chunk_idx,
                            "type": "text",
                            "file_path": file.name,
                            "total_chunks": len(chunks)
                        }
                    )
                )

    #  Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    #  Vector store
    vector_store = FAISS.from_documents(documents, embedding=embeddings)

    user_question = st.chat_input("Ask something...")

    #  Updated model
    llm = ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=0
    )

    if user_question:
        st.session_state.messages.append({
            "role": "user",
            "content": user_question
        })

        with st.chat_message("user"):
            st.write(user_question)

        #  Better retrieval
        docs = vector_store.similarity_search(user_question, k=5)

        context = "\n\n".join([doc.page_content for doc in docs])

        #  STRONG PROMPT (ANSWER FIRST → THEN ANALYZE)
        prompt = f"""
You are a senior software engineer and expert coding assistant.

Step 1: Directly answer the user's question clearly.
Step 2: Then analyze the provided context for deeper insights.

Instructions:
- Answer FIRST in 2-4 lines
- Then give detailed explanation
- If code is present, explain it
- If bug exists, identify and fix it
- Mention file names where relevant
- If question is about files, list them clearly
- If not found, say: "Not found in document"

Context:
{context}

Question:
{user_question}

Answer:
"""

        with st.spinner("Thinking..."):
            response = llm.invoke(prompt)

        with st.chat_message("assistant"):
            st.write(response.content)

            #  IMPROVED: Better source display with file tracking
            st.markdown("---")
            st.markdown("### 📚 **Sources Used:**")
            
            #  Group sources by file for better readability
            sources_by_file = {}
            for doc in docs:
                source_file = doc.metadata.get("source_file", "Unknown")
                internal_file = doc.metadata.get("internal_file")
                page = doc.metadata.get("page", "N/A")
                chunk = doc.metadata.get("chunk", 0)
                doc_type = doc.metadata.get("type", "document")
                file_path = doc.metadata.get("file_path", source_file)
                
                if source_file not in sources_by_file:
                    sources_by_file[source_file] = []
                
                sources_by_file[source_file].append({
                    "internal_file": internal_file,
                    "page": page,
                    "chunk": chunk,
                    "type": doc_type,
                    "file_path": file_path
                })
            
            # Display grouped sources
            for source_file, source_data_list in sources_by_file.items():
                # Display main file
                file_icon = "📦" if source_file.endswith('.zip') else "📄"
                st.markdown(f"**{file_icon} {source_file}**")
                
                # Display all chunks from this file
                for idx, source_data in enumerate(source_data_list, 1):
                    internal_file = source_data.get("internal_file")
                    page = source_data.get("page")
                    doc_type = source_data.get("type")
                    file_path = source_data.get("file_path")
                    
                    if internal_file and internal_file != "ZIP_MANIFEST":
                        # For files inside ZIP
                        st.write(
                            f"  └─ **{internal_file}** ({page}) `{doc_type.upper()}`"
                        )
                    elif internal_file == "ZIP_MANIFEST":
                        # For ZIP manifest
                        st.write(f"  └─ **ZIP Contents Index** `MANIFEST`")
                    else:
                        # For regular files
                        st.write(
                            f"  └─ **{file_path}** ({page}) `{doc_type.upper()}`"
                        )
            
            # 🔥 Additional summary
            st.markdown("---")
            st.markdown(f"**Total chunks analyzed:** {len(docs)}")

        st.session_state.messages.append({
            "role": "assistant",
            "content": response.content
        })