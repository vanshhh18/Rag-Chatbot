# Rag-Chatbot
Link for the chatbot:
https://rag-chatbot-wjdvvsnltezmebj9xvkrkr.streamlit.app/

# RAG Chatbot

A Retrieval-Augmented Generation chatbot built with **LangChain**, **HuggingFace**, **FAISS**, and **Groq** for ultra-fast LLM inference.

## Features

- Ingest custom documents (PDF, TXT, etc.)
- Semantic search via FAISS vector store
- HuggingFace embeddings for document encoding
- **Groq** for blazing-fast LLM inference
- Conversational memory with LangChain chains

## Setup

```bash
git clone https://github.com/your-username/rag-chatbot.git
cd rag-chatbot
pip install -r requirements.txt
cp .env.example .env  # Add your Groq & HuggingFace API tokens
```

## Usage

```bash
# Ingest documents
python src/ingest.py

# Run chatbot
python src/chatbot.py
```

## Tech Stack

| Layer       | Tool                                      |
|-------------|-------------------------------------------|
| Framework   | LangChain                                 |
| Embeddings  | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector DB   | FAISS                                     |
| LLM         | Groq (e.g. `llama3-8b-8192`)             |

## Requirements

```
langchain langchain-community langchain-groq
faiss-cpu sentence-transformers
huggingface_hub pypdf python-dotenv
```

## Environment Variables

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL_NAME=llama3-8b-8192
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

## License

MIT
