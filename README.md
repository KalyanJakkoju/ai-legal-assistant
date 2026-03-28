# ⚖️ AI Lawyer (Legal Assistant RAG Application)

AI Lawyer is a Retrieval-Augmented Generation (RAG) based web application built with Streamlit and LangChain. It allows users to upload complex legal documents (PDFs) and ask specific legal questions. The AI acts as an expert lawyer, providing structured and professional answers citing specific articles and reasoning directly referenced from the uploaded documents.

---

## 🌟 Features

- **Upload Legal Documents**: Upload any PDF file natively through the web interface.
- **AI-Powered Legal Expert**: Leverage the powerful `llama-3.3-70b-versatile` model (via Groq) to analyze text concisely.
- **Intelligent RAG Pipeline**: Uses advanced similarity search to find the most relevant context within your document to answer questions.
- **Fast Local Vector Caching**: Implements MD5 hash-based disk caching for FAISS vector stores, making subsequent document loads instantaneous.
- **Fast Embedding Generation**: Uses `FastEmbed` (ONNX-based BAAI/bge-small-en-v1.5) to quickly generate text embeddings on the CPU without the heavy PyTorch dependency or local Ollama servers.
- **Session Chat History**: Keeps track of the interactive conversation history within the same session.

---

## 🛠️ Technology Stack

- **Frontend Interface**: [Streamlit](https://streamlit.io/)
- **LLM/Orchestration Framework**: [LangChain](https://python.langchain.com/)
- **Language Model**: `ChatGroq` (`llama-3.3-70b-versatile`)
- **Embedding Model**: `FastEmbedEmbeddings` (`BAAI/bge-small-en-v1.5`) & `HuggingFaceEmbeddings`
- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **Document Loading**: `PyPDFLoader` (for speed) and `PDFPlumberLoader`

---

## 📂 Project Structure

- [main.py](cci:7://file:///c:/Users/JAKKOJU/Downloads/ai-legal-assistant/ai-legal-assistant/main.py:0:0-0:0) - The primary unified Streamlit application featuring optimized caching, fast embeddings, UI elements, and ChatGroq integration. **(Run this file to start the app)**
- [frontend.py](cci:7://file:///c:/Users/JAKKOJU/Downloads/ai-legal-assistant/ai-legal-assistant/frontend.py:0:0-0:0) - An alternative basic Streamlit UI skeleton that interacts with modular pipeline components.
- [rag_pipeline.py](cci:7://file:///c:/Users/JAKKOJU/Downloads/ai-legal-assistant/ai-legal-assistant/rag_pipeline.py:0:0-0:0) - Contains the logic to configure the LLM, retrieve documents, format prompt templates, and return answers.
- [vector_database.py](cci:7://file:///c:/Users/JAKKOJU/Downloads/ai-legal-assistant/ai-legal-assistant/vector_database.py:0:0-0:0) - Logic dedicated to loading raw PDFs, chunking text (via RecursiveCharacterTextSplitter), setting up embeddings, and indexing them into a local FAISS DB.
- [requirements.txt](cci:7://file:///c:/Users/JAKKOJU/Downloads/ai-legal-assistant/ai-legal-assistant/requirements.txt:0:0-0:0) / [Pipfile](cci:7://file:///c:/Users/JAKKOJU/Downloads/ai-legal-assistant/ai-legal-assistant/Pipfile:0:0-0:0) - Contains the Python dependencies.
- `pdfs/` - Directory where uploaded source PDFs are temporarily stored.
- `vectorstore/` - Directory caching FAISS vector embeddings to speed up repetitive document queries.

---

## 🚀 Getting Started

### 1. Prerequisites

Make sure you have Python 3.9+ installed and a valid API key for Groq.

### 2. Install Dependencies

You can install the dependencies using pip (or pipenv if using the Pipfile):

```bash
pip install -r requirements.txt
