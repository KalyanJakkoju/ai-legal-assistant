# ⚖️ AI Lawyer (RAG with Groq & FastEmbed)

A lightning-fast, production-ready Retrieval-Augmented Generation (RAG) legal assistant. Upload massive legal PDFs (contracts, laws, case files) and instantly ask complex legal questions.

Powered by **Llama 3.3 70B** via Groq for high-speed reasoning, and **FastEmbed (ONNX)** for blazing-fast local vector embeddings.

![AI Lawyer Demo](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-Llama_3.3_70B-f55036?style=for-the-badge)
![FastEmbed](https://img.shields.io/badge/FastEmbed-BAAI/bge--small-blue?style=for-the-badge)

---

## ✨ Features

- ⚡ **Lightning Fast Inference:** Uses `llama-3.3-70b-versatile` running on Groq's LPUs for near-instant responses.
- 🚀 **Optimized Chunking & Embedding:** Swapped PyTorch/Ollama embeddings for **FastEmbed (ONNX)**, yielding 5x faster CPU embedding. Uses `PyPDFLoader` to process 400+ page PDFs in minutes.
- 🗄️ **Smart FAISS Disk Caching:** Vector stores are persistently cached using MD5 hashing. A 400-page PDF processes once; next upload is **instant (~1s)**.
- 💬 **Session Chat History:** Maintains fully context-aware conversational history in the UI.
- 📡 **Live Streaming:** Answers stream token-by-token directly into the UI with a live typing cursor.

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **LLM:** `llama-3.3-70b-versatile` (via Groq API)
- **Embeddings:** `BAAI/bge-small-en-v1.5` (via FastEmbed ONNX runtime — no GPU/PyTorch needed)
- **Vector Store:** FAISS (Facebook AI Similarity Search)
- **Orchestration:** LangChain

---

## ⚙️ Environment Setup

We recommend using **Miniconda** to manage your environment natively.

### 1. Create a Conda Environment

```bash
conda create -n ai-lawyer python=3.11 -y
conda activate ai-lawyer
```

### 2. Install Dependencies

Install the required packages. (This project removes heavy dependencies like `torch` and `ollama` in favor of streamlined, fast ONNX runtimes).

```bash
pip install streamlit langchain langchain-groq langchain-huggingface langchain-community pypdf fastembed faiss-cpu python-dotenv
```

### 3. API Keys (`.env`)

Create a `.env` file in the root directory and add your Groq API key:

```env
GROQ_API_KEY="your_groq_api_key_here"
```

_(Optional: If you hit SSL Certificate errors in Git Bash on Windows, run `unset SSL_CERT_FILE` before running the app)._

---

## 🚀 Running the App

Start the Streamlit application:

```bash
streamlit run main.py
```

1. Open your browser to `http://localhost:8501`.
2. Upload a legal PDF file using the sidebar.
3. Once processed (or loaded from cache instantly), type your legal query in the chat input.
4. Review the AI's structured analysis containing the direct answer, articles violated, reasoning, and conclusion!

---

## 📂 Project Structure

- `main.py` — The core Streamlit unified application (UI, Embedding, Caching, LLM Chain).
- `pdfs/` — Local directory where uploaded documents are strictly securely staged.
- `vectorstore/cache/` — Persistent MD5-hashed FAISS database caches to eliminate re-embedding delays.
- `requirements.txt` — Standard pip environment locks.
- `.env` — Environment configurations.

---

## 📝 License

This project is open-source and available under the MIT License.
