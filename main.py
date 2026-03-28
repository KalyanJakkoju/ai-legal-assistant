import os
import hashlib
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader       # Fast PDF loader
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings  # ONNX-based, no torch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Lawyer", page_icon="⚖️", layout="wide")
st.title("⚖️ AI Lawyer")
st.caption("Upload a legal document and ask questions. Chat history is saved per session.")

# ── Constants ──────────────────────────────────────────────────────────────────
FAISS_CACHE_DIR = "vectorstore/cache"   # persistent per-file cache on disk
PDFS_DIR        = "pdfs/"

os.makedirs(FAISS_CACHE_DIR, exist_ok=True)
os.makedirs(PDFS_DIR, exist_ok=True)

# ── Prompt ────────────────────────────────────────────────────────────────────
PROMPT = ChatPromptTemplate.from_template("""
You are an expert AI Lawyer. Use the context provided to answer the user's legal question \
in a clear, structured, and professional manner.

Format your response as follows:
- Start with a direct answer to the question
- List the specific articles or laws violated (if any), with their titles
- Explain WHY each article is violated with brief reasoning
- End with a brief conclusion

If the answer is not in the context, say: "I don't have enough information in the provided \
document to answer this question."

Question: {question}
Context: {context}
Answer:
""")

# ── Cached singletons ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_embedding_model():
    """
    FastEmbed uses ONNX runtime — no PyTorch needed.
    ~3-5x faster than sentence-transformers on CPU.
    BAAI/bge-small-en-v1.5 is small (130MB) and accurate.
    """
    return FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5", max_length=512)

@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatGroq(model="llama-3.3-70b-versatile", streaming=True)

# ── Per-file vector store: persistent FAISS cache by MD5 hash ─────────────────
def get_file_hash(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()

def load_or_build_vector_store(file_name: str, file_bytes: bytes) -> FAISS:
    """
    • Hash-based disk cache — same PDF loads in ~1 second from disk.
    • PyPDFLoader: 10x faster than PDFPlumber for large docs.
    • Larger chunks (3000 chars) → fewer embeddings → faster build.
    • FastEmbed ONNX: 3-5x faster embedding than torch on CPU.
    """
    file_hash  = get_file_hash(file_bytes)
    cache_path = os.path.join(FAISS_CACHE_DIR, file_hash)

    if os.path.exists(cache_path):
        return FAISS.load_local(
            cache_path, get_embedding_model(), allow_dangerous_deserialization=True
        )

    # First-time build — show a progress UI
    with st.status("⚙️ Processing PDF (first time only)…", expanded=True) as status:

        st.write("📄 Loading pages…")
        loader    = PyPDFLoader(PDFS_DIR + file_name)   # much faster than PDFPlumber
        documents = loader.load()
        st.write(f"   → Loaded {len(documents)} pages")

        st.write("✂️ Splitting into chunks…")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=3000,      # bigger chunks = fewer embeddings = faster
            chunk_overlap=100,
            add_start_index=True,
        )
        chunks = splitter.split_documents(documents)
        st.write(f"   → {len(chunks)} chunks created")

        st.write("🔢 Embedding chunks via ONNX (FastEmbed)…")
        faiss_db = FAISS.from_documents(chunks, get_embedding_model())
        faiss_db.save_local(cache_path)

        status.update(
            label="✅ Done! Future loads of this PDF will be instant.", state="complete"
        )

    return faiss_db

# ── Session state: chat history ───────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_file_hash" not in st.session_state:
    st.session_state.current_file_hash = None

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📄 Document")
    uploaded_file = st.file_uploader("Upload PDF", type="pdf", accept_multiple_files=False)

    if uploaded_file:
        file_bytes = uploaded_file.getvalue()
        file_hash  = get_file_hash(file_bytes)
        cache_path = os.path.join(FAISS_CACHE_DIR, file_hash)
        if os.path.exists(cache_path):
            st.success(f"✅ **{uploaded_file.name}** — cached, ready instantly!")
        else:
            st.info(f"📋 **{uploaded_file.name}** — will be processed on first question.")

    st.divider()
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# ── Display chat history ──────────────────────────────────────────────────────
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat input ────────────────────────────────────────────────────────────────
user_query = st.chat_input("Ask your legal question…")

if user_query:
    if not uploaded_file:
        st.warning("⚠️ Please upload a PDF document first (use the sidebar).")
    else:
        # Save PDF to disk
        file_bytes = uploaded_file.getvalue()
        with open(PDFS_DIR + uploaded_file.name, "wb") as f:
            f.write(file_bytes)

        # Show user message immediately
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.chat_history.append({"role": "user", "content": user_query})

        # Load / build vector store
        faiss_db = load_or_build_vector_store(uploaded_file.name, file_bytes)

        # Retrieve top 3 relevant chunks
        retrieved_docs = faiss_db.similarity_search(user_query, k=3)
        context = "\n\n".join(doc.page_content for doc in retrieved_docs)

        # Stream the answer
        chain = PROMPT | get_llm()
        with st.chat_message("AI Lawyer"):
            placeholder   = st.empty()
            full_response = ""
            for chunk in chain.stream({"question": user_query, "context": context}):
                full_response += chunk.content
                placeholder.markdown(full_response + "▌")
            placeholder.markdown(full_response)

        st.session_state.chat_history.append(
            {"role": "AI Lawyer", "content": full_response}
        )
