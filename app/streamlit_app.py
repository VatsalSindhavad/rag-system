from pathlib import Path
import sys

# Add src/ to python path
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
sys.path.append(str(SRC_DIR))

import streamlit as st
from httpx import ConnectError
from ollama import ResponseError

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="RAG Question Answering System",
    page_icon="📚",
    layout="wide"
)

# Custom CSS for Premium styling
st.markdown(
    """
    <style>
    /* Gradient Title */
    .gradient-text {
        background: linear-gradient(135deg, #FF4B4B, #FF8F8F, #4B79FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Outfit', 'Inter', sans-serif;
        font-weight: 800;
        font-size: clamp(2rem, 5vw, 3rem);
        margin-bottom: 0px;
    }

    /* Subtitle styling */
    .subtitle-text {
        font-family: 'Inter', sans-serif;
        opacity: 0.8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    /* Styled metrics and answers */
    .stAlert {
        border-radius: 10px;
    }

    /* Horizontal rule */
    hr {
        border: 0;
        height: 1px;
        background: linear-gradient(to right, rgba(255,255,255,0), rgba(255,255,255,0.1), rgba(255,255,255,0));
        margin: 2rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ==========================================
# CUSTOM RAG INITIALIZATION
# ==========================================
@st.cache_resource
def initialize_custom_rag():
    from chunker import chunk_documents
    from loader import load_knowledge_base

    docs = load_knowledge_base()
    chunks = chunk_documents(docs)
    if not chunks:
        raise ValueError("Add a PDF with extractable text to data/documents.")

    from embedder import embed_chunks
    from vector_store import build_index

    embeddings = embed_chunks(chunks)
    index = build_index(embeddings)
    return chunks, index

# ==========================================
# LANGCHAIN RAG INITIALIZATION
# ==========================================
@st.cache_resource
def initialize_langchain():
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_ollama import OllamaLLM
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    documents = []
    data_dir = BASE_DIR / "data" / "documents"

    for pdf_file in data_dir.glob("*.pdf"):
        loader = PyPDFLoader(str(pdf_file))
        documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)
    if not chunks:
        raise ValueError("Add a PDF with extractable text to data/documents.")

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 5}
    )

    llm = OllamaLLM(
        model="phi3:mini"
    )

    return retriever, llm

# ==========================================
# HELPER RUN FUNCTIONS
# ==========================================
def run_custom_rag(query, top_k=5):
    chunks, index = initialize_custom_rag()

    from prompt_builder import build_prompt
    from rag_pipeline import generate_answer
    from retriever import retrieve

    retrieved_chunks = retrieve(
        query=query,
        index=index,
        chunks=chunks,
        k=top_k
    )
    prompt = build_prompt(
        query=query,
        retrieved_chunks=retrieved_chunks
    )
    answer = generate_answer(prompt)
    return answer, retrieved_chunks

def run_langchain_rag(query, top_k=5):
    langchain_retriever, langchain_llm = initialize_langchain()
    docs = langchain_retriever.vectorstore.similarity_search(
        query,
        k=top_k
    )
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = f"""Answer ONLY using the provided context.

Context:
{context}

Question:
{query}

Answer:
"""
    answer = langchain_llm.invoke(prompt)
    return answer, docs


def run_pipeline(pipeline, label, query, top_k, answer_key, sources_key):
    st.session_state[answer_key] = None
    st.session_state[sources_key] = None
    try:
        answer, sources = pipeline(query, top_k)
    except (ConnectionError, ConnectError) as error:
        st.error(f"{label} could not connect. Check that Ollama is running. {error}")
    except ResponseError as error:
        st.error(f"{label} received an Ollama error. {error}")
        if error.status_code == 404:
            st.code("ollama pull phi3:mini", language="bash")
    except Exception as error:
        st.error(f"{label} failed. {error}")
    else:
        st.session_state[answer_key] = answer
        st.session_state[sources_key] = sources

# ==========================================
# SESSION STATE INITIALIZATION
# ==========================================
if "custom_answer" not in st.session_state:
    st.session_state.custom_answer = None
if "custom_sources" not in st.session_state:
    st.session_state.custom_sources = None
if "langchain_answer" not in st.session_state:
    st.session_state.langchain_answer = None
if "langchain_sources" not in st.session_state:
    st.session_state.langchain_sources = None
if "current_query" not in st.session_state:
    st.session_state.current_query = ""

# ==========================================
# SIDEBAR SETTINGS (Defined early so top_k is available globally)
# ==========================================
with st.sidebar:
    st.header("⚙️ Settings")
    top_k = st.slider(
        "Top K Retrieval",
        min_value=1,
        max_value=10,
        value=5
    )

    st.divider()
    st.header("📁 Loaded Documents")
    data_dir = BASE_DIR / "data" / "documents"
    pdf_files = list(data_dir.glob("*.pdf"))
    if pdf_files:
        for pdf in pdf_files:
            st.success(f"📄 {pdf.name}")
    else:
        st.warning("No PDF files found.")

    st.divider()
    st.header("📊 Project Details")

    st.success("Custom RAG")
    st.markdown("""
• Custom Loader
• Custom Chunker
• Custom Retriever
• FAISS Index L2
• Prompt Grounding
• Phi3 Mini
""")

    st.success("LangChain RAG")
    st.markdown("""
• PyPDFLoader
• RecursiveCharacterTextSplitter
• HuggingFaceEmbeddings
• FAISS Retriever
• Ollama Integration
• Phi3 Mini
""")

    st.divider()
    st.info(f"Current Top-K: {top_k}")

# ==========================================
# MAIN UI HEADER
# ==========================================
st.markdown('<p class="gradient-text">📚 Retrieval-Augmented Generation</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle-text">Compare a custom RAG implementation with a LangChain-based implementation.</p>', unsafe_allow_html=True)

query = st.text_input(
    "💬 Ask a question:",
    placeholder="e.g., What is gradient descent?"
)

# Reset results if the query is changed
if query != st.session_state.current_query:
    st.session_state.custom_answer = None
    st.session_state.custom_sources = None
    st.session_state.langchain_answer = None
    st.session_state.langchain_sources = None
    st.session_state.current_query = query

# ==========================================
# TABS SETUP
# ==========================================
tab1, tab2, tab3 = st.tabs(
    [
        "🛠️ Custom RAG",
        "🦜 LangChain RAG",
        "⚖️ Compare Both"
    ]
)

# ==========================================
# CUSTOM TAB
# ==========================================
with tab1:
    st.markdown("### 🛠️ Custom RAG Pipeline")
    st.write("This pipeline uses our custom loader, character chunker, numpy embeddings, and custom FAISS search.")

    if st.button("Run Custom RAG", key="btn_custom"):
        if query:
            with st.spinner("Preparing the index if needed, then generating an answer…"):
                run_pipeline(
                    run_custom_rag, "Custom RAG", query, top_k,
                    "custom_answer", "custom_sources"
                )
        else:
            st.warning("Please enter a question first.")

    if st.session_state.custom_answer is not None:
        st.markdown("#### 🤖 Answer")
        st.info(st.session_state.custom_answer)

        st.markdown("#### 🔍 Retrieved Sources")
        for i, chunk in enumerate(st.session_state.custom_sources, start=1):
            with st.expander(f"Source {i}: {chunk['document_name']} (Page {chunk['page_number']})"):
                st.write(chunk["text"])

# ==========================================
# LANGCHAIN TAB
# ==========================================
with tab2:
    st.markdown("### 🦜 LangChain RAG Pipeline")
    st.write("This pipeline uses PyPDFLoader, RecursiveCharacterTextSplitter, HuggingFaceEmbeddings, and LangChain's FAISS wrapper.")

    if st.button("Run LangChain RAG", key="btn_langchain"):
        if query:
            with st.spinner("Preparing the index if needed, then generating an answer…"):
                run_pipeline(
                    run_langchain_rag, "LangChain RAG", query, top_k,
                    "langchain_answer", "langchain_sources"
                )
        else:
            st.warning("Please enter a question first.")

    if st.session_state.langchain_answer is not None:
        st.markdown("#### 🤖 Answer")
        st.success(st.session_state.langchain_answer)

        st.markdown("#### 🔍 Retrieved Sources")
        for i, doc in enumerate(st.session_state.langchain_sources, start=1):
            doc_name = doc.metadata.get("source", "Unknown Document")
            if "/" in doc_name or "\\" in doc_name:
                doc_name = Path(doc_name).name
            page_num = doc.metadata.get("page", 0) + 1

            with st.expander(f"Source {i}: {doc_name} (Page {page_num})"):
                st.write(doc.page_content)

# ==========================================
# COMPARE TAB
# ==========================================
with tab3:
    st.markdown("### ⚖️ Side-by-Side Comparison")
    st.write("Compare the custom implementation directly with the LangChain implementation to analyze retrieval quality and model generation.")

    if st.button("Compare Pipelines", key="btn_compare"):
        if query:
            with st.spinner("Preparing both indexes if needed, then generating answers…"):
                run_pipeline(
                    run_custom_rag, "Custom RAG", query, top_k,
                    "custom_answer", "custom_sources"
                )
                run_pipeline(
                    run_langchain_rag, "LangChain RAG", query, top_k,
                    "langchain_answer", "langchain_sources"
                )
        else:
            st.warning("Please enter a question first.")

    if st.session_state.custom_answer is not None and st.session_state.langchain_answer is not None:
        st.subheader("Answer Comparison")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🛠️ Custom RAG")
            st.metric("Retrieved Chunks", len(st.session_state.custom_sources))
            st.info(st.session_state.custom_answer)

        with col2:
            st.markdown("#### 🦜 LangChain RAG")
            st.metric("Retrieved Chunks", len(st.session_state.langchain_sources))
            st.success(st.session_state.langchain_answer)

        st.markdown("---")

        col_src1, col_src2 = st.columns(2)

        with col_src1:
            st.markdown("#### 🛠️ Custom RAG Sources")
            for i, chunk in enumerate(st.session_state.custom_sources, start=1):
                with st.expander(f"Custom Source {i}: {chunk['document_name']} (Page {chunk['page_number']})"):
                    st.write(chunk["text"])

        with col_src2:
            st.markdown("#### 🦜 LangChain Sources")
            for i, doc in enumerate(st.session_state.langchain_sources, start=1):
                doc_name = doc.metadata.get("source", "Unknown Document")
                if "/" in doc_name or "\\" in doc_name:
                    doc_name = Path(doc_name).name
                page_num = doc.metadata.get("page", 0) + 1

                with st.expander(f"LangChain Source {i}: {doc_name} (Page {page_num})"):
                    st.write(doc.page_content)
