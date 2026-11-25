# 📚 Dual-Pipeline RAG System & Comparison Hub

A comparative Retrieval-Augmented Generation (RAG) project designed to evaluate and run two distinct pipeline architectures side-by-side: a **Custom RAG pipeline** implemented from scratch using Python, NumPy, and FAISS; and a **LangChain RAG pipeline** utilizing LangChain wrappers and community integrations. Both systems use a local **Phi-3 Mini** model powered by Ollama to answer questions based on a local PDF knowledge base.

An interactive **Streamlit web application** is provided to query, compare responses, explore retrieved source passages, and configure retrieval settings in real-time.

---

## 🚀 Key Features

*   **Custom RAG Pipeline (From Scratch)**:
    *   **Document Loading**: Custom parser extracting clean text page-by-page via `pypdf`.
    *   **Text Chunking**: Sliding-window character-based text splitter with configurable size and overlap.
    *   **Embeddings**: Embeds chunks using `sentence-transformers` (`all-MiniLM-L6-v2`) outputting standard NumPy arrays.
    *   **Vector Database**: Direct integration with a raw FAISS index (`IndexFlatL2`) for L2-distance vector search.
    *   **Retrieval**: Direct vector retrieval returning matching text, metadata, page numbers, and distance scores.
    *   **Prompt Grounding**: Contextually formatted system prompts to restrict answers to retrieved sources.
*   **LangChain RAG Pipeline**:
    *   **Document Loading**: Uses `PyPDFLoader` to load and parse document page structures.
    *   **Text Chunking**: Employs `RecursiveCharacterTextSplitter` for semantic structure preservation.
    *   **Embeddings**: Employs `HuggingFaceEmbeddings` wrapping the Hugging Face hub models.
    *   **Vector Database**: Uses the LangChain-native `FAISS` community vectorstore.
    *   **Ollama Integration**: Employs `OllamaLLM` to communicate with the local `phi3:mini` model.
*   **Real-time Comparative UI**:
    *   Sleek dark-themed Streamlit layout with tabbed interfaces.
    *   Interactive controls to adjust **Top-K** retrieval count dynamically.
    *   **Side-by-Side comparison mode** to view and compare answers, response times, and source references from both pipelines simultaneously.
*   **Evaluation & Parameter Tuning**:
    *   **Recall@K Evaluator**: Measures the accuracy of retrieval engines against a curated QA dataset.
    *   **Parameter Sweeps**: Automatic grid search over chunk sizes (200 to 1000 characters) and L2 distance thresholds (0.60 to 0.90) to discover configuration models that maximize Recall.

---

## 📂 Project Structure

```text
rag-system/
│
├── app/
│   └── streamlit_app.py        # Streamlit interactive UI application
│
├── src/
│   ├── loader.py               # Custom PDF document loader & text extractor
│   ├── chunker.py              # Custom character-based text splitter
│   ├── embedder.py             # SentenceTransformers wrapper (MiniLM)
│   ├── vector_store.py         # FAISS IndexFlatL2 manager
│   ├── retriever.py            # Custom vector query searcher
│   ├── prompt_builder.py       # Grounded prompt template constructor
│   ├── rag_pipeline.py         # CLI wrapper for the custom RAG pipeline
│   ├── evaluator.py            # Recall@K evaluation engine
│   └── experiments.py          # Grid search parameter tuner for chunk sizes/thresholds
│
├── langchain/
│   └── rag_chain.py            # LangChain CLI wrapper for the alternative pipeline
│
├── data/
│   └── documents/              # Local knowledge base (Store PDF files here)
│
├── evaluation/
│   └── qa_dataset.json         # Evaluator test set containing questions & expected docs
│
├── requirements.txt            # Python environment dependencies
├── test_ollama.py              # Verification script for local Ollama API
└── README.md                   # Project documentation
```

---

## 🛠️ Prerequisites & Setup

### 1. Install & Configure Ollama
This project utilizes a local LLM to guarantee data privacy and zero API costs.
1. Download and install [Ollama](https://ollama.com/).
2. Start the Ollama application.
3. Open your terminal or powershell and pull the **Phi-3 Mini (3.8B)** model:
   ```bash
   ollama pull phi3:mini
   ```
4. Verify the model is running:
   ```bash
   python test_ollama.py
   ```

### 2. Set Up Python Environment
1. Clone or navigate to the project root directory:
   ```bash
   cd rag-system
   ```
2. Create and activate a virtual environment:
   ```bash
   # Windows (PowerShell)
   python -m venv venv
   .\venv\Scripts\Activate.ps1

   # Linux/macOS
   python -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

*(Note: If your `requirements.txt` is empty, please ensure the following libraries are installed)*:
```bash
pip install streamlit ollama langchain-community langchain-text-splitters langchain-huggingface langchain-ollama faiss-cpu sentence-transformers numpy pypdf
```

---

## 🎮 How to Run

Create `data/documents/` and add PDFs with extractable text before running either pipeline. The bundled evaluation questions expect the four filenames listed in `evaluation/qa_dataset.json`.

Create `data/documents/` and add PDFs with extractable text before running either pipeline. The bundled evaluation questions expect the filenames listed in `evaluation/qa_dataset.json`.

### Run the Interactive Web UI
Launch the comparative web panel to query the system with a browser-based UI:
```bash
python -m streamlit run app/streamlit_app.py
```
* Once launched, open `http://localhost:8501` in your browser.
* Each pipeline builds its index on first use, which can take a minute or more with your local PDFs. The first run needs access to Hugging Face if `all-MiniLM-L6-v2` is absent from the local cache. Ollama must be running with `phi3:mini` available to generate answers.
* If the embedding model is already cached and you are offline, set `$env:HF_HUB_OFFLINE = "1"` in PowerShell before launching Streamlit.
* Use the **Sidebar** to view loaded PDFs and adjust the **Top-K** slider.
* Ask questions in the input field and switch between tabs to inspect **Custom RAG**, **LangChain RAG**, or the **Side-by-Side Comparison**.

### Run CLI Pipelines
You can also run both pipelines inside your command terminal:

*   **Run Custom RAG CLI**:
    ```bash
    python src/rag_pipeline.py
    ```
*   **Run LangChain RAG CLI**:
    ```bash
    python langchain/rag_chain.py
    ```
Both scripts will load the PDFs, initialize the indexes, and open a command-line prompt: `Ask a question (or 'exit')`.

---

## 📊 Evaluation & Parameter Optimization

To assess the quality of retrieval components separate from text generation, the project uses a test dataset mapping user queries to ground truth documents.

### Run Retrieval Evaluation (Recall@K)
Run the evaluator to test retrieval success rates at $K \in \{1, 3, 5, 10\}$:
```bash
python src/evaluator.py
```
This tests the Custom Retriever against the questions in `evaluation/qa_dataset.json` and prints the retrieval Recall metrics.

### Run Parameter Sweeps
Optimize retrieval parameters by testing different chunk sizes and similarity thresholds:
```bash
python src/experiments.py
```
The script runs experiments over:
*   **Chunk sizes**: `[200, 400, 600, 800, 1000]` characters.
*   **L2 Distance Thresholds**: `[0.60, 0.70, 0.80, 0.90]`.
The configuration scoring the highest Recall index highlights the optimal settings for your specific document database.

---

## ⚙️ Configuration & Tunable Parameters

| Parameter | Location | Default Value | Description |
| :--- | :--- | :--- | :--- |
| `chunk_size` | `src/chunker.py` / `streamlit_app.py` | `500` | Target character length of text splits. |
| `overlap` | `src/chunker.py` / `streamlit_app.py` | `100` | Overlap character length between sequential chunks. |
| `k` (Top-K) | `app/streamlit_app.py` | `5` (configurable via slider) | Number of source documents sent to LLM context. |
| `embedding_model` | `src/embedder.py` | `"all-MiniLM-L6-v2"` | SentenceTransformers model for custom pipeline embeddings. |
| `llm_model` | `app/streamlit_app.py` | `"phi3:mini"` | Name of the local LLM model running under Ollama. |
