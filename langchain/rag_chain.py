from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "documents"


def load_documents():
    documents = []

    for pdf_file in DATA_DIR.glob("*.pdf"):
        loader = PyPDFLoader(str(pdf_file))
        documents.extend(loader.load())

    return documents


def build_rag():

    documents = load_documents()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

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


if __name__ == "__main__":

    retriever, llm = build_rag()

    while True:

        query = input(
            "\nAsk a question (or 'exit'): "
        )

        if query.lower() == "exit":
            break

        docs = retriever.invoke(query)

        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

        prompt = f"""
Answer using only the context below.

Context:
{context}

Question:
{query}

Answer:
"""

        answer = llm.invoke(prompt)

        print("\nAnswer:")
        print("-" * 50)
        print(answer)