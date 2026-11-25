import ollama

from loader import load_knowledge_base
from chunker import chunk_documents
from embedder import embed_chunks
from vector_store import build_index
from retriever import retrieve
from prompt_builder import build_prompt


def generate_answer(prompt):
    """
    Generate answer using Phi3 Mini.
    """

    response = ollama.chat(
        model="phi3:mini",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]


def build_rag_system():

    print("Loading documents...")
    docs = load_knowledge_base()

    print("Chunking...")
    chunks = chunk_documents(docs)

    print("Creating embeddings...")
    embeddings = embed_chunks(chunks)

    print("Building FAISS index...")
    index = build_index(embeddings)

    return chunks, index


def ask_question(query, chunks, index):

    retrieved_chunks = retrieve(
        query=query,
        index=index,
        chunks=chunks,
        k=5
    )

    prompt = build_prompt(
        query=query,
        retrieved_chunks=retrieved_chunks
    )

    answer = generate_answer(prompt)

    return answer, retrieved_chunks


if __name__ == "__main__":

    chunks, index = build_rag_system()

    while True:

        query = input("\nAsk a question (or 'exit'): ")

        if query.lower() == "exit":
            break

        answer, retrieved_chunks = ask_question(
            query,
            chunks,
            index
        )

        print("\nRetrieved Sources:")
        print("-" * 50)

        for chunk in retrieved_chunks:

            print(
                f"{chunk['document_name']} | "
                f"Page {chunk['page_number']}"
            )

        print("\nAnswer:")
        print("-" * 50)

        print(answer)