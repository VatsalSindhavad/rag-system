def build_prompt(query, retrieved_chunks):
    """
    Build grounded prompt using retrieved context.
    """

    context = "\n\n".join(
        [
            f"[Source: {chunk['document_name']} | Page: {chunk['page_number']}]\n{chunk['text']}"
            for chunk in retrieved_chunks
        ]
    )

    prompt = f"""
You are an AI assistant.

Answer ONLY using the provided context.

If the answer is not present in the context, say:

"I could not find this information in the provided documents."

Context:
{context}

Question:
{query}

Answer:
"""

    return prompt


if __name__ == "__main__":

    from loader import load_knowledge_base
    from chunker import chunk_documents
    from embedder import embed_chunks
    from vector_store import build_index
    from retriever import retrieve

    docs = load_knowledge_base()

    chunks = chunk_documents(docs)

    embeddings = embed_chunks(chunks)

    index = build_index(embeddings)

    query = "What is gradient descent?"

    results = retrieve(
        query=query,
        index=index,
        chunks=chunks,
        k=3
    )

    prompt = build_prompt(
        query=query,
        retrieved_chunks=results
    )

    print(prompt)