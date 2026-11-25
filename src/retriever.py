from embedder import embed_query


def retrieve(query, index, chunks, k=5):
    """
    Retrieve top-k relevant chunks.
    """

    query_embedding = embed_query(query)

    distances, indices = index.search(
        query_embedding.reshape(1, -1),
        k
    )

    results = []

    for rank, idx in enumerate(indices[0]):

        chunk = chunks[idx]

        results.append(
            {
                "rank": rank + 1,
                "score": float(distances[0][rank]),
                "document_name": chunk["document_name"],
                "page_number": chunk["page_number"],
                "text": chunk["text"]
            }
        )

    return results


if __name__ == "__main__":

    from loader import load_knowledge_base
    from chunker import chunk_documents
    from embedder import embed_chunks
    from vector_store import build_index

    docs = load_knowledge_base()

    chunks = chunk_documents(docs)

    embeddings = embed_chunks(chunks)

    index = build_index(embeddings)

    query = "What is gradient descent?"

    results = retrieve(
        query=query,
        index=index,
        chunks=chunks,
        k=5
    )

    print("\nQuestion:")
    print(query)

    print("\nTop Results:\n")

    for result in results:

        print("=" * 80)

        print(
            f"Rank: {result['rank']}"
        )

        print(
            f"Source: {result['document_name']}"
        )

        print(
            f"Page: {result['page_number']}"
        )

        print(
            f"Score: {result['score']:.4f}"
        )

        print()

        print(
            result["text"][:400]
        )