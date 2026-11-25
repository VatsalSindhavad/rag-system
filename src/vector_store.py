import faiss
import numpy as np


def build_index(embeddings):
    """
    Build FAISS index from embeddings.
    """

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(
        embeddings.astype(np.float32)
    )

    return index


if __name__ == "__main__":

    from loader import load_knowledge_base
    from chunker import chunk_documents
    from embedder import embed_chunks

    docs = load_knowledge_base()

    chunks = chunk_documents(docs)

    embeddings = embed_chunks(chunks)

    index = build_index(embeddings)

    print("\nFAISS Index Created")

    print(
        f"Vectors Stored: {index.ntotal}"
    )