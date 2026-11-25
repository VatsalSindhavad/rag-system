from sentence_transformers import SentenceTransformer
import numpy as np


model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


def embed_chunks(chunks):
    """
    Convert chunks into embeddings.
    """

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        show_progress_bar=True
    )

    return np.array(embeddings)


def embed_query(query):
    """
    Convert user query into embedding.
    """

    return model.encode(query)


if __name__ == "__main__":

    from loader import load_knowledge_base
    from chunker import chunk_documents

    docs = load_knowledge_base()

    chunks = chunk_documents(docs)

    embeddings = embed_chunks(chunks)

    print("\nEmbedding Shape:")
    print(embeddings.shape)

    print("\nExample Vector:")
    print(embeddings[0][:10])