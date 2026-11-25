import json
from pathlib import Path

from loader import load_knowledge_base
from chunker import chunk_documents
from embedder import embed_chunks, embed_query
from vector_store import build_index


def load_dataset():

    project_root = (
        Path(__file__).resolve().parent.parent
    )

    dataset_path = (
        project_root
        / "evaluation"
        / "qa_dataset.json"
    )

    with open(dataset_path, "r") as f:
        return json.load(f)


def retrieve_with_threshold(
    query,
    index,
    chunks,
    threshold,
    k=10
):

    query_embedding = embed_query(query)

    distances, indices = index.search(
        query_embedding.reshape(1, -1),
        k
    )

    results = []

    for rank, idx in enumerate(indices[0]):

        score = float(distances[0][rank])

        if score <= threshold:

            results.append(
                {
                    "document_name":
                        chunks[idx]["document_name"]
                }
            )

    return results


def calculate_recall(
    dataset,
    index,
    chunks,
    threshold
):

    hits = 0

    for sample in dataset:

        results = retrieve_with_threshold(
            query=sample["question"],
            index=index,
            chunks=chunks,
            threshold=threshold
        )

        retrieved_documents = {
            r["document_name"]
            for r in results
        }

        if (
            sample["expected_document"]
            in retrieved_documents
        ):
            hits += 1

    return hits / len(dataset)


if __name__ == "__main__":

    docs = load_knowledge_base()

    dataset = load_dataset()

    chunk_sizes = [
        200,
        400,
        600,
        800,
        1000
    ]

    thresholds = [
        0.60,
        0.70,
        0.80,
        0.90
    ]

    print("\nChunk Size Experiments")
    print("=" * 50)

    for chunk_size in chunk_sizes:

        chunks = chunk_documents(
            docs,
            chunk_size=chunk_size,
            overlap=100
        )

        embeddings = embed_chunks(chunks)

        index = build_index(embeddings)

        recall = calculate_recall(
            dataset,
            index,
            chunks,
            threshold=0.80
        )

        print(
            f"Chunk Size {chunk_size}: "
            f"Recall = {recall:.2%}"
        )

    print("\nThreshold Experiments")
    print("=" * 50)

    chunks = chunk_documents(
        docs,
        chunk_size=500,
        overlap=100
    )

    embeddings = embed_chunks(chunks)

    index = build_index(embeddings)

    for threshold in thresholds:

        recall = calculate_recall(
            dataset,
            index,
            chunks,
            threshold=threshold
        )

        print(
            f"Threshold {threshold}: "
            f"Recall = {recall:.2%}"
        )