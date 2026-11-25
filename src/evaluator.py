import json
from pathlib import Path

from loader import load_knowledge_base
from chunker import chunk_documents
from embedder import embed_chunks
from vector_store import build_index
from retriever import retrieve


def load_dataset(
    dataset_path=Path(__file__).resolve().parents[1] / "evaluation" / "qa_dataset.json"
):
    with open(dataset_path, "r") as f:
        return json.load(f)


def calculate_recall_at_k(
    dataset,
    chunks,
    index,
    k
):
    hits = 0

    for sample in dataset:

        question = sample["question"]

        expected_document = sample[
            "expected_document"
        ]

        results = retrieve(
            query=question,
            index=index,
            chunks=chunks,
            k=k
        )

        retrieved_documents = {
            result["document_name"]
            for result in results
        }

        if expected_document in retrieved_documents:
            hits += 1

    return hits / len(dataset)


if __name__ == "__main__":

    docs = load_knowledge_base()

    chunks = chunk_documents(docs)

    embeddings = embed_chunks(chunks)

    index = build_index(embeddings)

    dataset = load_dataset()

    for k in [1, 3, 5, 10]:

        recall = calculate_recall_at_k(
            dataset,
            chunks,
            index,
            k
        )

        print(
            f"Recall@{k}: {recall:.2%}"
        )