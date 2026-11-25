from typing import List, Dict


def chunk_documents(
    documents: List[Dict],
    chunk_size: int = 500,
    overlap: int = 100
):
    """
    Split documents into overlapping chunks.
    """

    chunks = []
    chunk_id = 0

    for document in documents:

        document_name = document["document_name"]

        for page in document["pages"]:

            page_number = page["page_number"]
            text = page["text"]

            start = 0

            while start < len(text):

                end = start + chunk_size

                chunk_text = text[start:end]

                if chunk_text.strip():

                    chunks.append(
                        {
                            "chunk_id": chunk_id,
                            "document_name": document_name,
                            "page_number": page_number,
                            "text": chunk_text
                        }
                    )

                    chunk_id += 1

                start += chunk_size - overlap

    return chunks


if __name__ == "__main__":

    from loader import load_knowledge_base

    docs = load_knowledge_base()

    chunks = chunk_documents(
        docs,
        chunk_size=500,
        overlap=100
    )

    print(f"Total Chunks: {len(chunks)}")

    print("\nExample Chunk:\n")

    print(chunks[0])