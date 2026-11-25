from pathlib import Path
from pypdf import PdfReader


# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

# data/documents folder
DATA_DIR = BASE_DIR / "data" / "documents"


def load_knowledge_base():
    """
    Load all PDFs from data/documents

    Returns:
        list of document dictionaries
    """

    documents = []

    pdf_files = list(DATA_DIR.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in:\n{DATA_DIR}")
        return documents

    for pdf_path in pdf_files:
        try:
            reader = PdfReader(pdf_path)

            pages = []

            for page_num, page in enumerate(reader.pages, start=1):

                text = page.extract_text()

                if text and text.strip():

                    pages.append(
                        {
                            "page_number": page_num,
                            "text": text.strip()
                        }
                    )

            documents.append(
                {
                    "document_name": pdf_path.name,
                    "pages": pages
                }
            )

            print(
                f"Loaded: {pdf_path.name} | Pages: {len(pages)}"
            )

        except Exception as e:
            print(
                f"Failed to load {pdf_path.name}: {e}"
            )

    return documents


if __name__ == "__main__":

    print("=" * 50)
    print("RAG Document Loader")
    print("=" * 50)

    print(f"\nSearching in:\n{DATA_DIR}\n")

    docs = load_knowledge_base()

    print(f"\nLoaded {len(docs)} documents")

    if docs:

        print("\nExample Document:")
        print("-" * 50)

        print("Name:", docs[0]["document_name"])
        print("Pages:", len(docs[0]["pages"]))

        if docs[0]["pages"]:

            print("\nFirst Page Preview:")
            print("-" * 50)

            preview = docs[0]["pages"][0]["text"][:500]

            print(preview)