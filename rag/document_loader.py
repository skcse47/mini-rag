import fitz
from langchain_core.documents import Document


def load_pdf(file_path):
    """
    Load a PDF as one LangChain Document per non-empty page.
    """

    document = fitz.open(file_path)

    pages = []

    for page_number, page in enumerate(document):

        text = page.get_text().strip()

        if not text:
            continue

        pages.append(
            Document(
                page_content=text,
                metadata={"page": page_number + 1}
            )
        )

    document.close()

    return pages
