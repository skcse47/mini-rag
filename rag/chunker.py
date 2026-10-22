import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


SECTION_PATTERN = re.compile(
    r"^\s*\d+\\?\.\s+.+$"
)


def split_into_sections(text):
    sections = []

    current_section = []

    for line in text.splitlines():

        line = line.strip()

        if not line:
            continue

        if SECTION_PATTERN.match(line):

            if current_section:
                sections.append(
                    "\n".join(current_section)
                )

            current_section = [line]

        else:
            current_section.append(line)

    if current_section:
        sections.append(
            "\n".join(current_section)
        )

    return sections


def chunk_pages(
    pages,
    chunk_size=1000,
    overlap=200
):
    """
    Create section-aware chunks from page Documents.

    Sections are detected using numbered headings such as:
    1. Company Overview
    2. NovaCRM
    6. Data Retention

    Small sections remain intact.
    Large sections are split with RecursiveCharacterTextSplitter.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    )

    chunks = []

    for page in pages:

        for section in split_into_sections(page.page_content):

            # If section fits within chunk size,
            # keep the entire section together.
            if len(section) <= chunk_size:

                chunks.append(
                    Document(
                        page_content=section,
                        metadata=dict(page.metadata)
                    )
                )

                continue

            chunks.extend(
                splitter.create_documents(
                    [section],
                    metadatas=[dict(page.metadata)]
                )
            )

    return chunks
