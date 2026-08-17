import re


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
    Create section-aware chunks from PDF pages.

    Sections are detected using numbered headings such as:
    1. Company Overview
    2. NovaCRM
    6. Data Retention

    Small sections remain intact.
    Large sections are split into paragraph-based chunks.
    """

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than 0"
        )

    if overlap < 0:
        raise ValueError(
            "overlap cannot be negative"
        )

    if overlap >= chunk_size:
        raise ValueError(
            "overlap must be smaller than chunk_size"
        )

    chunks = []

    for page in pages:

        sections = split_into_sections(
            page["text"]
        )

        for section in sections:

            # If section fits within chunk size,
            # keep the entire section together.
            if len(section) <= chunk_size:

                chunks.append({
                    "text": section,
                    "page": page["page"]
                })

                continue

            # Large section → split into paragraphs.
            paragraphs = [
                paragraph.strip()
                for paragraph in section.split("\n\n")
                if paragraph.strip()
            ]

            current_chunk = ""

            for paragraph in paragraphs:

                if (
                    len(current_chunk)
                    + len(paragraph)
                    + 2
                    <= chunk_size
                ):

                    if current_chunk:
                        current_chunk += "\n\n"

                    current_chunk += paragraph

                else:

                    if current_chunk:

                        chunks.append({
                            "text": current_chunk,
                            "page": page["page"]
                        })

                    overlap_text = (
                        current_chunk[-overlap:]
                        if current_chunk
                        else ""
                    )

                    current_chunk = (
                        overlap_text
                        + "\n\n"
                        + paragraph
                        if overlap_text
                        else paragraph
                    )

            if current_chunk:

                chunks.append({
                    "text": current_chunk,
                    "page": page["page"]
                })

    return chunks