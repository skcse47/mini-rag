def chunk_pages(
    pages,
    chunk_size=1000,
    overlap=200
):
    """
    Create chunks from PDF pages while preserving page numbers.

    chunk_size and overlap are measured in characters.
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

        paragraphs = [
            paragraph.strip()
            for paragraph in page["text"].split("\n\n")
            if paragraph.strip()
        ]

        current_chunk = ""

        for paragraph in paragraphs:

            # If adding the paragraph stays within
            # the target size, keep building the chunk.
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

                # Start a new chunk.
                current_chunk = paragraph

        # Add the final chunk from the page.
        if current_chunk:

            chunks.append({
                "text": current_chunk,
                "page": page["page"]
            })

    return chunks