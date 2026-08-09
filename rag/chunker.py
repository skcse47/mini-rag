def chunk_pages(pages, chunk_size=4, overlap=1):
    """
    Create overlapping chunks while preserving page numbers.
    """

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than 0"
        )

    if overlap >= chunk_size:
        raise ValueError(
            "overlap must be smaller than chunk_size"
        )

    chunks = []

    for page in pages:

        lines = [
            line.strip()
            for line in page["text"].splitlines()
            if line.strip()
        ]

        stride = chunk_size - overlap

        for i in range(
            0,
            len(lines),
            stride
        ):

            chunk_lines = lines[
                i:i + chunk_size
            ]

            if not chunk_lines:
                break

            chunks.append({
                "text": "\n".join(chunk_lines),
                "page": page["page"]
            })

            if i + chunk_size >= len(lines):
                break

    return chunks