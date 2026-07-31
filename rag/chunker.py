def chunk_text(text, chunk_size=2, overlap=1):
    """
    Split text into overlapping chunks.

    For now, each non-empty line is treated as one sentence.
    """

    sentences = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    stride = chunk_size - overlap

    chunks = []

    for i in range(0, len(sentences), stride):

        chunk = sentences[i:i + chunk_size]

        if not chunk:
            break

        chunks.append("\n".join(chunk))

        if i + chunk_size >= len(sentences):
            break

    return chunks