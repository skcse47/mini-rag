def chunk_text(text):
    """
    Temporary chunker.

    Later we'll implement recursive chunking,
    overlap, token-based chunking, etc.
    """

    return [
        line.strip()
        for line in text.split("\n")
        if line.strip()
    ]