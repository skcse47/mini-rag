from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """
Python is a programming language It is widely used for AI It is also used for web development It has a simple syntax.
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=60,
    chunk_overlap=20
)

chunks = splitter.split_text(text)

for i, chunk in enumerate(chunks, start=1):
    print(f"\nChunk {i}")
    print("-" * 20)
    print(chunk)