from rag.document_loader import load_pdf
from rag.chunker import chunk_pages
from rag.embedder import create_embeddings
from rag.document import create_document_id
from rag.qdrant_store import insert_documents, create_collection
from rag.qdrant_store import close_qdrant


PDF_PATH = "data/Leave-Policy.pdf"


document_id = create_document_id()

pages = load_pdf(PDF_PATH)

chunks = chunk_pages(
    pages,
    chunk_size=4,
    overlap=1
)

texts = [
    chunk["text"]
    for chunk in chunks
]

embeddings = create_embeddings(texts)

create_collection()

insert_documents(
    chunks=chunks,
    embeddings=embeddings,
    document_id=document_id,
    filename="Leave-Policy.pdf"
)

print(
    f"Document indexed successfully: {document_id}"
)

close_qdrant()