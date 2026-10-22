from rag.document_loader import load_pdf
from rag.chunker import chunk_pages
from rag.document import create_document_id
from rag.qdrant_store import insert_documents
from rag.qdrant_store import close_qdrant


PDF_PATH = "data/Leave-Policy.pdf"


document_id = create_document_id()

pages = load_pdf(PDF_PATH)

chunks = chunk_pages(
    pages,
    chunk_size=1000,
    overlap=200
)

insert_documents(
    chunks=chunks,
    document_id=document_id,
    filename="Leave-Policy.pdf"
)

print(
    f"Document indexed successfully: {document_id}"
)

close_qdrant()