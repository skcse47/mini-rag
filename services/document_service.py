from rag.document import create_document_id
from rag.document_loader import load_pdf
from rag.chunker import chunk_pages
from rag.qdrant_store import insert_documents, delete_document


def index_document(file_path, filename):

    document_id = create_document_id()

    pages = load_pdf(file_path)

    chunks = chunk_pages(
        pages,
        chunk_size=1000,
        overlap=200
    )

    insert_documents(
        chunks=chunks,
        document_id=document_id,
        filename=filename
    )

    return document_id

def remove_document(document_id):
    delete_document(document_id)

    return {
        "document_id": document_id,
        "status": "deleted"
    }
