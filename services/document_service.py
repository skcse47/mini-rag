from rag.document import create_document_id
from rag.document_loader import load_pdf
from rag.chunker import chunk_pages
from rag.embedder import create_embeddings
from rag.qdrant_store import create_collection, insert_documents, delete_document


def index_document(file_path, filename):

    document_id = create_document_id()

    pages = load_pdf(file_path)

    chunks = chunk_pages(
        pages,
        chunk_size=1000,
        overlap=200
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
        filename=filename
    )

    return document_id

def remove_document(document_id):
    delete_document(document_id)

    return {
        "document_id": document_id,
        "status": "deleted"
    }