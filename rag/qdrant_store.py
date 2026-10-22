from datetime import datetime, timezone
from functools import lru_cache
import uuid

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue
)

from rag.config import (
    COLLECTION_NAME,
    REGISTRY_COLLECTION_NAME,
    EMBEDDING_DIMENSION,
    QDRANT_PATH,
)
from rag.embedder import get_embeddings


client = QdrantClient(
    path=QDRANT_PATH
)


def _collection_exists(name):

    return client.collection_exists(name)


def create_collection():
    """
    Ensure the chunk collection and the document registry exist.
    The registry holds one point per document so listing documents
    never has to scan every chunk.
    """

    if not _collection_exists(COLLECTION_NAME):

        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=EMBEDDING_DIMENSION,
                distance=Distance.COSINE
            )
        )

    if not _collection_exists(REGISTRY_COLLECTION_NAME):

        client.create_collection(
            collection_name=REGISTRY_COLLECTION_NAME,
            # Registry points are looked up by id/scroll, never by
            # similarity, so a 1-dim placeholder vector is enough.
            vectors_config=VectorParams(
                size=1,
                distance=Distance.DOT
            )
        )

        _backfill_registry()


@lru_cache(maxsize=1)
def get_vector_store():
    """
    LangChain vector store over the chunk collection. The payload keys
    match the layout used before the LangChain migration, so existing
    chunks stay searchable without re-indexing.
    """

    create_collection()

    return QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=get_embeddings(),
        content_payload_key="text",
        metadata_payload_key="metadata",
    )


def document_filter(document_id):

    return Filter(
        must=[
            FieldCondition(
                key="metadata.document_id",
                match=MatchValue(
                    value=document_id
                )
            )
        ]
    )


def insert_documents(
    chunks,
    document_id,
    filename
):
    """
    Embed and store chunk Documents, then register the document.
    """

    for chunk in chunks:

        chunk.metadata.update({
            "document_id": document_id,
            "filename": filename,
        })

    get_vector_store().add_documents(
        chunks,
        ids=[str(uuid.uuid4()) for _ in chunks]
    )

    register_document(
        document_id,
        filename,
        chunk_count=len(chunks)
    )


def search(
    query,
    top_k=3,
    document_id=None
):
    """
    Return (Document, score) pairs for the query, optionally
    restricted to a single document.
    """

    return get_vector_store().similarity_search_with_score(
        query,
        k=top_k,
        filter=document_filter(document_id) if document_id else None
    )


def register_document(document_id, filename, chunk_count=None):

    client.upsert(
        collection_name=REGISTRY_COLLECTION_NAME,
        points=[
            PointStruct(
                id=document_id,
                vector=[1.0],
                payload={
                    "document_id": document_id,
                    "filename": filename,
                    "chunk_count": chunk_count,
                    "created_at": datetime.now(timezone.utc).isoformat()
                }
            )
        ]
    )


def _backfill_registry():
    """
    Register documents that were indexed before the registry existed,
    paging through every chunk once.
    """

    if not _collection_exists(COLLECTION_NAME):
        return

    documents = {}
    offset = None

    while True:

        points, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=1000,
            offset=offset,
            with_payload=["metadata"],
            with_vectors=False
        )

        for point in points:

            metadata = point.payload.get("metadata", {})
            document_id = metadata.get("document_id")

            if not document_id:
                continue

            entry = documents.setdefault(
                document_id,
                {"filename": metadata.get("filename"), "chunk_count": 0}
            )
            entry["chunk_count"] += 1

        if offset is None:
            break

    for document_id, entry in documents.items():

        register_document(
            document_id,
            entry["filename"],
            chunk_count=entry["chunk_count"]
        )


def list_documents():

    create_collection()

    documents = []
    offset = None

    while True:

        points, offset = client.scroll(
            collection_name=REGISTRY_COLLECTION_NAME,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False
        )

        for point in points:

            documents.append({
                "document_id": point.payload["document_id"],
                "status": "ready",
                "filename": point.payload.get("filename")
            })

        if offset is None:
            break

    return documents


def delete_document(document_id):

    create_collection()

    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=document_filter(document_id)
    )

    client.delete(
        collection_name=REGISTRY_COLLECTION_NAME,
        points_selector=[document_id]
    )


def close_qdrant():
    client.close()
