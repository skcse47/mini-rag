from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter, 
    FieldCondition,
    MatchValue
)
import uuid
from rag.config import COLLECTION_NAME, EMBEDDING_DIMENSION 

COLLECTION_NAME = COLLECTION_NAME

client = QdrantClient(
    path="storage/qdrant"
)

def create_collection():

    collections = client.get_collections()

    existing_collections = [
        collection.name
        for collection in collections.collections
    ]

    if COLLECTION_NAME not in existing_collections:

        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=EMBEDDING_DIMENSION,
                distance=Distance.COSINE
            )
        )


def insert_documents(
    chunks,
    embeddings,
    document_id,
    filename
):

    points = []

    for idx, (chunk, embedding) in enumerate(
        zip(chunks, embeddings)
    ):

        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "text": chunk["text"],
                    "metadata": {
                        "document_id": document_id,
                        "filename": filename,
                        "page": chunk["page"]
                    }
                }
            )
        )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )



def search(
    query_embedding,
    top_k=3,
    document_id=None
):

    query_filter = None

    if document_id:

        query_filter = Filter(
            must=[
                FieldCondition(
                    key="metadata.document_id",
                    match=MatchValue(
                        value=document_id
                    )
                )
            ]
        )

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        query_filter=query_filter,
        limit=top_k
    )

    return results.points

def list_documents():

    results = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=1000,
        with_payload=True,
        with_vectors=False
    )

    points, _ = results

    documents = {}

    for point in points:

        metadata = point.payload.get(
            "metadata",
            {}
        )

        document_id = metadata.get(
            "document_id"
        )

        filename = metadata.get(
            "filename"
        )

        if document_id and document_id not in documents:

            documents[document_id] = {
                "document_id": document_id,
                "filename": filename
            }

    return list(documents.values())


def delete_document(document_id):

    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=Filter(
            must=[
                FieldCondition(
                    key="metadata.document_id",
                    match=MatchValue(
                        value=document_id
                    )
                )
            ]
        )
    )
    

def close_qdrant():
    client.close()