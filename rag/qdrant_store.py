from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter, 
    FieldCondition,
    MatchValue
)

COLLECTION_NAME = "company_documents"

client = QdrantClient(
    path="storage/qdrant"
)


def create_collection():

    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=768,
            distance=Distance.COSINE
        )
    )


def insert_documents(chunks, embeddings):

    points = []

    for idx, (chunk, embedding) in enumerate(
        zip(chunks, embeddings)
    ):

        points.append(
            PointStruct(
                id=idx,
                vector=embedding,
                payload={
                    "text": chunk,
                    "metadata": {
                        "source": "company.txt",
                        "department": "HR",
                        "document_type": "policy"
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
    metadata_filter=None
):

    query_filter = None

    if metadata_filter:

        conditions = []

        for key, value in metadata_filter.items():

            conditions.append(
                FieldCondition(
                    key=f"metadata.{key}",
                    match=MatchValue(value=value)
                )
            )

        query_filter = Filter(
            must=conditions
        )

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        query_filter=query_filter,
        limit=top_k
    )

    return results.points

def close_qdrant():
    client.close()