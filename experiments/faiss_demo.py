import json
import numpy as np
import faiss

with open("storage/vector_store.json", encoding="utf-8") as f:
    records = json.load(f)

vectors = np.array(
    [record["embedding"] for record in records],
    dtype="float32"
)

dimension = vectors.shape[1]

index = faiss.IndexFlatIP(dimension)

faiss.normalize_L2(vectors)

index.add(vectors)

faiss.write_index(index, "storage/faiss.index")

print("FAISS index created.")