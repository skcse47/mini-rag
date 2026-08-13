from fastapi import FastAPI, UploadFile, File
import os
import shutil

from services.document_service import index_document
from rag.qdrant_store import list_documents


app = FastAPI()


UPLOAD_DIR = "data/uploads"

os.makedirs(
    UPLOAD_DIR,
    exist_ok=True
)


@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...)
):

    file_path = os.path.join(
        UPLOAD_DIR,
        file.filename
    )

    with open(file_path, "wb") as buffer:

        shutil.copyfileobj(
            file.file,
            buffer
        )

    document_id = index_document(
        file_path,
        file.filename
    )

    return {
        "document_id": document_id,
        "filename": file.filename,
        "status": "indexed"
    }


@app.get("/documents")
def get_documents():
    return {
        "documents": list_documents()
    }