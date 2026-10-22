from fastapi import FastAPI, UploadFile, File
import os
import shutil

from services.document_service import index_document, remove_document
from rag.qdrant_store import list_documents
from pydantic import BaseModel
from services.chat_service import answer_question
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.concurrency import run_in_threadpool
import json
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    document_id: str
    question: str

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

    # Embedding a large PDF takes a while; keep the event loop free.
    document_id = await run_in_threadpool(
        index_document,
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

# @app.post("/chat")
# def chat(request: ChatRequest):

#     result = answer_question(
#         question=request.question,
#         document_id=request.document_id
#     )

#     return result


@app.post("/chat")
async def chat(request: ChatRequest):

    async def generate():

        async for chunk in answer_question(
            question=request.question,
            document_id=request.document_id
        ):

            if chunk["type"] == "text":
                yield f"data: {json.dumps({'text': chunk['text']})}\n\n"

            elif chunk["type"] == "sources":
                yield f"data: {json.dumps({'sources': chunk['sources']})}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.delete("/documents/{document_id}")
def delete_document(document_id: str):

    return remove_document(document_id)