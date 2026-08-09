import utils.clients
import utils.http_session
import utils.transport
import utils.session
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from sentencemodel import (
    load_pdf,
    semantic_chunk,
    upload_to_qdrant,
    search_qdrant,
    generate_answer,
    generate_answer_with_context,
)

app = FastAPI(
    title="Financial RAG API",
    version="1.0.0"
)


# -----------------------------
# Request Models
# -----------------------------

class UploadRequest(BaseModel):
    pdf_path: str


class ConversationMessage(BaseModel):
    role: str
    content: str


class QuestionRequest(BaseModel):
    question: str
    conversational_history: Optional[List[ConversationMessage]] = []


# -----------------------------
# Upload PDF
# -----------------------------

@app.post("aethal/upload")
def upload_pdf(request: UploadRequest):
    try:

        texts = load_pdf(request.pdf_path)

        nodes = semantic_chunk(texts)

        point_ids = upload_to_qdrant(nodes)

        return {
            "status": "success",
            "message": "PDF uploaded successfully.",
            "chunks_uploaded": len(point_ids)
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# -----------------------------
# RAG Question Answering
# -----------------------------

@app.post("/ask-aethal-finance")
def ask_rag(request: QuestionRequest):
    try:

        results = search_qdrant(request.question)

        answer = generate_answer_with_context(
            request.question,
            results,
            request.conversational_history
        )

        return {
            "status": "success",
            "question": request.question,
            "answer": answer
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# -----------------------------
# Gemini Only
# -----------------------------

@app.post("/ask-gemini")
def ask_gemini(request: QuestionRequest):
    try:

        answer = generate_answer(request.question)

        return {
            "status": "success",
            "question": request.question,
            "answer": answer
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )