"""
main.py — FastAPI application entry point
Endpoints:
  POST   /upload            — ingest one or more invoice files
  GET    /invoices          — list all ingested invoices
  DELETE /invoices/{doc_id} — remove an invoice from the store
  POST   /chat              — streaming SSE chat with RAG context
  GET    /health            — health check
"""

import os
import uuid
import logging
import asyncio
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from ingest import ingest_invoice, delete_invoice, list_invoices
from retrieval import retrieve_context
from llm import chat_stream

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Directories ───────────────────────────────────────────────────────────────
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "../uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_EXTENSIONS = {
    ".pdf", ".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"
}

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Invoice RAG Chatbot API",
    description="RAG-powered invoice Q&A using Gemini (v1) or custom SLM (v2)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────
class ChatMessage(BaseModel):
    role: str   # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    filter_doc_id: Optional[str] = None  # restrict context to one invoice


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}


@app.post("/upload")
async def upload_invoices(files: list[UploadFile] = File(...)):
    """
    Accept one or more invoice files, save them, and ingest into vector store.
    Supported: PDF, JPG, JPEG, PNG, TIFF, BMP, WEBP
    """
    results = []
    errors = []

    for file in files:
        suffix = Path(file.filename or "").suffix.lower()

        if suffix not in SUPPORTED_EXTENSIONS:
            errors.append(
                {"filename": file.filename, "error": f"Unsupported type: {suffix}"}
            )
            continue

        doc_id = str(uuid.uuid4())
        safe_name = f"{doc_id}{suffix}"
        save_path = UPLOAD_DIR / safe_name

        try:
            # Save file
            content = await file.read()
            save_path.write_bytes(content)
            logger.info(f"Saved upload: {save_path}")

            # Run ingestion in a thread (CPU/IO-bound work)
            loop = asyncio.get_event_loop()
            meta = await loop.run_in_executor(
                None,
                ingest_invoice,
                save_path,
                file.filename,
                doc_id,
            )
            results.append(meta)

        except Exception as e:
            logger.exception(f"Ingestion failed for {file.filename}")
            # Clean up saved file on error
            if save_path.exists():
                save_path.unlink()
            errors.append({"filename": file.filename, "error": str(e)})

    if not results and errors:
        raise HTTPException(status_code=422, detail=errors)

    return {"ingested": results, "errors": errors}


@app.get("/invoices")
async def get_invoices():
    """List all ingested invoice documents."""
    try:
        invoices = list_invoices()
        return {"invoices": invoices, "count": len(invoices)}
    except Exception as e:
        logger.exception("Failed to list invoices")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/invoices/{doc_id}")
async def remove_invoice(doc_id: str):
    """Remove an invoice and all its chunks from the vector store."""
    try:
        loop = asyncio.get_event_loop()
        deleted = await loop.run_in_executor(None, delete_invoice, doc_id)

        # Also remove the uploaded file
        for f in UPLOAD_DIR.iterdir():
            if f.stem == doc_id:
                f.unlink()
                break

        return {"doc_id": doc_id, "chunks_deleted": deleted}
    except Exception as e:
        logger.exception(f"Failed to delete invoice {doc_id}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
async def chat(request: ChatRequest):
    """
    Stream a chat response using RAG context from the vector store.
    Returns Server-Sent Events (SSE) stream.
    """
    messages = [m.model_dump() for m in request.messages]

    if not messages or messages[-1]["role"] != "user":
        raise HTTPException(
            status_code=400, detail="Last message must be from the user."
        )

    # Retrieve relevant context
    query = messages[-1]["content"]
    try:
        loop = asyncio.get_event_loop()
        context = await loop.run_in_executor(
            None,
            retrieve_context,
            query,
            5,
            request.filter_doc_id,
        )
    except Exception as e:
        logger.warning(f"Context retrieval failed: {e}")
        context = ""

    # Stream response as SSE
    async def event_generator():
        try:
            async for token in chat_stream(messages, context):
                # SSE format: data: <token>\n\n
                escaped = token.replace("\n", "\\n")
                yield f"data: {escaped}\n\n"
        except Exception as e:
            logger.exception("Streaming error")
            yield f"data: [ERROR] {str(e)}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
