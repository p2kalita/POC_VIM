"""
ingest.py — Invoice ingestion pipeline
Handles:
  - PDF (native text extraction via pdfplumber)
  - Scanned PDF (pdf2image → Gemini Vision OCR)
  - Images: JPG, JPEG, PNG, TIFF, BMP, WEBP (Gemini Vision OCR)
  - Text chunking with invoice metadata
  - Embedding via Gemini text-embedding-004
  - Storage in FAISS index + JSON metadata (no C++ build tools required)
"""

import os
import io
import json
import uuid
import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pdfplumber
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_VISION_MODEL: str = os.getenv("GEMINI_VISION_MODEL", "gemini-3.5-flash")
GEMINI_EMBEDDING_MODEL: str = os.getenv(
    "GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001"
)

VECTOR_STORE_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", "../chroma_db"))

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}
SUPPORTED_PDF_EXTS = {".pdf"}

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


# ── FAISS store paths ─────────────────────────────────────────────────────────

def _store_dir() -> Path:
    d = VECTOR_STORE_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d

def _index_path()    -> Path: return _store_dir() / "index.faiss"
def _meta_path()     -> Path: return _store_dir() / "metadata.json"


# ── FAISS helpers ─────────────────────────────────────────────────────────────

def _load_store() -> tuple[object, list[dict]]:
    """Load FAISS index + metadata list from disk. Creates empty store if missing."""
    import faiss  # type: ignore

    meta_path  = _meta_path()
    index_path = _index_path()

    if meta_path.exists() and index_path.exists():
        index = faiss.read_index(str(index_path))
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    else:
        index    = None   # will be created on first add
        metadata = []

    return index, metadata


def _save_store(index, metadata: list[dict]) -> None:
    """Persist FAISS index and metadata to disk."""
    import faiss  # type: ignore

    faiss.write_index(index, str(_index_path()))
    with open(_meta_path(), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False)


def _cosine_normalize(vecs: np.ndarray) -> np.ndarray:
    """L2-normalise embedding vectors for cosine similarity via inner product."""
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-10, norms)
    return vecs / norms


# ── Gemini helpers ────────────────────────────────────────────────────────────

def _gemini_vision_ocr(image_bytes: bytes, mime_type: str = "image/jpeg") -> str:
    """Extract structured text from an invoice image using Gemini Vision."""
    import google.generativeai as genai          # type: ignore
    import google.generativeai.types as gtypes   # type: ignore

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(GEMINI_VISION_MODEL)

    prompt = (
        "You are an invoice OCR engine. "
        "Extract ALL text from this invoice image exactly as shown. "
        "Preserve the structure: vendor name, address, invoice number, date, "
        "line items (description, qty, unit price, total), subtotal, taxes, "
        "grand total, payment terms, and any notes. "
        "Output plain text. Do NOT add commentary."
    )

    image_part = gtypes.BlobPart(data=image_bytes, mime_type=mime_type)
    response = model.generate_content([prompt, image_part])
    return response.text or ""


def _gemini_embed(texts: list[str]) -> list[list[float]]:
    """Embed a list of strings using Gemini text-embedding-004."""
    import google.generativeai as genai  # type: ignore

    genai.configure(api_key=GEMINI_API_KEY)
    embeddings = []
    for text in texts:
        result = genai.embed_content(
            model=GEMINI_EMBEDDING_MODEL,
            content=text,
            task_type="RETRIEVAL_DOCUMENT",
        )
        embeddings.append(result["embedding"])
    return embeddings


# ── Text chunking ─────────────────────────────────────────────────────────────

def _chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    if not text.strip():
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end  = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start += chunk_size - overlap
    return chunks


# ── PDF processing ────────────────────────────────────────────────────────────

def _extract_pdf_text(pdf_path: Path) -> dict[int, str]:
    """Extract text per page; fall back to Gemini Vision for scanned pages."""
    page_texts: dict[int, str] = {}

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            native_text = page.extract_text() or ""

            if len(native_text.strip()) > 50:
                page_texts[i + 1] = native_text
            else:
                logger.info(f"Page {i+1}: falling back to Vision OCR")
                try:
                    from pdf2image import convert_from_path  # type: ignore
                    images = convert_from_path(
                        str(pdf_path), first_page=i + 1, last_page=i + 1, dpi=200
                    )
                    if images:
                        buf = io.BytesIO()
                        images[0].save(buf, format="JPEG")
                        ocr_text = _gemini_vision_ocr(buf.getvalue(), "image/jpeg")
                        page_texts[i + 1] = ocr_text
                    else:
                        page_texts[i + 1] = native_text
                except Exception as e:
                    logger.warning(f"Vision OCR failed for page {i+1}: {e}")
                    page_texts[i + 1] = native_text

    return page_texts


# ── Image processing ──────────────────────────────────────────────────────────

def _extract_image_text(image_path: Path) -> str:
    """OCR a single invoice image via Gemini Vision."""
    with Image.open(image_path) as img:
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=90)
        image_bytes = buf.getvalue()
    return _gemini_vision_ocr(image_bytes, "image/jpeg")


# ── Main ingest function ──────────────────────────────────────────────────────

def ingest_invoice(
    file_path: Path,
    original_filename: str,
    doc_id: Optional[str] = None,
) -> dict:
    """
    Ingest an invoice file into the FAISS vector store.
    Returns metadata about the ingested document.
    """
    import faiss  # type: ignore

    doc_id = doc_id or str(uuid.uuid4())
    suffix = file_path.suffix.lower()

    logger.info(f"Ingesting: {original_filename} (id={doc_id})")

    # ── Extract text ──────────────────────────────────────────────────────────
    if suffix in SUPPORTED_PDF_EXTS:
        page_texts = _extract_pdf_text(file_path)
    elif suffix in SUPPORTED_IMAGE_EXTS:
        page_texts = {1: _extract_image_text(file_path)}
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    if not page_texts:
        raise ValueError(f"No text extracted from {original_filename}")

    # ── Chunk ─────────────────────────────────────────────────────────────────
    all_chunks: list[str] = []
    chunk_metas: list[dict] = []

    for page_num, text in page_texts.items():
        for chunk_idx, chunk in enumerate(_chunk_text(text)):
            all_chunks.append(chunk)
            chunk_metas.append({
                "id":          f"{doc_id}_{page_num}_{chunk_idx}",
                "doc_id":      doc_id,
                "filename":    original_filename,
                "page":        page_num,
                "chunk_index": chunk_idx,
                "text":        chunk,
            })

    if not all_chunks:
        raise ValueError(f"No text chunks produced from {original_filename}")

    # ── Embed ─────────────────────────────────────────────────────────────────
    raw_embeddings = _gemini_embed(all_chunks)
    vecs = _cosine_normalize(np.array(raw_embeddings, dtype=np.float32))
    dim  = vecs.shape[1]

    # ── Load or create FAISS index ────────────────────────────────────────────
    index, existing_meta = _load_store()

    if index is None:
        # Inner-product index (cosine sim after L2-normalisation)
        index = faiss.IndexFlatIP(dim)

    index.add(vecs)
    existing_meta.extend(chunk_metas)

    # ── Persist ───────────────────────────────────────────────────────────────
    _save_store(index, existing_meta)

    logger.info(f"Stored {len(all_chunks)} chunks for {original_filename}")

    return {
        "doc_id":   doc_id,
        "filename": original_filename,
        "pages":    len(page_texts),
        "chunks":   len(all_chunks),
    }


def delete_invoice(doc_id: str) -> int:
    """
    Remove all chunks for a document from the FAISS store.
    FAISS doesn't support deletion natively; we rebuild the index without them.
    """
    import faiss  # type: ignore

    _, metadata = _load_store()
    if not metadata:
        return 0

    keep = [m for m in metadata if m.get("doc_id") != doc_id]
    removed = len(metadata) - len(keep)

    if removed == 0:
        return 0

    if not keep:
        # Empty store — just delete the files
        _index_path().unlink(missing_ok=True)
        _meta_path().unlink(missing_ok=True)
        return removed

    # Rebuild index from kept chunks
    raw_embeddings = _gemini_embed([m["text"] for m in keep])
    vecs = _cosine_normalize(np.array(raw_embeddings, dtype=np.float32))
    new_index = faiss.IndexFlatIP(vecs.shape[1])
    new_index.add(vecs)

    _save_store(new_index, keep)
    return removed


def list_invoices() -> list[dict]:
    """Return a deduplicated list of ingested invoices."""
    _, metadata = _load_store()
    seen: dict[str, dict] = {}
    for m in metadata:
        if m.get("doc_id") not in seen:
            seen[m["doc_id"]] = {
                "doc_id":   m["doc_id"],
                "filename": m.get("filename", "unknown"),
            }
    return list(seen.values())
