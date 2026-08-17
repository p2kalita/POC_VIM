# 🧾 SPAR Invoice RAG Chatbot

An AI-powered invoice Q&A system built by **SPAR Infosys LLC**, using **Retrieval-Augmented Generation (RAG)** with a clean, modern chat frontend powered by an in-house vLLM endpoint.

---

## Features

| Feature | Detail |
|---|---|
| **Supported formats** | PDF (native + scanned), JPG, JPEG, PNG, TIFF, BMP, WEBP |
| **Primary AI backend** | vLLM — `Qwen/Qwen2.5-3B-Instruct` served via Colab + ngrok |
| **Fallback AI backend** | Google Gemini (when `LLM_BASE_URL` is not set) |
| **Vector store** | FAISS — local, persistent (no C++ build tools needed) |
| **Embeddings** | Gemini `gemini-embedding-001` |
| **Frontend** | Clean white theme, animated orb, SSE streaming, file upload |
| **Privacy** | Your data never leaves your infrastructure |
| **Launcher** | One-click `start.bat` for Windows |

---

## Quick Start (Windows)

### 1. Prerequisite
- **Python 3.10+** — [python.org](https://www.python.org/downloads/)
- **Gemini API Key** — [Get one free at Google AI Studio](https://aistudio.google.com/app/apikey) *(used for embeddings & OCR)*
- *(Optional)* **Poppler** for scanned PDFs — see below

### 2. Launch

```bat
double-click start.bat
```

On first run it will:
1. Create `backend/.env` from the template and open it in Notepad
2. Ask you to paste your `GEMINI_API_KEY`
3. Create a Python virtual environment
4. Install all dependencies
5. Start the API server at `http://127.0.0.1:8000`
6. Open the frontend in your browser

### 3. Use It

1. **Upload** invoices using the 📎 attach button or drag-and-drop
2. Wait for the green toast: *"Ingested X invoice(s)"*
3. **Ask questions** in the chat — answers stream back word-by-word
4. Use **Writing Styles** dropdown to change tone (Formal / Casual / Creative)
5. Enable **Citation** toggle to request source references in answers

---

## Directory Structure

```
POC_VIM/
├── backend/
│   ├── main.py          # FastAPI app (endpoints + SSE streaming)
│   ├── ingest.py        # Invoice parsing + FAISS embedding
│   ├── retrieval.py     # Semantic search against FAISS index
│   ├── llm.py           # LLM abstraction (vLLM / Gemini fallback)
│   ├── requirements.txt
│   ├── .env.example
│   └── .env             # ← your secrets (git-ignored)
├── frontend/
│   ├── index.html       # UI shell
│   ├── style.css        # Design system (white theme, purple accent)
│   ├── app.js           # Chat logic, SSE streaming, file upload
│   └── logo.png         # SPAR brand logo
├── vllm/
│   ├── test_openai.py   # Test vLLM endpoint via OpenAI SDK
│   ├── test_simple.py   # Test vLLM endpoint via raw HTTP
│   └── test_app.py      # Example LLMClient wrapper
├── chroma_db/           # auto-created: FAISS vector store
├── uploads/             # auto-created: raw invoice files
├── start.bat            # Windows launcher
└── README.md
```

---

## vLLM Endpoint (In-house AI — Colab + ngrok)

The chatbot is configured to use **`Qwen/Qwen2.5-3B-Instruct`** served by vLLM in Google Colab and exposed via an ngrok public URL.

### How it works

```
[Browser] → [FastAPI backend] → [vLLM on Colab via ngrok] → [Qwen2.5-3B]
```

### Colab Setup

Run this in your Colab notebook to start the vLLM server:

```python
# Install vLLM and ngrok
!pip install vllm pyngrok -q

# Start vLLM server
import subprocess, threading
def run_vllm():
    subprocess.run([
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", "Qwen/Qwen2.5-3B-Instruct",
        "--host", "0.0.0.0",
        "--port", "8000"
    ])
threading.Thread(target=run_vllm, daemon=True).start()

# Expose via ngrok
from pyngrok import ngrok
import time; time.sleep(5)  # wait for server to start
tunnel = ngrok.connect(8000)
print("LLM_BASE_URL =", tunnel.public_url + "/v1")
```

### Update the URL after each Colab restart

Each Colab session generates a new ngrok URL. Update **one line** in `backend/.env`:

```env
LLM_BASE_URL=https://<your-new-url>.ngrok-free.dev/v1
```

Then restart the FastAPI backend. The frontend's **Settings** modal shows a live 🟢 **online** / 🔴 **offline** status badge for the vLLM endpoint.

### Test the endpoint independently

```bash
cd vllm/
python test_openai.py    # chat completions via OpenAI SDK
python test_simple.py    # raw HTTP completions
python test_app.py       # full LLMClient example
```

---

## Optional: Scanned PDF Support (Poppler)

For **scanned PDFs** (images embedded inside PDF), install Poppler:

1. Download from [poppler-windows](https://github.com/oschwartz10612/poppler-windows/releases)
2. Extract and add the `bin/` folder to your Windows `PATH`
3. Restart the terminal / `start.bat`

Text-based PDFs work without Poppler. Image invoices (JPG, PNG, etc.) always use Gemini Vision OCR.

---

## Environment Variables (`backend/.env`)

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | *(required)* | Google Gemini API key — used for embeddings & OCR |
| `GEMINI_CHAT_MODEL` | `gemini-2.0-flash` | Gemini model (fallback only) |
| `GEMINI_EMBEDDING_MODEL` | `models/gemini-embedding-001` | Embedding model |
| `GEMINI_VISION_MODEL` | `gemini-2.0-flash` | Vision/OCR model |
| `LLM_BASE_URL` | *(set to activate vLLM)* | vLLM ngrok URL, must end with `/v1` |
| `LLM_API_KEY` | `dummy-key` | API key for vLLM (any non-empty string) |
| `LLM_CHAT_MODEL` | `Qwen/Qwen2.5-3B-Instruct` | Model name served by vLLM |
| `VLLM_MAX_TOKENS` | `1024` | Max tokens per response |
| `VLLM_TEMPERATURE` | `0.2` | Generation temperature |
| `UPLOAD_DIR` | `../uploads` | Directory for uploaded invoice files |
| `CHROMA_PERSIST_DIR` | `../chroma_db` | FAISS index storage directory |

---

## API Reference

Interactive docs: **http://127.0.0.1:8000/docs** (Swagger UI)

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/upload` | Upload one or more invoice files |
| `GET` | `/invoices` | List all ingested invoices |
| `DELETE` | `/invoices/{doc_id}` | Remove an invoice from the vector store |
| `POST` | `/chat` | Streaming SSE chat with RAG context |
| `GET` | `/health` | Health check + active LLM info |
| `GET` | `/llm-info` | Active LLM backend details (backend/model/status) |

---

## Troubleshooting

| Issue | Solution |
|---|---|
| All buttons unresponsive | Open browser console (F12) — check for JS errors on load |
| `GEMINI_API_KEY` invalid | Check `backend/.env` — no quotes needed around the key |
| Upload fails with 422 | File type not supported, or Gemini Vision quota exceeded |
| Chat gets stuck / no response | Check vLLM badge in Settings — ngrok URL may have changed |
| Words run together in response | Ensure you are on the latest `app.js` (SSE space-stripping fix) |
| Chat returns empty context | Upload invoices first; check `chroma_db/` folder exists |
| Scanned PDF not working | Install Poppler and add to PATH (see above) |
| vLLM badge shows 🔴 offline | Restart Colab notebook and update `LLM_BASE_URL` in `.env` |
| Port 8000 in use | Edit `start.bat` and `.env`: change `PORT` to `8001` |
| `BlobPart` AttributeError | Already fixed in `ingest.py` — use inline dict for Gemini Vision |
