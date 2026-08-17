# 🧾 Invoice RAG Chatbot

An AI-powered invoice Q&A system using **Retrieval-Augmented Generation (RAG)** with a ChatGPT-style frontend.

---

## Features

| Feature | Detail |
|---|---|
| **Supported formats** | PDF (native + scanned), JPG, JPEG, PNG, TIFF, BMP, WEBP |
| **AI backend** | Google Gemini 2.0 Flash (v1) · Any OpenAI-compatible SLM (v2) |
| **Vector store** | FAISS — local, persistent (no C++ build tools needed) |
| **Embeddings** | Gemini `text-embedding-004` |
| **Frontend** | ChatGPT-style — dark theme, streaming, drag-and-drop |
| **Launcher** | One-click `start.bat` for Windows |

---

## Quick Start (Windows)

### 1. Prerequisites
- **Python 3.10+** — [python.org](https://www.python.org/downloads/)
- **Gemini API Key** — [Get one free at Google AI Studio](https://aistudio.google.com/app/apikey)
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

1. **Upload** invoices using the sidebar (drag-and-drop or click)
2. Wait for the green toast: *"Ingested X invoice(s)"*
3. **Ask questions** in the chat — answers stream back word-by-word
4. Click an invoice in the sidebar to filter answers to that document only

---

## Directory Structure

```
POC_VIM/
├── backend/
│   ├── main.py          # FastAPI app
│   ├── ingest.py        # Invoice parsing + embedding
│   ├── retrieval.py     # ChromaDB semantic search
│   ├── llm.py           # LLM abstraction (Gemini / SLM)
│   ├── requirements.txt
│   ├── .env.example
│   └── .env             # ← your secrets (git-ignored)
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
├── chroma_db/           # auto-created: vector store
├── uploads/             # auto-created: raw invoice files
├── start.bat            # Windows launcher
└── README.md
```

---

## Optional: Scanned PDF Support (Poppler)

For **scanned PDFs** (images inside PDF), install Poppler:

1. Download from [poppler-windows](https://github.com/oschwartz10612/poppler-windows/releases)
2. Extract and add the `bin/` folder to your Windows `PATH`
3. Restart the terminal / `start.bat`

Text-based PDFs work without Poppler.

---

## Version 2 — Custom SLM Endpoint (Colab / Ollama / LM Studio)

Replace Gemini with any OpenAI-compatible model by setting two env vars in `backend/.env`:

```env
LLM_BASE_URL=https://xxxx.ngrok-free.app/v1
LLM_CHAT_MODEL=mistral-7b-instruct
```

### Colab Setup (Google Colab)

Run this cell in your Colab notebook to start an OpenAI-compatible server:

```python
# Install
!pip install llama-cpp-python[server] pyngrok -q

# Start server (replace with your model path)
import subprocess, threading
def run():
    subprocess.run([
        "python", "-m", "llama_cpp.server",
        "--model", "/content/your_model.gguf",
        "--host", "0.0.0.0", "--port", "8080"
    ])
threading.Thread(target=run, daemon=True).start()

# Expose via ngrok
from pyngrok import ngrok
tunnel = ngrok.connect(8080)
print("LLM_BASE_URL =", tunnel.public_url + "/v1")
```

Paste the printed URL into the frontend **Settings** modal (⚙️) or `backend/.env`.

---

## API Reference

The full interactive API docs are available at:
**http://127.0.0.1:8000/docs** (Swagger UI)

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/upload` | Upload one or more invoice files |
| `GET` | `/invoices` | List all ingested invoices |
| `DELETE` | `/invoices/{doc_id}` | Remove an invoice |
| `POST` | `/chat` | Streaming SSE chat with RAG context |
| `GET` | `/health` | Health check |

---

## Troubleshooting

| Issue | Solution |
|---|---|
| `GEMINI_API_KEY` invalid | Check `backend/.env` — no quotes needed |
| Upload fails with 422 | File type not supported, or Gemini Vision quota exceeded |
| Chat returns empty context | Upload invoices first; check ChromaDB `chroma_db/` folder exists |
| Scanned PDF not working | Install Poppler and add to PATH (see above) |
| Port 8000 in use | Edit `start.bat` and `.env`: change PORT to 8001 |
