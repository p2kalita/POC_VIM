"""
llm.py — LLM abstraction layer
Supports:
  v1: Google Gemini API        (default, when LLM_BASE_URL is NOT set)
  v2: vLLM / any OpenAI-compatible endpoint
      Set LLM_BASE_URL in .env to activate (e.g. Colab + ngrok tunnel).

vLLM-specific notes:
  - Exposed via ngrok as an OpenAI-compatible server (/v1/chat/completions)
  - Model name must match exactly what vLLM is serving (e.g. Qwen/Qwen2.5-3B-Instruct)
  - API key can be any non-empty string ("dummy-key" works)
  - LLM_BASE_URL must end with /v1  (or we append it automatically)
"""

import os
import logging
from typing import AsyncIterator

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
_raw_base_url: str | None = os.getenv("LLM_BASE_URL")

# Normalise: strip trailing slash, ensure /v1 suffix
def _normalise_base_url(url: str | None) -> str | None:
    if not url:
        return None
    url = url.rstrip("/")
    if not url.endswith("/v1"):
        url = url + "/v1"
    return url

LLM_BASE_URL:     str | None = _normalise_base_url(_raw_base_url)
LLM_API_KEY:      str        = os.getenv("LLM_API_KEY",      "dummy-key")
LLM_CHAT_MODEL:   str        = os.getenv("LLM_CHAT_MODEL",   "Qwen/Qwen2.5-3B-Instruct")
VLLM_MAX_TOKENS:  int        = int(os.getenv("VLLM_MAX_TOKENS",  "1024"))
VLLM_TEMPERATURE: float      = float(os.getenv("VLLM_TEMPERATURE", "0.2"))

GEMINI_API_KEY:       str = os.getenv("GEMINI_API_KEY", "")
GEMINI_CHAT_MODEL:    str = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.0-flash")

# Log which backend will be used at import time
if LLM_BASE_URL:
    logger.info(
        f"[LLM] vLLM mode  | base_url={LLM_BASE_URL} | model={LLM_CHAT_MODEL}"
    )
else:
    logger.info(
        f"[LLM] Gemini mode | model={GEMINI_CHAT_MODEL}"
    )

# ── Shared system prompt ──────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an intelligent invoice assistant.
You have access to extracted text from one or more invoices provided as context.
Answer user questions accurately based ONLY on the invoice context provided.
When citing information, mention the invoice filename or ID if available.
If the context does not contain the answer, say so clearly — do not hallucinate.
Format currency amounts, dates, and numbers clearly.
Be concise and professional."""


# ── vLLM / OpenAI-compatible backend ─────────────────────────────────────────
async def vllm_chat_stream(
    messages: list[dict], context: str
) -> AsyncIterator[str]:
    """
    Stream tokens from a vLLM endpoint exposed via ngrok.
    Uses the openai Python SDK pointed at the custom base_url.
    """
    from openai import AsyncOpenAI, APIConnectionError, APIStatusError  # type: ignore

    client = AsyncOpenAI(
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        # Give the Colab/ngrok tunnel extra time to respond
        timeout=60.0,
    )

    # Build system message: inject RAG context if available
    system_content = SYSTEM_PROMPT
    if context:
        system_content += f"\n\n[INVOICE CONTEXT]\n{context}"

    oai_messages = [{"role": "system", "content": system_content}]
    oai_messages.extend(messages)

    logger.info(
        f"[vLLM] Streaming chat | model={LLM_CHAT_MODEL} "
        f"| msgs={len(oai_messages)} | context_chars={len(context)}"
    )

    try:
        stream = await client.chat.completions.create(
            model=LLM_CHAT_MODEL,
            messages=oai_messages,
            stream=True,
            temperature=VLLM_TEMPERATURE,
            max_tokens=VLLM_MAX_TOKENS,
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content

    except APIConnectionError as e:
        logger.error(f"[vLLM] Connection error — is the ngrok tunnel still alive? {e}")
        raise RuntimeError(
            "Cannot reach the vLLM server. "
            "Please check that your Colab notebook is running and "
            "update LLM_BASE_URL in backend/.env with the current ngrok URL."
        ) from e

    except APIStatusError as e:
        logger.error(f"[vLLM] API status error {e.status_code}: {e.message}")
        raise RuntimeError(f"vLLM returned HTTP {e.status_code}: {e.message}") from e


# ── Gemini backend (v1 fallback) ──────────────────────────────────────────────
async def gemini_chat_stream(
    messages: list[dict], context: str
) -> AsyncIterator[str]:
    """Stream tokens from the Google Gemini API."""
    import google.generativeai as genai  # type: ignore

    genai.configure(api_key=GEMINI_API_KEY)

    model = genai.GenerativeModel(
        model_name=GEMINI_CHAT_MODEL,
        system_instruction=SYSTEM_PROMPT,
    )

    # Build Gemini conversation history (all messages except the last user msg)
    history = []
    for msg in messages[:-1]:
        role = "user" if msg["role"] == "user" else "model"
        history.append({"role": role, "parts": [msg["content"]]})

    chat = model.start_chat(history=history)

    last_user_msg = messages[-1]["content"]
    augmented_prompt = (
        f"[INVOICE CONTEXT]\n{context}\n\n[USER QUESTION]\n{last_user_msg}"
        if context
        else last_user_msg
    )

    logger.info(f"[Gemini] Streaming chat | model={GEMINI_CHAT_MODEL}")

    response = await chat.send_message_async(augmented_prompt, stream=True)
    async for chunk in response:
        if chunk.text:
            yield chunk.text


# ── Public interface ──────────────────────────────────────────────────────────
async def chat_stream(
    messages: list[dict], context: str = ""
) -> AsyncIterator[str]:
    """
    Unified streaming chat interface.
    Routes to vLLM (v2) when LLM_BASE_URL is configured,
    otherwise falls back to Gemini (v1).
    """
    if LLM_BASE_URL:
        async for token in vllm_chat_stream(messages, context):
            yield token
    else:
        async for token in gemini_chat_stream(messages, context):
            yield token


# ── Info helper (used by /llm-info endpoint) ──────────────────────────────────
def get_llm_info() -> dict:
    """Return a dict describing the active LLM backend."""
    if LLM_BASE_URL:
        return {
            "backend": "vllm",
            "base_url": LLM_BASE_URL,
            "model": LLM_CHAT_MODEL,
            "max_tokens": VLLM_MAX_TOKENS,
            "temperature": VLLM_TEMPERATURE,
        }
    return {
        "backend": "gemini",
        "model": GEMINI_CHAT_MODEL,
    }
