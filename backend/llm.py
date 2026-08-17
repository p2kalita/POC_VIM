"""
llm.py — LLM abstraction layer
Supports:
  v1: Google Gemini API  (default, when LLM_BASE_URL is not set)
  v2: Any OpenAI-compatible endpoint (Colab SLM via ngrok, Ollama, etc.)
      Set LLM_BASE_URL in .env to activate.
"""

import os
import json
from typing import AsyncIterator

from dotenv import load_dotenv

load_dotenv()

# ── Determine mode ────────────────────────────────────────────────────────────
LLM_BASE_URL: str | None = os.getenv("LLM_BASE_URL")
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_CHAT_MODEL: str = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.0-flash")
LLM_API_KEY: str = os.getenv("LLM_API_KEY", "not-needed")
LLM_CHAT_MODEL: str = os.getenv("LLM_CHAT_MODEL", "gpt-3.5-turbo")

SYSTEM_PROMPT = """You are an intelligent invoice assistant. 
You have access to extracted text from one or more invoices provided as context.
Answer user questions accurately based ONLY on the invoice context provided.
When citing information, mention the invoice filename or ID if available.
If the context does not contain the answer, say so clearly — do not hallucinate.
Format currency amounts, dates, and numbers clearly.
Be concise and professional."""


# ── Gemini backend (v1) ───────────────────────────────────────────────────────
def _build_gemini_client():
    import google.generativeai as genai  # type: ignore
    genai.configure(api_key=GEMINI_API_KEY)
    return genai


async def gemini_chat_stream(
    messages: list[dict], context: str
) -> AsyncIterator[str]:
    """Stream tokens from Gemini API."""
    import google.generativeai as genai  # type: ignore
    genai.configure(api_key=GEMINI_API_KEY)

    model = genai.GenerativeModel(
        model_name=GEMINI_CHAT_MODEL,
        system_instruction=SYSTEM_PROMPT,
    )

    # Build conversation history for Gemini format
    history = []
    for msg in messages[:-1]:  # all except the latest user message
        role = "user" if msg["role"] == "user" else "model"
        history.append({"role": role, "parts": [msg["content"]]})

    chat = model.start_chat(history=history)

    # Inject retrieved context into the latest user message
    last_user_msg = messages[-1]["content"]
    augmented_prompt = (
        f"[INVOICE CONTEXT]\n{context}\n\n[USER QUESTION]\n{last_user_msg}"
        if context
        else last_user_msg
    )

    response = await chat.send_message_async(augmented_prompt, stream=True)
    async for chunk in response:
        if chunk.text:
            yield chunk.text


# ── OpenAI-compatible backend (v2) ───────────────────────────────────────────
async def openai_chat_stream(
    messages: list[dict], context: str
) -> AsyncIterator[str]:
    """Stream tokens from an OpenAI-compatible endpoint."""
    from openai import AsyncOpenAI  # type: ignore

    client = AsyncOpenAI(
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
    )

    # Inject context into the system message
    system_with_context = SYSTEM_PROMPT
    if context:
        system_with_context += f"\n\n[INVOICE CONTEXT]\n{context}"

    oai_messages = [{"role": "system", "content": system_with_context}]
    oai_messages.extend(messages)

    stream = await client.chat.completions.create(
        model=LLM_CHAT_MODEL,
        messages=oai_messages,
        stream=True,
        temperature=0.2,
        max_tokens=2048,
    )

    async for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            yield delta.content


# ── Public interface ──────────────────────────────────────────────────────────
async def chat_stream(
    messages: list[dict], context: str = ""
) -> AsyncIterator[str]:
    """
    Unified streaming chat interface.
    Automatically selects Gemini (v1) or OpenAI-compatible (v2)
    based on whether LLM_BASE_URL is configured.
    """
    if LLM_BASE_URL:
        async for token in openai_chat_stream(messages, context):
            yield token
    else:
        async for token in gemini_chat_stream(messages, context):
            yield token
