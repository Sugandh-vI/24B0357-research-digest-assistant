"""
qa_chain.py — RAG Q&A chain using Chroma retrieval + Groq LLM.
Retrieves relevant paper chunks, sends them as context to the LLM,
and returns an answer with citations.
"""
import os
import time
import random
from typing import Dict, Any

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from app.rag.vector_store import query_similar

_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

SYSTEM_PROMPT = (
    "You are a knowledgeable research assistant. The user has just reviewed a set of "
    "research papers. Answer their follow-up question using ONLY the provided context "
    "excerpts below. Always cite the paper titles you drew from (e.g., '[Title: ...]'). "
    "If the context doesn't contain enough information to answer confidently, say so "
    "honestly rather than hallucinating."
)


def _build_llm() -> ChatGroq:
    return ChatGroq(model=_MODEL, temperature=0.3)


def answer_question(question: str, top_k: int = 4, max_retries: int = 5) -> Dict[str, Any]:
    """
    Runs RAG retrieval then asks the Groq LLM grounded in retrieved context.
    Returns a dict with 'answer' and 'citations' (list of paper titles/links).
    """
    chunks = query_similar(question, top_k=top_k)

    if not chunks:
        return {
            "answer": "No papers have been loaded yet. Please generate a digest first.",
            "citations": [],
        }

    # Build context string
    context_parts = []
    citations = []
    seen_titles = set()

    for chunk in chunks:
        meta = chunk.get("metadata", {})
        title = meta.get("title", "Unknown Paper")
        link = meta.get("link", "")
        context_parts.append(f"--- Paper: {title} ---\n{chunk['document']}\n")
        if title not in seen_titles:
            seen_titles.add(title)
            citations.append({"title": title, "link": link})

    context_text = "\n".join(context_parts)
    user_msg = f"Context:\n{context_text}\n\nQuestion: {question}"

    llm = _build_llm()

    for attempt in range(max_retries):
        try:
            response = llm.invoke(
                [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=user_msg),
                ]
            )
            return {"answer": response.content.strip(), "citations": citations}
        except Exception as exc:
            error_str = str(exc).lower()
            if "429" in error_str or "rate_limit" in error_str or "too many" in error_str:
                if attempt < max_retries - 1:
                    wait = (2 ** attempt) + random.uniform(0, 1)
                    time.sleep(wait)
                    continue
            raise

    return {
        "answer": "Could not retrieve an answer due to rate-limit errors. Please try again shortly.",
        "citations": citations,
    }
