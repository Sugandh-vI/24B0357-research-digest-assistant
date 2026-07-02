"""
critique_agent.py — CritiqueAgent
Reviews a paper with a skeptical-reviewer persona.
Returns: relevance_score (1-5), limitations, target_reader.
Includes exponential backoff for Groq 429 rate-limit errors.
"""
import os
import time
import random
import json
import re
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage


_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

SYSTEM_PROMPT = (
    "You are a tough but fair academic peer reviewer. Your job is to evaluate a research "
    "paper given its title, abstract, and a plain-English summary.\n\n"
    "Return a JSON object with exactly these keys:\n"
    '  "relevance_score": integer 1-5 (5 = highly relevant to the given query topic, 1 = tangentially related)\n'
    '  "limitations": a 2-3 sentence list of the key weaknesses or limitations of the paper\n'
    '  "target_reader": one sentence describing who would benefit most from reading this paper\n\n'
    "Respond ONLY with valid JSON. Do not include markdown fences or extra text."
)


def _build_llm() -> ChatGroq:
    return ChatGroq(model=_MODEL, temperature=0.2)


def run_critique(
    title: str,
    abstract: str,
    summary: str,
    topic: str,
    max_retries: int = 5,
) -> dict:
    """
    Critique a paper. Returns a dict with relevance_score, limitations, target_reader.
    Implements exponential backoff on 429 rate-limit errors.
    """
    llm = _build_llm()
    user_msg = (
        f"User's research topic: {topic}\n\n"
        f"Paper Title: {title}\n\n"
        f"Abstract:\n{abstract}\n\n"
        f"Plain-English Summary:\n{summary}\n\n"
        "Please evaluate this paper."
    )

    for attempt in range(max_retries):
        try:
            response = llm.invoke(
                [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=user_msg),
                ]
            )
            raw = response.content.strip()
            # Strip markdown fences if the model adds them despite instructions
            raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("```").strip()
            data = json.loads(raw)
            return {
                "relevance_score": int(data.get("relevance_score", 3)),
                "limitations": str(data.get("limitations", "Not available.")),
                "target_reader": str(data.get("target_reader", "General researchers.")),
            }
        except Exception as exc:
            error_str = str(exc).lower()
            if "429" in error_str or "rate_limit" in error_str or "too many" in error_str:
                if attempt < max_retries - 1:
                    wait = (2 ** attempt) + random.uniform(0, 1)
                    time.sleep(wait)
                    continue
            # If JSON parsing failed, return safe defaults
            if "json" in str(type(exc).__name__).lower() or "json" in error_str:
                return {
                    "relevance_score": 3,
                    "limitations": "Could not parse critique.",
                    "target_reader": "General researchers.",
                }
            raise

    return {
        "relevance_score": 3,
        "limitations": "Critique unavailable due to rate-limit errors.",
        "target_reader": "General researchers.",
    }
