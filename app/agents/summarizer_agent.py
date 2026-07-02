"""
summarizer_agent.py — SummarizerAgent
Produces a plain-English 3-4 sentence summary of a paper's abstract using Groq.
Includes exponential backoff for Groq rate-limit (429) errors.
"""
import os
import time
import random
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage


_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

SYSTEM_PROMPT = (
    "You are a friendly science communicator who translates dense academic abstracts "
    "into clear, engaging 3-4 sentence summaries for readers who are curious but not "
    "specialists in the subfield. Focus on: what problem is solved, how it's solved, "
    "and what the key result is. Write in plain English. Avoid jargon."
)


def _build_llm() -> ChatGroq:
    return ChatGroq(model=_MODEL, temperature=0.3)


def run_summarize(title: str, abstract: str, max_retries: int = 5) -> str:
    """
    Summarize a paper's abstract. Returns a plain-English summary string.
    Implements exponential backoff on 429 rate-limit errors.
    """
    llm = _build_llm()
    user_msg = f"Title: {title}\n\nAbstract:\n{abstract}\n\nPlease summarize this paper."

    for attempt in range(max_retries):
        try:
            response = llm.invoke(
                [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=user_msg),
                ]
            )
            return response.content.strip()
        except Exception as exc:
            error_str = str(exc).lower()
            if "429" in error_str or "rate_limit" in error_str or "too many" in error_str:
                if attempt < max_retries - 1:
                    wait = (2 ** attempt) + random.uniform(0, 1)
                    time.sleep(wait)
                    continue
            raise
    return "Summary unavailable due to repeated rate-limit errors."
