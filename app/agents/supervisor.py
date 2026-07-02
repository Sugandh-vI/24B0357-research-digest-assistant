"""
supervisor.py — LangGraph Supervisor orchestrating all agents.

Graph flow:
  START → supervisor_node
    supervisor_node → search        (initial call)
    search          → supervisor_node
    supervisor_node → summarize     (for each unprocessed paper)
    summarize       → supervisor_node
    supervisor_node → critique      (for each summarized paper, if not flagged low-relevance)
    critique        → supervisor_node
    supervisor_node → END           (when all papers processed)

Conditional edges make this a true Supervisor pattern, not a linear pipeline.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import StateGraph, END

from app.agents.search_agent import run_search
from app.agents.summarizer_agent import run_summarize
from app.agents.critique_agent import run_critique


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

class PaperState(TypedDict, total=False):
    title: str
    authors: List[str]
    abstract: str
    link: str
    published: str
    relevance_flagged_low: bool
    summary: Optional[str]
    critique: Optional[Dict[str, Any]]


class DigestState(TypedDict, total=False):
    topic: str
    papers: List[PaperState]
    current_index: int          # which paper is being processed
    phase: str                  # "search" | "summarize" | "critique" | "done"
    digest: List[Dict[str, Any]]


# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------

def supervisor_node(state: DigestState) -> DigestState:
    """
    Routing logic — decides what to do next based on current state.
    Sets state["phase"] to communicate its decision to add_conditional_edges.
    """
    papers = state.get("papers", [])
    idx = state.get("current_index", 0)

    # First invocation: papers not yet fetched
    if not papers:
        return {**state, "phase": "search", "current_index": 0}

    # Find the next paper that needs a summary
    for i in range(len(papers)):
        p = papers[i]
        if p.get("summary") is None and not p.get("relevance_flagged_low", False):
            return {**state, "phase": "summarize", "current_index": i}

    # Find the next summarised paper that needs a critique
    for i in range(len(papers)):
        p = papers[i]
        if (
            p.get("summary") is not None
            and p.get("critique") is None
            and not p.get("relevance_flagged_low", False)
        ):
            return {**state, "phase": "critique", "current_index": i}

    # All done — compile ranked digest
    processed = [
        p for p in papers
        if p.get("summary") is not None and p.get("critique") is not None
    ]
    # Also include flagged-low papers at the bottom with score=1 so nothing is silently dropped
    flagged = [p for p in papers if p.get("relevance_flagged_low")]

    ranked = sorted(processed, key=lambda x: x["critique"]["relevance_score"], reverse=True)

    digest = []
    for p in ranked + flagged:
        digest.append(
            {
                "title": p["title"],
                "authors": p["authors"],
                "link": p["link"],
                "published": p["published"],
                "summary": p.get("summary", "N/A"),
                "critique": p.get("critique", {}),
                "relevance_score": (p.get("critique") or {}).get("relevance_score", 1),
            }
        )

    return {**state, "phase": "done", "digest": digest}


def search_node(state: DigestState) -> DigestState:
    """Calls SearchAgent and stores results. Re-routes to supervisor if < 2 results."""
    topic = state["topic"]
    papers: List[PaperState] = run_search(topic)  # type: ignore[assignment]

    if len(papers) < 2:
        # Flag all as low-relevance so Supervisor skips critique and goes straight to done
        for p in papers:
            p["relevance_flagged_low"] = True

    return {**state, "papers": papers, "current_index": 0}


def summarize_node(state: DigestState) -> DigestState:
    """Summarises the paper at current_index."""
    papers = [dict(p) for p in state["papers"]]
    idx = state["current_index"]
    p = papers[idx]
    summary = run_summarize(p["title"], p["abstract"])
    p["summary"] = summary
    papers[idx] = p
    return {**state, "papers": papers}


def critique_node(state: DigestState) -> DigestState:
    """Critiques the paper at current_index."""
    papers = [dict(p) for p in state["papers"]]
    idx = state["current_index"]
    p = papers[idx]
    critique = run_critique(
        title=p["title"],
        abstract=p["abstract"],
        summary=p.get("summary", ""),
        topic=state["topic"],
    )
    # Supervisor routing: if relevance_score <= 1, flag as low-relevance
    if critique["relevance_score"] <= 1:
        p["relevance_flagged_low"] = True
    p["critique"] = critique
    papers[idx] = p
    return {**state, "papers": papers}


# ---------------------------------------------------------------------------
# Conditional edge router
# ---------------------------------------------------------------------------

def route(state: DigestState) -> str:
    """Maps state["phase"] to the next node name."""
    phase = state.get("phase", "search")
    return phase  # "search" | "summarize" | "critique" | "done"


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

def build_graph() -> Any:
    builder = StateGraph(DigestState)

    builder.add_node("supervisor", supervisor_node)
    builder.add_node("search", search_node)
    builder.add_node("summarize", summarize_node)
    builder.add_node("critique", critique_node)

    # Entry point is always the supervisor
    builder.set_entry_point("supervisor")

    # Supervisor decides where to go next via conditional edges
    builder.add_conditional_edges(
        "supervisor",
        route,
        {
            "search": "search",
            "summarize": "summarize",
            "critique": "critique",
            "done": END,
        },
    )

    # After each worker node, always return to supervisor for next decision
    builder.add_edge("search", "supervisor")
    builder.add_edge("summarize", "supervisor")
    builder.add_edge("critique", "supervisor")

    return builder.compile()


# Singleton compiled graph
_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_digest(topic: str) -> List[Dict[str, Any]]:
    """
    Public entry point. Runs the full LangGraph Supervisor flow.
    Returns a list of paper dicts (ranked digest).
    """
    graph = get_graph()
    initial_state: DigestState = {
        "topic": topic,
        "papers": [],
        "current_index": 0,
        "phase": "",
        "digest": [],
    }
    final_state = graph.invoke(initial_state)
    return final_state.get("digest", [])
