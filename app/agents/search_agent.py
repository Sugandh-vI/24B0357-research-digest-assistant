"""
search_agent.py — SearchAgent
Fetches top 5-8 arXiv papers matching the user's research topic.
"""
import arxiv
from typing import List, Dict, Any


def run_search(topic: str, max_results: int = 7) -> List[Dict[str, Any]]:
    """
    Search arXiv for papers matching the topic.
    Returns a list of dicts with title, authors, abstract, link, published.
    """
    client = arxiv.Client(num_retries=3, delay_seconds=1)
    search = arxiv.Search(
        query=topic,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    papers = []
    for result in client.results(search):
        papers.append(
            {
                "title": result.title,
                "authors": [str(a) for a in result.authors],
                "abstract": result.summary,
                "link": result.entry_id,
                "published": result.published.strftime("%Y-%m-%d"),
                "relevance_flagged_low": False,  # Supervisor may flip this
            }
        )
    return papers
