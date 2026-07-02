# 🔬 Research Paper Digest Assistant

> A fully agentic, multi-agent system that searches arXiv, summarises research papers in plain English, peer-reviews them with a skeptical AI critic, and lets you ask follow-up questions via a RAG-powered chat interface — all running locally in your browser.

---

## Problem it Solves

Keeping up with research is overwhelming. A single arXiv search can return hundreds of papers filled with dense notation and field-specific jargon. This assistant solves that by automatically:

1. **Searching** arXiv for the top 5–8 papers matching your topic
2. **Summarising** each paper in plain English (3-4 sentences)
3. **Critiquing** each paper like a peer reviewer — rating relevance, surfacing limitations, and identifying who would benefit from reading it
4. **Ranking** results by relevance score so the most useful papers surface first
5. **Answering follow-up questions** grounded in the actual paper content via Retrieval-Augmented Generation (RAG)

---

## Architecture — Supervisor Multi-Agent Pattern (LangGraph)

The system is built as a LangGraph `StateGraph` using the **Supervisor** pattern, not a fixed linear pipeline. The Supervisor node makes real routing decisions using `add_conditional_edges`:

```
                    ┌─────────────────────────────────┐
                    │          SUPERVISOR NODE         │
                    │  (Routes based on current state) │
                    └────┬──────────┬─────────┬────────┘
                         │          │         │
               ┌─────────▼──┐  ┌────▼────┐  ┌▼────────────┐
               │ SearchAgent │  │Summarize│  │CritiqueAgent│
               │  (arXiv)   │  │  Agent  │  │ (reviewer)  │
               └─────────┬──┘  └────┬────┘  └┬────────────┘
                         │          │         │
                         └──────────▼─────────┘
                                    │
                           back to Supervisor
                                    │
                              (when done)
                                    │
                                   END
                           (ranked digest output)
```

**Routing logic (conditional edges):**

| State condition | Route |
|---|---|
| No papers fetched yet | → `search` node |
| Paper without summary found | → `summarize` node |
| Summarised paper without critique | → `critique` node |
| All papers processed | → `END` (compile digest) |

The Supervisor also applies **dynamic skipping**: if CritiqueAgent scores a paper `relevance_score ≤ 1`, it is flagged as low-relevance and excluded from the main ranking. If fewer than 2 papers are returned by Search, the system gracefully skips critique and exits cleanly.

---

## Why Multiple Agents? (Not Just One LLM Call)

| Concern | Single LLM | Multi-Agent |
|---|---|---|
| **Context window** | All papers + all tasks in one huge prompt — token overflow risk | Each agent gets focused, relevant context only |
| **Specialisation** | Generic prompt diluted across tasks | Summariser uses a "science communicator" persona; CritiqueAgent uses a "skeptical reviewer" persona — different temperatures & system prompts |
| **Routing logic** | No dynamic decisions — always processes everything | Supervisor can skip irrelevant papers, retry on poor search results |
| **Parallelism** | Sequential | Each agent is independently swappable and testable |
| **Debuggability** | One black box | Clear node-level execution trace via LangGraph |

In short: three distinct agents with distinct personas, temperatures, and responsibilities produce noticeably better output than one mega-prompt trying to do everything.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend API | FastAPI + Uvicorn |
| Frontend | Jinja2 templates + Vanilla JS + CSS |
| Orchestration | LangGraph `StateGraph` (Supervisor pattern) |
| LLM | Groq API — `llama-3.3-70b-versatile` (configurable) |
| LLM integration | `langchain-groq` |
| arXiv search | `arxiv` Python package |
| Vector store | Chroma (persisted locally) |
| Embeddings | `sentence-transformers` — `all-MiniLM-L6-v2` (local, free) |
| Config | `python-dotenv` |

---

## Setup Instructions

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd research-digest-assistant
```

### 2. Create and activate a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> **Note:** The first run will download the `all-MiniLM-L6-v2` sentence-transformers model (~90 MB). This happens automatically and only once.

### 4. Set up environment variables
```bash
cp .env.example .env
# Edit .env and add your Groq API key:
# GROQ_API_KEY=your_key_here
# GROQ_MODEL=llama-3.3-70b-versatile   # optional, this is the default
```

Get a free Groq API key at: https://console.groq.com

### 5. Run the application
```bash
uvicorn app.main:app --reload
```

Open your browser at **http://localhost:8000**

---

## Usage Walkthrough

### Step 1 — Enter a research topic
Type a topic like:
```
retrieval augmented generation for code generation
```
Click **Generate Digest**.

### Step 2 — Watch the agents work
The progress indicator shows:
```
🔍 Search → 📝 Summarize → 🔬 Critique → 📊 Rank
```

The Supervisor routes through each agent for every paper (~30-90 seconds depending on number of papers and Groq rate limits).

### Step 3 — Read the ranked digest
Papers appear ranked by relevance score (5 = highly relevant, 1 = tangential). Each card shows:
- **Plain-English Summary** — what the paper does in 3-4 sentences
- **Peer Review Critique** — relevance score, limitations, target reader

Example output:
```
[5/5] RAG-based Code Generation with LLM Agents
      Chen et al. · 2024-03-12
      Summary: This paper proposes a retrieval-augmented approach to...
      Limitations: Evaluation is limited to Python; no multi-file...
      Target Reader: ML engineers building code-generation tools...

[4/5] Benchmarking LLMs for Code Completion Tasks
      ...
```

### Step 4 — Ask follow-up questions
Once the digest loads, the chat box activates. Ask things like:
- *"Which paper is most practical for production use?"*
- *"Which papers evaluate on real-world codebases?"*
- *"What are the common limitations across all papers?"*

The RAG system retrieves relevant paper chunks from Chroma and answers with citations.

---

## Project Structure

```
research-digest-assistant/
├── app/
│   ├── main.py                  # FastAPI app, routes
│   ├── agents/
│   │   ├── search_agent.py      # arXiv search
│   │   ├── summarizer_agent.py  # Groq LLM summarization
│   │   ├── critique_agent.py    # Groq LLM peer review
│   │   └── supervisor.py        # LangGraph StateGraph
│   ├── rag/
│   │   ├── vector_store.py      # Chroma setup + embed/query
│   │   └── qa_chain.py          # RAG Q&A with citations
│   ├── templates/
│   │   └── index.html           # Jinja2 frontend template
│   └── static/
│       ├── style.css            # Dark glassmorphism design
│       └── app.js               # Vanilla JS frontend logic
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Known Limitations & Future Improvements

### Current limitations
- **Session state is in-memory** — refreshing the page clears the Chroma collection. Digest results are not persisted between server restarts.
- **Groq free-tier rate limits** — processing 7 papers (14 LLM calls: 7 summaries + 7 critiques) can hit the free-tier rate limit. The code implements exponential backoff, but very large batches may still be slow.
- **No full-text access** — only arXiv abstracts are processed, not full PDFs. This limits depth of summaries and critique.
- **Single user / no auth** — clearing the vector store on each new digest request affects all concurrent users.

### Future improvements
- [ ] Add async processing (run summarize/critique in parallel with `asyncio.gather`)
- [ ] Fetch and chunk full PDFs from arXiv for richer RAG context
- [ ] Add persistent user sessions with a database (SQLite or PostgreSQL)
- [ ] Support additional sources: Semantic Scholar, Google Scholar
- [ ] Add a "compare papers" mode that shows a side-by-side table
- [ ] Stream LLM responses to the frontend using SSE for faster perceived performance
- [ ] Add LangSmith tracing for observability of the LangGraph execution
- [ ] Containerise with Docker for easy deployment
