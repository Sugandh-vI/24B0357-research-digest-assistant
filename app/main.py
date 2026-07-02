"""
main.py — FastAPI application entry point.
Routes:
  GET  /           → serves index.html
  POST /api/digest → runs LangGraph digest flow
  POST /api/ask    → RAG follow-up Q&A
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (two levels up from this file)
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    load_dotenv(dotenv_path=_env_path)
else:
    load_dotenv()  # fall back to any .env in CWD

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, field_validator

from app.agents.supervisor import run_digest
from app.rag.vector_store import embed_papers, clear_collection
from app.rag.qa_chain import answer_question

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="Research Paper Digest Assistant")

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# ---------------------------------------------------------------------------
# Request/Response models
# ---------------------------------------------------------------------------

class DigestRequest(BaseModel):
    topic: str

    @field_validator("topic")
    @classmethod
    def topic_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("topic must not be empty")
        return v


class AskRequest(BaseModel):
    question: str

    @field_validator("question")
    @classmethod
    def question_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("question must not be empty")
        return v


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.post("/api/digest")
async def digest(req: DigestRequest):
    try:
        # Clear previous session's vector store
        clear_collection()

        # Run the full LangGraph supervisor flow
        papers = run_digest(req.topic)

        if not papers:
            return JSONResponse(
                status_code=200,
                content={
                    "status": "no_results",
                    "message": f"No papers found for topic: '{req.topic}'. Try a different search term.",
                    "papers": [],
                },
            )

        # Persist papers into vector store for follow-up Q&A
        embed_papers(papers)

        return JSONResponse(
            content={"status": "ok", "topic": req.topic, "papers": papers}
        )

    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Digest generation failed: {str(exc)}",
        )


@app.post("/api/ask")
async def ask(req: AskRequest):
    try:
        result = answer_question(req.question)
        return JSONResponse(content={"status": "ok", **result})
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Q&A failed: {str(exc)}",
        )
