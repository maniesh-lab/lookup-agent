import asyncio
import secrets
import time
import uuid
from contextlib import asynccontextmanager

import structlog
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field
from steadfast import SteadfastError

from app.agent import Agent, create_agent
from app.core import tracing
from app.core.config import settings
from app.core.logging import configure_logging
from app.db import plain_url

log = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs once when the server starts (before `yield`) and once when it stops."""
    configure_logging(settings.log_level)
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    # The checkpointer saves each conversation's messages in Postgres, so a
    # thread_id keeps its memory across requests and server restarts.
    async with AsyncPostgresSaver.from_conn_string(plain_url(settings.database_url)) as saver:
        await saver.setup()
        app.state.agent = create_agent(checkpointer=saver)
        log.info(
            "app_started",
            prompt_version=settings.prompt_version,
            llm_provider=settings.llm_provider,
            tracing=tracing.enabled(),
        )
        yield
    tracing.flush()


app = FastAPI(
    title="lookup-agent",
    description="An agent that looks things up in policy documents, a Postgres database "
    "and a live currency API before it answers.",
    lifespan=lifespan,
)


@app.middleware("http")
async def correlation_id_middleware(request: Request, call_next):
    """Give every request an id, put it on every log line, and return it in a header."""
    correlation_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(correlation_id=correlation_id)
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        log.exception("request_failed", method=request.method, path=request.url.path)
        raise
    response.headers["X-Request-ID"] = correlation_id
    log.info(
        "request",
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        duration_ms=round((time.perf_counter() - start) * 1000, 1),
    )
    return response


def get_agent(request: Request) -> Agent:
    return request.app.state.agent


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health")
def health():
    return {"status": "ok"}


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000, examples=["What's your return window?"])
    thread_id: str | None = Field(
        default=None, description="Send the thread_id from a previous reply to continue that conversation."
    )


class Source(BaseModel):
    type: str
    name: str
    section: str | None = None
    query: str | None = None
    detail: str | None = None


class ChatResponse(BaseModel):
    answer: str
    thread_id: str
    sources: list[Source]
    tools_used: list[str]
    steps: int
    prompt_version: str


@app.post("/chat", response_model=ChatResponse, response_model_exclude_none=True)
async def chat(body: ChatRequest, agent: Agent = Depends(get_agent)):
    correlation_id = structlog.contextvars.get_contextvars().get("correlation_id")
    try:
        result = await agent.ask(body.message, thread_id=body.thread_id, correlation_id=correlation_id)
    except SteadfastError as exc:
        # Steadfast already retried (and tried the fallback provider) before giving up.
        log.error("llm_unavailable", error=str(exc))
        raise HTTPException(
            status_code=503, detail="The language model is unavailable right now. Please try again shortly."
        ) from exc
    return ChatResponse(
        answer=result.answer,
        thread_id=result.thread_id,
        sources=[Source(**s) for s in result.sources],
        tools_used=result.tools_used,
        steps=result.steps,
        prompt_version=result.prompt_version,
    )


def check_ingest_token(x_ingest_token: str | None = Header(default=None)) -> None:
    if settings.ingest_token and not (
        x_ingest_token and secrets.compare_digest(x_ingest_token, settings.ingest_token)
    ):
        raise HTTPException(status_code=401, detail="Missing or wrong X-Ingest-Token header.")


class IngestResponse(BaseModel):
    added: list[str]
    updated: list[str]
    unchanged: list[str]
    removed: list[str]
    chunks_written: int


@app.post("/ingest", response_model=IngestResponse, dependencies=[Depends(check_ingest_token)])
async def ingest():
    """Re-read data/policies and update the search index (unchanged files are skipped)."""
    from app.retrieval.ingest import ingest_folder

    stats = await asyncio.to_thread(ingest_folder)
    return IngestResponse(**stats.as_dict())
