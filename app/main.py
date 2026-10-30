import asyncio
import contextlib
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field
from steadfast import SteadfastError

from app.agent import Agent, create_agent
from app.core import tracing
from app.core.config import settings
from app.core.logging import configure_logging
from app.db import plain_url

log = structlog.get_logger(__name__)

CHAT_PAGE = Path(__file__).parent / "static" / "index.html"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs once when the server starts (before `yield`) and once when it stops."""
    configure_logging(settings.log_level)
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from psycopg.rows import dict_row
    from psycopg_pool import AsyncConnectionPool

    # The checkpointer saves each conversation's messages in Postgres, so a
    # thread_id keeps its memory across requests and server restarts. A pool
    # that checks each connection before use means a Postgres restart costs
    # one reconnect instead of breaking every later request.
    pool = AsyncConnectionPool(
        plain_url(settings.database_url),
        min_size=1,
        max_size=5,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        check=AsyncConnectionPool.check_connection,
        open=False,
    )
    async with pool:
        saver = AsyncPostgresSaver(pool)
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


class CorrelationIdMiddleware:
    """Give every request an id, put it on every log line, and return it in a header.

    Written as plain ASGI (not @app.middleware("http")) on purpose: that
    shortcut hides client disconnects from the endpoint, and /chat needs to
    see them to cancel work when the user presses Stop.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers") or [])
        correlation_id = headers.get(b"x-request-id", b"").decode() or str(uuid.uuid4())
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(correlation_id=correlation_id)
        start = time.perf_counter()
        status = 500

        async def send_with_id(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                message = {
                    **message,
                    "headers": [*message.get("headers", []), (b"x-request-id", correlation_id.encode())],
                }
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        except Exception:
            log.exception("request_failed", method=scope["method"], path=scope["path"])
            raise
        log.info(
            "request",
            method=scope["method"],
            path=scope["path"],
            status=status,
            duration_ms=round((time.perf_counter() - start) * 1000, 1),
        )


app.add_middleware(CorrelationIdMiddleware)


def get_agent(request: Request) -> Agent:
    return request.app.state.agent


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/", include_in_schema=False)
def chat_page():
    """A small chat page that talks to POST /chat, so anyone can try the agent."""
    return FileResponse(CHAT_PAGE, media_type="text/html")


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


# How often /chat checks whether the caller has gone away (pressed Stop, closed the tab).
DISCONNECT_CHECK_SECONDS = 0.5
CLIENT_CLOSED_REQUEST = 499  # nginx's status for "the client hung up first"


@app.post(
    "/chat",
    response_model=ChatResponse,
    response_model_exclude_none=True,
    responses={
        CLIENT_CLOSED_REQUEST: {"description": "The caller stopped waiting; the answer was cancelled."}
    },
)
async def chat(body: ChatRequest, request: Request, agent: Agent = Depends(get_agent)):
    correlation_id = structlog.contextvars.get_contextvars().get("correlation_id")
    task = asyncio.create_task(
        agent.ask(body.message, thread_id=body.thread_id, correlation_id=correlation_id)
    )
    # While the agent works, keep checking that someone is still waiting for the
    # answer. If not, cancel it so no more LLM calls (and tokens) are spent.
    while not task.done():
        await asyncio.wait({task}, timeout=DISCONNECT_CHECK_SECONDS)
        if not task.done() and await request.is_disconnected():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            log.info("chat_cancelled", thread_id=body.thread_id)
            return Response(status_code=CLIENT_CLOSED_REQUEST)
    try:
        result = task.result()
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
