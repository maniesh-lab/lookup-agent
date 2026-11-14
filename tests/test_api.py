"""The HTTP API, with the agent replaced by a fake (the server's startup,
which needs Postgres and an API key, is not run)."""

import asyncio
import base64
import hashlib
import re
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import main
from app.agent.runner import ChatResult, ThreadTooLong
from app.retrieval.ingest import IngestStats


class FakeAgent:
    def __init__(self):
        self.asked = []

    async def ask(self, question, thread_id=None, correlation_id=None):
        self.asked.append((question, thread_id, correlation_id))
        return ChatResult(
            answer="You have 30 days.",
            thread_id=thread_id or "new-thread",
            prompt_version="v2",
            steps=2,
            tools_used=["search_docs"],
            sources=[{"type": "document", "name": "returns.md", "section": "Returns Policy > Return window"}],
        )


@pytest.fixture
def agent():
    fake = FakeAgent()
    main.app.dependency_overrides[main.get_agent] = lambda: fake
    yield fake
    main.app.dependency_overrides.clear()


@pytest.fixture(autouse=True)
def open_chat_defaults(monkeypatch):
    """Every test starts with the defaults: no API key, the default rate limit, no counts."""
    monkeypatch.setattr(main.settings, "chat_api_key", None)
    monkeypatch.setattr(main.settings, "chat_rate_limit_per_minute", 20)
    main.chat_rate_limiter.reset()
    yield
    main.chat_rate_limiter.reset()


@pytest.fixture
def client():
    return TestClient(main.app)  # not used as a context manager, so startup doesn't run


def test_chat_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "Lumen Lane help" in response.text
    assert 'fetch("/chat"' in response.text


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_every_response_has_a_request_id(client):
    response = client.get("/health")
    assert response.headers["X-Request-ID"]


def test_request_id_from_the_caller_is_kept(client):
    response = client.get("/health", headers={"X-Request-ID": "abc-123"})
    assert response.headers["X-Request-ID"] == "abc-123"


@pytest.mark.parametrize("given", ["x" * 5000, "has spaces", "caf\u00e9".encode(), "a\tb"])
def test_odd_request_ids_are_replaced_with_a_fresh_one(client, given):
    """Too long, spaces or non-ASCII: the header sent back would be messy (non-ASCII came back as
    garbled bytes that some clients can't read), so a new id is made instead."""
    response = client.get("/health", headers={"X-Request-ID": given})
    assert re.fullmatch(r"[0-9a-f-]{36}", response.headers["X-Request-ID"])


def test_a_crash_still_returns_a_request_id(client):
    class BrokenAgent:
        async def ask(self, *args, **kwargs):
            raise RuntimeError("database went away")

    main.app.dependency_overrides[main.get_agent] = BrokenAgent
    try:
        response = client.post("/chat", json={"message": "hi"}, headers={"X-Request-ID": "req-500"})
    finally:
        main.app.dependency_overrides.clear()
    assert response.status_code == 500
    assert response.headers["X-Request-ID"] == "req-500"
    assert response.text == "Internal Server Error"


def test_chat(client, agent):
    response = client.post(
        "/chat", json={"message": "what's your return window?"}, headers={"X-Request-ID": "req-1"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "You have 30 days."
    assert body["thread_id"] == "new-thread"
    assert body["sources"][0] == {
        "type": "document",
        "name": "returns.md",
        "section": "Returns Policy > Return window",
    }  # empty fields (query, detail) are left out, not sent as null
    assert body["tools_used"] == ["search_docs"]
    # The request's correlation id reaches the agent (and from there the traces).
    assert agent.asked == [("what's your return window?", None, "req-1")]


def test_chat_continues_a_thread(client, agent):
    response = client.post("/chat", json={"message": "and for premium?", "thread_id": "t-1"})
    assert response.json()["thread_id"] == "t-1"


def test_chat_rejects_an_empty_message(client, agent):
    assert client.post("/chat", json={"message": ""}).status_code == 422


def test_ingest(client, monkeypatch):
    import app.retrieval.ingest as ingest_module

    monkeypatch.setattr(
        ingest_module, "ingest_folder", lambda: IngestStats(added=["returns.md"], chunks_written=5)
    )
    monkeypatch.setattr(main.settings, "ingest_token", None)
    response = client.post("/ingest")
    assert response.status_code == 200
    assert response.json()["added"] == ["returns.md"]


def test_ingest_needs_the_token_when_one_is_set(client, monkeypatch):
    import app.retrieval.ingest as ingest_module

    monkeypatch.setattr(ingest_module, "ingest_folder", lambda: IngestStats())
    monkeypatch.setattr(main.settings, "ingest_token", "secret")
    assert client.post("/ingest").status_code == 401
    assert client.post("/ingest", headers={"X-Ingest-Token": "wrong"}).status_code == 401
    assert client.post("/ingest", headers={"X-Ingest-Token": "é".encode("latin-1")}).status_code == 401
    assert client.post("/ingest", headers={"X-Ingest-Token": "secret"}).status_code == 200


def test_chat_returns_503_when_the_llm_is_down(client):
    from steadfast import AllProvidersFailed

    class DownAgent:
        async def ask(self, *args, **kwargs):
            raise AllProvidersFailed([("groq", RuntimeError("down"))])

    main.app.dependency_overrides[main.get_agent] = lambda: DownAgent()
    try:
        response = client.post("/chat", json={"message": "hi"})
    finally:
        main.app.dependency_overrides.clear()
    assert response.status_code == 503
    assert "unavailable" in response.json()["detail"]


async def test_chat_is_cancelled_when_the_caller_hangs_up(monkeypatch):
    import asyncio

    from app.main import ChatRequest, chat

    cancelled = asyncio.Event()

    class SlowAgent:
        async def ask(self, *args, **kwargs):
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                cancelled.set()
                raise

    class GoneRequest:
        async def is_disconnected(self):
            return True

    monkeypatch.setattr(main, "DISCONNECT_CHECK_SECONDS", 0.01)
    response = await chat(ChatRequest(message="hi"), GoneRequest(), SlowAgent())

    assert response.status_code == 499
    assert cancelled.is_set()


def test_chat_page_has_a_stop_button(client):
    page = client.get("/").text
    assert "AbortController" in page and 'id="i-stop"' in page


async def test_search_warm_up_failure_does_not_stop_the_server(monkeypatch):
    def broken_warm_up(mode):
        raise OSError("no network and no cached model")

    monkeypatch.setattr(main, "warm_up", broken_warm_up)
    await main.warm_up_search()  # logged, not raised


async def test_search_warm_up_loads_for_the_configured_mode(monkeypatch):
    modes = []
    monkeypatch.setattr(main, "warm_up", lambda mode: modes.append(mode) or 0.5)
    monkeypatch.setattr(main.settings, "search_mode", "hybrid")
    await main.warm_up_search()
    assert modes == ["hybrid"]


# --- API key and rate limit on /chat ------------------------------------------------


def test_chat_needs_no_key_by_default(client, agent):
    response = client.post("/chat", json={"message": "hi"})
    assert response.status_code == 200
    assert len(agent.asked) == 1


@pytest.mark.parametrize(
    "headers", [{}, {"X-API-Key": "wrong"}, {"X-API-Key": ""}, {"X-API-Key": "é".encode("latin-1")}]
)
def test_chat_refuses_a_missing_or_wrong_key(client, agent, monkeypatch, headers):
    monkeypatch.setattr(main.settings, "chat_api_key", "the-key")
    response = client.post("/chat", json={"message": "hi"}, headers=headers)
    assert response.status_code == 401
    assert "needs an API key" in response.json()["detail"]
    assert agent.asked == []  # refused before the agent ran


def test_chat_accepts_the_right_key(client, agent, monkeypatch):
    monkeypatch.setattr(main.settings, "chat_api_key", "the-key")
    response = client.post("/chat", json={"message": "hi"}, headers={"X-API-Key": "the-key"})
    assert response.status_code == 200
    assert len(agent.asked) == 1


def test_chat_rate_limit_returns_429_before_the_agent_runs(client, agent, monkeypatch):
    monkeypatch.setattr(main.settings, "chat_rate_limit_per_minute", 3)
    statuses = [client.post("/chat", json={"message": "hi"}).status_code for _ in range(5)]
    assert statuses == [200, 200, 200, 429, 429]
    assert len(agent.asked) == 3

    response = client.post("/chat", json={"message": "hi"})
    assert response.json()["detail"].startswith("Slow down")
    assert 1 <= int(response.headers["Retry-After"]) <= 60
    assert response.headers["X-Request-ID"]


def test_wrong_keys_count_towards_the_rate_limit(client, agent, monkeypatch):
    monkeypatch.setattr(main.settings, "chat_api_key", "the-key")
    monkeypatch.setattr(main.settings, "chat_rate_limit_per_minute", 2)
    statuses = [
        client.post("/chat", json={"message": "hi"}, headers={"X-API-Key": "guess"}).status_code
        for _ in range(3)
    ]
    assert statuses == [401, 401, 429]


def test_chat_rate_limit_can_be_turned_off(client, agent, monkeypatch):
    monkeypatch.setattr(main.settings, "chat_rate_limit_per_minute", 0)
    assert all(client.post("/chat", json={"message": "hi"}).status_code == 200 for _ in range(30))


def test_chat_page_explains_401_and_429_without_holding_a_key(client):
    page = client.get("/").text
    assert "This server needs an API key" in page and "Slow down" in page
    assert 'X-API-Key"' not in page and 'headers: { "Content-Type": "application/json" }' in page


# --- Server startup and shutdown (lifespan), with fakes instead of Postgres ---------


class FakePool:
    check_connection = None  # the lifespan passes AsyncConnectionPool.check_connection

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def fake_startup(monkeypatch, *, setup_fails=False):
    """Swap the lifespan's database pieces for fakes; return a log of what ran."""
    import langgraph.checkpoint.postgres.aio as saver_module
    import psycopg_pool

    ran = []

    class FakeSaver:
        def __init__(self, pool):
            pass

        async def setup(self):
            if setup_fails:
                raise OSError("Postgres is down")

    monkeypatch.setattr(psycopg_pool, "AsyncConnectionPool", FakePool)
    monkeypatch.setattr(saver_module, "AsyncPostgresSaver", FakeSaver)
    monkeypatch.setattr(main, "warm_up", lambda mode: ran.append(f"warm_up {mode}") or 0.1)
    monkeypatch.setattr(main, "create_agent", lambda checkpointer: FakeAgent())
    monkeypatch.setattr(main, "close_search_pool", lambda: ran.append("close_search_pool"))
    return ran


def test_startup_loads_search_before_the_first_question_and_shutdown_closes_the_pool(monkeypatch):
    ran = fake_startup(monkeypatch)
    with TestClient(main.app) as client:  # as a context manager, so the lifespan runs
        assert ran == [f"warm_up {main.settings.search_mode}"]
        assert client.post("/chat", json={"message": "hi"}).json()["answer"] == "You have 30 days."
    assert ran[-1] == "close_search_pool"


def test_the_search_pool_is_closed_even_when_startup_fails(monkeypatch):
    ran = fake_startup(monkeypatch, setup_fails=True)
    with pytest.raises(OSError, match="Postgres is down"):
        with TestClient(main.app):
            pass
    assert ran[-1] == "close_search_pool"


# --- Limits and protective headers --------------------------------------------------


def test_a_too_large_body_is_refused_before_it_is_read(client, agent):
    body = b'{"message": "' + b"x" * main.MAX_REQUEST_BYTES + b'"}'
    response = client.post("/chat", content=body, headers={"Content-Type": "application/json"})
    assert response.status_code == 413 and "too large" in response.json()["detail"]
    assert response.headers["X-Request-ID"] and response.headers["X-Content-Type-Options"] == "nosniff"
    assert agent.asked == [] and main.chat_rate_limiter.tracked_clients() == 0


def test_a_too_large_body_without_a_length_is_refused_while_it_arrives(client, agent):
    def chunks():  # no Content-Length: the body is sent in pieces
        yield b'{"message": "'
        for _ in range(8):
            yield b"x" * 16_384
        yield b'"}'

    response = client.post("/chat", content=chunks(), headers={"Content-Type": "application/json"})
    assert response.status_code == 413 and agent.asked == []


def test_a_normal_long_question_still_fits(client, agent):
    emoji_question = "\U0001f600" * 2000  # the longest allowed message, in the bulkiest JSON form
    response = client.post("/chat", json={"message": emoji_question})
    assert response.status_code == 200


@pytest.mark.parametrize("thread_id", ["t" * 129, "has space", "../../etc", "a;b"])
def test_odd_thread_ids_are_refused(client, agent, thread_id):
    response = client.post("/chat", json={"message": "hi", "thread_id": thread_id})
    assert response.status_code == 422 and agent.asked == []


def test_a_full_conversation_gets_409(client):
    class FullAgent:
        async def ask(self, *args, **kwargs):
            raise ThreadTooLong("This conversation has reached its limit of 50 questions.")

    main.app.dependency_overrides[main.get_agent] = FullAgent
    try:
        response = client.post("/chat", json={"message": "one more", "thread_id": "t-1"})
    finally:
        main.app.dependency_overrides.clear()
    assert response.status_code == 409 and "limit of 50" in response.json()["detail"]


@pytest.mark.parametrize("path", ["/health", "/", "/no-such-page"])
def test_every_response_has_the_protective_headers(client, path):
    response = client.get(path)
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["Referrer-Policy"] == "no-referrer"


def test_chat_page_policy_allows_only_its_own_script_and_no_outside_images(client):
    csp = client.get("/").headers["Content-Security-Policy"]
    page = Path(main.CHAT_PAGE).read_text(encoding="utf-8")
    for script in re.findall(r"<script>(.*?)</script>", page, re.DOTALL):
        digest = base64.b64encode(hashlib.sha256(script.encode()).digest()).decode()
        assert f"'sha256-{digest}'" in csp
    assert "unsafe-inline" not in csp and "img-src 'self' data:" in csp and "default-src 'none'" in csp
    assert 'style="' not in page  # inline style attributes would need 'unsafe-inline'


def test_chat_page_pins_its_libraries_and_strips_images_from_answers(client):
    page = client.get("/").text
    assert page.count('integrity="sha384-') == 2
    assert '"img"' in page and "FORBID_TAGS" in page


async def test_two_ingests_never_run_at_once(monkeypatch):
    import app.retrieval.ingest as ingest_module

    running, most = 0, 0
    lock = threading.Lock()

    def slow_ingest():
        nonlocal running, most
        with lock:
            running += 1
            most = max(most, running)
        time.sleep(0.05)
        with lock:
            running -= 1
        return IngestStats()

    monkeypatch.setattr(ingest_module, "ingest_folder", slow_ingest)
    await asyncio.gather(main.ingest(), main.ingest())
    assert most == 1
