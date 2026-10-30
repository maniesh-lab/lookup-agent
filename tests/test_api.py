"""The HTTP API, with the agent replaced by a fake (the server's startup,
which needs Postgres and an API key, is not run)."""

import pytest
from fastapi.testclient import TestClient

from app import main
from app.agent.runner import ChatResult
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


def test_long_request_id_is_cut(client):
    response = client.get("/health", headers={"X-Request-ID": "x" * 5000})
    assert response.headers["X-Request-ID"] == "x" * main.MAX_REQUEST_ID_CHARS


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
