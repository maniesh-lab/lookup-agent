"""Shared test setup.

Unit tests need no database, no API key and no model download: the LLM,
the embedder and the reranker are replaced with small fakes.

Integration tests (marked `integration`) need a Postgres with pgvector.
They run when TEST_DATABASE_URL is set (CI sets it) and are skipped
otherwise. Warning: they rebuild the tables in that database.
"""

import hashlib
import os

# Settings are read when app.core.config is first imported, so the
# environment has to be ready before any app module is imported.
os.environ.setdefault("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/lookup_db")
os.environ.setdefault("LANGFUSE_PUBLIC_KEY", "")
os.environ.setdefault("LANGFUSE_SECRET_KEY", "")

import numpy as np
import pytest
from steadfast import LLMResponse, ToolCall

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeEmbedder:
    """Bag-of-words vectors: texts sharing words get similar vectors."""

    dim = 384

    def _vector(self, text: str) -> list[float]:
        v = np.zeros(self.dim)
        for word in text.lower().split():
            v[int(hashlib.md5(word.strip(".,?!").encode()).hexdigest(), 16) % self.dim] += 1
        norm = np.linalg.norm(v)
        return (v / norm if norm else v).tolist()

    def embed_documents(self, texts):
        return [self._vector(t) for t in texts]

    def embed_query(self, text):
        return self._vector(text)


class FakeReranker:
    def score(self, query, texts):
        q = set(query.lower().split())
        return [float(len(q & set(t.lower().split()))) for t in texts]


class FakeLLM:
    """Returns scripted replies in order and records what it was sent."""

    model = "fake-model"

    def __init__(self, replies: list[LLMResponse]):
        self.replies = list(replies)
        self.calls: list[dict] = []

    async def chat(self, messages, *, tools=None, tool_choice=None):
        self.calls.append(
            {"messages": [dict(m) for m in messages], "tools": tools, "tool_choice": tool_choice}
        )
        if not self.replies:
            return LLMResponse(text="(no more scripted replies)", finish_reason="stop")
        return self.replies.pop(0)


def tool_reply(name: str, arguments: dict, call_id: str = "call_1") -> LLMResponse:
    import json

    return LLMResponse(
        tool_calls=[
            ToolCall(id=call_id, name=name, arguments=arguments, raw_arguments=json.dumps(arguments))
        ],
        finish_reason="tool_calls",
    )


def text_reply(text: str) -> LLMResponse:
    return LLMResponse(text=text, finish_reason="stop")


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()


@pytest.fixture
def fake_reranker():
    return FakeReranker()


# ---------------------------------------------------------------------------
# Integration database
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def test_db_url():
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL not set")
    return url


@pytest.fixture(scope="session")
def seeded_db(test_db_url):
    """Load the schema and seed data once for the whole test session."""
    from pathlib import Path

    from app.db import connect
    from db import seed

    schema = Path(__file__).resolve().parent.parent / "db" / "schema.sql"
    with connect(test_db_url) as conn:
        conn.execute(schema.read_text(encoding="utf-8"))
        conn.commit()
        fingerprint = seed.write(seed.build(), conn)
    return {"url": test_db_url, "fingerprint": fingerprint}
