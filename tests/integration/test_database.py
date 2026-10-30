"""Tests against a real Postgres with pgvector.

Run with TEST_DATABASE_URL pointing at a database you don't mind being
rebuilt, for example:
    TEST_DATABASE_URL=postgresql://postgres:postgres@localhost:5432/lookup_test pytest -m integration
"""

import json
import re

import psycopg
import pytest
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.conninfo import conninfo_to_dict, make_conninfo

from app.agent.graph import build_graph
from app.agent.runner import Agent
from app.core.config import settings
from app.db import connect, plain_url
from app.retrieval import search as search_module
from app.retrieval.ingest import ingest_folder
from app.retrieval.search import search
from app.tools import query_database as qd
from app.tools.registry import default_registry
from db import seed
from tests.conftest import FakeEmbedder, FakeLLM, FakeReranker, text_reply, tool_reply

pytestmark = pytest.mark.integration


def readonly_url(url: str) -> str:
    params = conninfo_to_dict(plain_url(url))
    params.update(user="lookup_reader", password="lookup_reader")
    return make_conninfo(**params)


@pytest.fixture
def db(seeded_db):
    with connect(seeded_db["url"]) as conn:
        yield conn


@pytest.fixture
def readonly(seeded_db):
    conn = psycopg.connect(
        readonly_url(seeded_db["url"]),
        options="-c default_transaction_read_only=on -c statement_timeout=1000",
    )
    yield conn
    conn.close()


@pytest.fixture(scope="module")
def ingested(seeded_db):
    with connect(seeded_db["url"], vector=True) as conn:
        return ingest_folder(embedder=FakeEmbedder(), conn=conn)


def one(conn, sql, *params):
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchone()


# --- Seed ---------------------------------------------------------------------


def test_seed_is_repeatable(seeded_db, db):
    assert seed.write(seed.build(), db) == seeded_db["fingerprint"]


def test_planted_cases(db):
    assert (
        one(
            db,
            "SELECT o.total FROM orders o JOIN customers c ON c.id = o.customer_id"
            " WHERE c.full_name = 'Priya Sharma' AND o.order_date = '2026-02-18'",
        )[0]
        == 173.50
    )
    assert (
        one(
            db,
            "SELECT count(*) FROM support_tickets t JOIN customers c ON c.id = t.customer_id"
            " WHERE c.full_name = 'Omar Haddad' AND t.topic = 'damaged_item'"
            " AND t.created_at = '2026-05-09'",
        )[0]
        == 1
    )
    names = one(
        db,
        """
        SELECT array_agg(DISTINCT c.full_name ORDER BY c.full_name)
        FROM customers c
        JOIN orders o ON o.customer_id = c.id
        JOIN support_tickets t ON t.customer_id = c.id
        WHERE c.plan = 'premium' AND o.status = 'delivered'
          AND o.order_date BETWEEN '2026-01-10' AND '2026-03-31'
          AND t.topic = 'refund' AND t.created_at BETWEEN '2026-04-01' AND '2026-04-30'""",
    )[0]
    assert names == ["Anjali Gurung", "Daniel Brooks", "Mei Lin Chen", "Rohan Mehta", "Sophie Laurent"]


def test_no_tie_for_third_place(db):
    with db.cursor() as cur:
        cur.execute("""
            SELECT sum(oi.quantity) FROM order_items oi JOIN orders o ON o.id = oi.order_id
            WHERE o.status <> 'cancelled' AND o.order_date BETWEEN '2026-04-01' AND '2026-06-30'
            GROUP BY oi.product_id ORDER BY 1 DESC LIMIT 4""")
        totals = [r[0] for r in cur.fetchall()]
    assert totals[2] != totals[3]


def test_order_totals_match_their_items(db):
    assert (
        one(
            db,
            """
        SELECT count(*) FROM orders o
        WHERE o.total <> (SELECT sum(quantity * unit_price) FROM order_items WHERE order_id = o.id)""",
        )[0]
        == 0
    )


def test_no_store_data_for_none_questions(db):
    """The NONE questions must stay unanswerable."""
    with db.cursor() as cur:
        cur.execute("SELECT column_name FROM information_schema.columns WHERE table_schema = 'public'")
        columns = {r[0] for r in cur.fetchall()}
    assert not columns & {"cost", "profit", "stock", "packed_by", "visits"}


# --- Read-only SQL tool -------------------------------------------------------------


def test_reader_cannot_write(readonly):
    with pytest.raises(psycopg.errors.ReadOnlySqlTransaction):
        readonly.execute("INSERT INTO products VALUES (999, 'x', 'y', 1)")


def test_reader_cannot_see_other_tables(readonly):
    result = qd.run_query("SELECT * FROM chunks", conn=readonly)
    assert "permission denied" in result["error"]


def test_row_limit_and_timeout(readonly):
    result = qd.run_query("SELECT id FROM order_items", conn=readonly, row_limit=10)
    assert result["row_count"] == 10 and result["truncated"] is True
    readonly.rollback()
    assert "stopped" in qd.run_query("SELECT pg_sleep(3)", conn=readonly)["error"]


def test_long_values_are_cut(readonly):
    result = qd.run_query("SELECT repeat('x', 100000) AS big", conn=readonly)
    cell = result["rows"][0][0]
    assert len(cell) < 2100 and "100000 characters in total" in cell


def test_schema_guide_matches_the_database(db):
    with db.cursor() as cur:
        cur.execute(
            "SELECT table_name, column_name FROM information_schema.columns WHERE table_schema = 'public'"
        )
        actual = {}
        for table, column in cur.fetchall():
            actual.setdefault(table, set()).add(column)
    guide = guide_columns(qd.SCHEMA_GUIDE)
    assert set(guide) == {"customers", "products", "orders", "order_items", "support_tickets"}
    for table, names in guide.items():
        assert names == actual[table], table


def guide_columns(guide: str) -> dict[str, set[str]]:
    """Read "- table(col, col -> other.id (note), ...)" entries from the schema guide."""
    tables = {}
    for match in re.finditer(r"- (\w+)\(", guide):
        depth, i = 1, match.end()
        while depth:
            depth += {"(": 1, ")": -1}.get(guide[i], 0)
            i += 1
        body = re.sub(r"\([^)]*\)", "", guide[match.end() : i - 1])
        tables[match.group(1)] = {part.split()[0] for part in body.split(",") if part.strip()}
    return tables


# --- Ingestion and search --------------------------------------------------------------


def test_ingest_fills_chunks(seeded_db, ingested):
    assert len(ingested.added) == 9
    with connect(seeded_db["url"]) as conn:
        assert one(conn, "SELECT count(*) FROM chunks")[0] == ingested.chunks_written > 0
        again = ingest_folder(embedder=FakeEmbedder(), conn=conn)
    assert again.chunks_written == 0 and len(again.unchanged) == 9


@pytest.mark.parametrize("mode", ["vector", "hybrid", "rerank"])
def test_search_modes(seeded_db, ingested, mode):
    with connect(seeded_db["url"], vector=True) as conn:
        hits = search(
            "who pays for return shipping when you sent the wrong item",
            mode=mode,
            k=3,
            conn=conn,
            embedder=FakeEmbedder(),
            reranker=FakeReranker(),
        )
    assert hits and hits[0].source == "returns.md"


# --- The agent with real tools and a real checkpointer ----------------------------------


async def test_agent_end_to_end_with_real_tools(seeded_db, ingested, monkeypatch):
    monkeypatch.setattr(settings, "database_url", seeded_db["url"])
    monkeypatch.setattr(settings, "readonly_database_url", readonly_url(seeded_db["url"]))
    monkeypatch.setattr(search_module, "get_embedder", FakeEmbedder)
    monkeypatch.setattr(search_module, "get_reranker", FakeReranker)

    llm = FakeLLM(
        [
            tool_reply(
                "query_database",
                {
                    "sql": "SELECT count(*) FROM orders WHERE status = 'processing'"
                    " AND order_date BETWEEN '2026-03-01' AND '2026-03-31'"
                },
                "c1",
            ),
            tool_reply("search_docs", {"query": "how long until an order ships processing time"}, "c2"),
            text_reply("3 orders. They should have shipped within 2 business days."),
            text_reply("You asked about March orders."),
        ]
    )
    registry = default_registry()

    async with AsyncPostgresSaver.from_conn_string(plain_url(seeded_db["url"])) as saver:
        await saver.setup()
        graph = build_graph(llm, registry, prompt_version="v2", max_steps=4, checkpointer=saver)
        agent = Agent(graph, registry, prompt_version="v2", max_steps=4)
        result = await agent.ask("How many March orders are still processing, and how fast should they ship?")

    sql_result = json.loads(result.tool_calls[0].result)
    assert sql_result["rows"] == [[3]]
    assert "shipping.md" in {s["name"] for s in result.sources}
    assert {s["type"] for s in result.sources} == {"database", "document"}

    # A new saver (like after a server restart) still remembers the thread.
    async with AsyncPostgresSaver.from_conn_string(plain_url(seeded_db["url"])) as saver:
        graph = build_graph(llm, registry, prompt_version="v2", max_steps=4, checkpointer=saver)
        agent = Agent(graph, registry, prompt_version="v2", max_steps=4)
        await agent.ask("What did I just ask?", thread_id=result.thread_id)
    remembered = [m.get("content") for m in llm.calls[-1]["messages"]]
    assert any("March orders" in (c or "") for c in remembered)


@pytest.mark.parametrize("eval_set", ["main", "hard"])
def test_every_truth_query_runs(db, eval_set):
    """Each eval case's truth SQL must run on the seeded data and return a value."""
    from eval.run_eval import load_cases

    for case in load_cases(None, eval_set):
        for truth in case.get("truth", []):
            with db.cursor() as cur:
                cur.execute(truth["sql"])
                rows = cur.fetchall()
            assert rows and rows[0][0] is not None, case["id"]


def test_hard_set_traps_really_have_no_data(db):
    """The trap questions only work if the data they ask about doesn't exist."""
    assert one(db, "SELECT count(*) FROM customers WHERE full_name ILIKE %s", "%montoya%")[0] == 0
    assert (
        one(
            db,
            "SELECT count(*) FROM orders o JOIN customers c ON c.id = o.customer_id"
            " WHERE c.full_name = 'Priya Sharma' AND o.order_date = '2026-03-03'",
        )[0]
        == 0
    )
    assert one(db, "SELECT count(*) FROM orders WHERE order_date >= '2026-08-01'")[0] == 0
    assert one(db, "SELECT count(*) FROM products WHERE category ILIKE %s", "%toy%")[0] == 0
