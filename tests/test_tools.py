"""Tool helpers that don't need a database or network."""

import json
from decimal import Decimal

import httpx
import psycopg
import pytest
from pydantic import BaseModel

from app.tools.external_api import convert
from app.tools.query_database import MAX_ROW_BYTES, UnsafeQuery, capped_query, check_sql, run_query
from app.tools.registry import Tool, ToolRegistry, default_registry

# --- SQL guard -------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT count(*) FROM orders",
        "select * from customers where plan = 'premium';",
        "WITH t AS (SELECT 1) SELECT * FROM t",
        "SELECT * FROM support_tickets WHERE description LIKE '%delete my account%'",
        "SELECT created_at, updated_total FROM x",  # words that only contain 'create'/'update'
        "SELECT * FROM support_tickets WHERE description LIKE '%;%'",  # ';' inside a string is fine
        "SELECT count(*) FROM orders -- delete this later",  # words in a comment aren't code
        "/* top sellers */ SELECT name FROM products",
        'SELECT "delete" FROM x',  # a quoted name, not the keyword
        "SELECT $$a;b$$, $t$ ; $t$",  # dollar-quoted strings
        "SELECT a$b FROM t",  # '$' inside a name doesn't start a dollar quote
    ],
)
def test_allowed_queries(sql):
    assert check_sql(sql)


@pytest.mark.parametrize(
    "sql, reason",
    [
        ("", "empty"),
        ("DELETE FROM orders", "Only SELECT"),
        ("SELECT 1; DROP TABLE orders", "one statement"),
        ("WITH x AS (DELETE FROM orders RETURNING *) SELECT * FROM x", "found 'DELETE'"),
        ("SELECT set_config('x', 'y', false); SET ROLE postgres", "one statement"),
        # E'\'' is a whole string; read as a plain string it would hide the DELETE
        ("WITH a AS (SELECT E'\\''), b AS (DELETE FROM orders RETURNING 1) SELECT ''", "found 'DELETE'"),
        ("SELECT * INTO copy_of_orders FROM orders", "found 'INTO'"),
        # A quote character inside a comment, dollar quote or quoted name used to make
        # the check think a string started, hiding the ';' of a second statement.
        ("SELECT $$'$$; DROP TABLE orders; SELECT $$'$$", "one statement"),
        ("SELECT 1 /* ' */ ; DELETE FROM orders; /* ' */", "one statement"),
        ("SELECT 1 -- '\n; DELETE FROM orders --'", "one statement"),
        ('SELECT 1 AS "\'", 2; DROP TABLE orders; SELECT 3 AS "\'"', "one statement"),
        ("SELECT 1 --x\r; DROP TABLE orders", "one statement"),  # Postgres ends a comment at \r
        ("SELECT $\u00e9$'$\u00e9$; DROP TABLE orders; SELECT $\u00e9$'$\u00e9$", "one statement"),
        ("SELECT 1 /* unclosed ; DROP TABLE orders", "one statement"),
    ],
)
def test_blocked_queries(sql, reason):
    with pytest.raises(UnsafeQuery, match=reason):
        check_sql(sql)


class FakeCursor:
    def __init__(self, conn, name):
        self.conn, self.name = conn, name

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql):
        self.conn.executed.append(sql)
        if self.conn.error is not None:
            raise self.conn.error
        self.description = [type("Col", (), {"name": n})() for n in self.conn.columns]

    def fetchmany(self, size):
        return self.conn.rows[:size]


class FakeConnection:
    """Just enough of a psycopg connection for run_query."""

    def __init__(self, columns=(), rows=(), error=None):
        self.columns, self.rows, self.error = list(columns), list(rows), error
        self.executed: list[str] = []
        self.rolled_back = False

    def cursor(self, name=None):
        return FakeCursor(self, name)

    def rollback(self):
        self.rolled_back = True


def test_run_query_reads_columns_without_running_the_query_then_caps_rows():
    conn = FakeConnection(["name", "total"], [("Ana", Decimal("12.50"), 40), ("Bo", Decimal("3"), 40)])
    result = run_query("SELECT name, total FROM orders", conn=conn)
    assert result == {
        "columns": ["name", "total"],
        "rows": [["Ana", 12.5], ["Bo", 3.0]],
        "row_count": 2,
        "truncated": False,
    }
    describe, capped = conn.executed
    assert describe.endswith("LIMIT 0") and "SELECT name, total FROM orders" in describe
    assert f"<= {MAX_ROW_BYTES}" in capped and "AS q(c0, c1)" in capped
    assert conn.rolled_back


def test_run_query_leaves_out_oversized_rows_and_says_so():
    conn = FakeConnection(["big"], [(None, MAX_ROW_BYTES + 1), ("small", 30)])
    result = run_query("SELECT big FROM t", conn=conn)
    assert result["rows"] == [["small"]] and result["rows_left_out"] == 1
    assert "left(description, 200)" in result["note"]


def test_run_query_marks_a_cut_result_and_shortens_long_values():
    rows = [("x" * 5000, 5100)] + [("y", 30)] * 10
    result = run_query("SELECT v FROM t", conn=FakeConnection(["v"], rows), row_limit=3)
    assert result["truncated"] is True and result["row_count"] == 3
    assert result["rows"][0][0].endswith("(cut, 5000 characters in total)")


def test_run_query_turns_database_errors_into_messages():
    timeout = run_query("SELECT 1", conn=FakeConnection(error=psycopg.errors.QueryCanceled()))
    assert "was stopped" in timeout["error"]
    denied = run_query("SELECT 1", conn=FakeConnection(error=psycopg.errors.InsufficientPrivilege("no")))
    assert denied["error"].startswith("Database error:")
    assert run_query("DROP TABLE orders", conn=FakeConnection())["error"].startswith("Only SELECT")


def test_capped_query_keeps_a_trailing_comment_from_hiding_the_end():
    sql = capped_query("SELECT 1 -- note", 1)
    assert "SELECT 1 -- note\n)" in sql
    assert capped_query("SELECT FROM t", 0).startswith("SELECT s.row_bytes")


# --- Currency API ------------------------------------------------------------


def mock_client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_convert_success():
    def handler(request):
        assert request.url.params["base"] == "EUR"
        assert request.url.params["symbols"] == "INR"
        return httpx.Response(
            200, json={"amount": 1.0, "base": "EUR", "date": "2026-10-16", "rates": {"INR": 97.5}}
        )

    result = convert(250, "eur", "inr", client=mock_client(handler))
    assert result["converted"] == 24375.0
    assert result["rate"] == 97.5
    assert result["rate_date"] == "2026-10-16"


def test_convert_same_currency_skips_the_api():
    def handler(request):  # pragma: no cover - must not be called
        raise AssertionError("API should not be called")

    assert convert(10, "USD", "usd", client=mock_client(handler))["converted"] == 10


def test_convert_timeout_gives_a_clean_message():
    def handler(request):
        raise httpx.ConnectTimeout("too slow")

    result = convert(10, "USD", "GBP", client=mock_client(handler))
    assert "didn't answer in time" in result["error"]


def test_convert_server_error_gives_a_clean_message():
    result = convert(10, "USD", "GBP", client=mock_client(lambda r: httpx.Response(500)))
    assert "isn't available" in result["error"]


def test_convert_unknown_currency():
    result = convert(10, "USD", "XYZ", client=mock_client(lambda r: httpx.Response(404)))
    assert "doesn't support" in result["error"]


def test_convert_rejects_bad_codes():
    assert "3 letters" in convert(10, "US", "GBP")["error"]


# --- Registry -----------------------------------------------------------------


class EchoArgs(BaseModel):
    text: str


def test_registry_runs_a_tool_and_reports_problems():
    def broken(text):
        raise RuntimeError("bug")

    registry = ToolRegistry(
        [
            Tool("echo", "Echo.", EchoArgs, lambda text: text.upper(), "document"),
            Tool("broken", "Broken.", EchoArgs, broken, "document"),
        ]
    )
    assert registry.run("echo", {"text": "hi"}) == "HI"
    assert "Unknown tool" in json.loads(registry.run("nope", {}))["error"]
    assert "Bad arguments for echo: text" in json.loads(registry.run("echo", {}))["error"]
    assert "RuntimeError" in json.loads(registry.run("broken", {"text": "x"}))["error"]


def test_default_registry_specs():
    specs = default_registry().specs()
    names = [s["function"]["name"] for s in specs]
    assert names == ["search_docs", "query_database", "convert_currency"]
    sql_spec = specs[1]["function"]
    assert "support_tickets" in sql_spec["description"]  # the schema guide is included
    assert sql_spec["parameters"]["required"] == ["sql"]
