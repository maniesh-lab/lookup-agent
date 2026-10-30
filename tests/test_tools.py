"""Tool helpers that don't need a database or network."""

import json

import httpx
import pytest
from pydantic import BaseModel

from app.tools.external_api import convert
from app.tools.query_database import UnsafeQuery, check_sql
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
    ],
)
def test_blocked_queries(sql, reason):
    with pytest.raises(UnsafeQuery, match=reason):
        check_sql(sql)


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
