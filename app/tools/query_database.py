"""Tool 2: run a read-only SQL query against the shop database.

Safety comes in layers, so one mistake doesn't open a hole:
1. The query must be a single SELECT (or WITH ... SELECT) statement.
2. It runs as lookup_reader, a user that can only SELECT from the shop
   tables (see db/schema.sql).
3. The transaction is read-only and has a statement timeout.
4. At most sql_row_limit rows come back; the result says if it was cut.

Errors are returned to the model as text instead of raised, so it can
read the Postgres error and fix its query.
"""

import json
import re
from datetime import date, datetime
from decimal import Decimal

import psycopg
from pydantic import BaseModel, Field

from app.core.config import settings
from app.db import connect_readonly

SCHEMA_GUIDE = """\
Tables (PostgreSQL). All money is in USD. Orders and tickets cover 2026-01-01 to 2026-06-30;
customers signed up from 2025-07-01 on.
- customers(id, full_name, email, city, country, plan, signup_date)
    plan: 'basic' or 'premium'
- products(id, name, category, price)
    category: 'electronics', 'home', 'apparel', 'books', 'sports', 'beauty'
- orders(id, customer_id -> customers.id, order_date, status, total)
    status: 'delivered', 'processing', 'cancelled'; total = sum of its items
- order_items(id, order_id -> orders.id, product_id -> products.id, quantity, unit_price)
- support_tickets(id, customer_id -> customers.id, order_id -> orders.id (may be NULL),
                  topic, status, created_at, description)
    topic: 'delivery', 'refund', 'damaged_item', 'quality'; status: 'open', 'closed'
There is no data about costs, profit, stock levels, warehouse staff or website traffic."""

DESCRIPTION = (
    "Run ONE read-only SQL SELECT query on the shop database and get the rows back as JSON. "
    "Use it for questions about customers, orders, products, totals, counts or support tickets. "
    "Dates are DATE columns: compare with 'YYYY-MM-DD' literals; BETWEEN is inclusive. "
    f"Results are limited to {settings.sql_row_limit} rows.\n\n{SCHEMA_GUIDE}"
)

FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|truncate|grant|revoke|copy|call|do|"
    r"vacuum|analyze|lock|set|reset|comment|refresh|listen|notify)\b",
    re.IGNORECASE,
)


class QueryDatabaseArgs(BaseModel):
    sql: str = Field(description="A single PostgreSQL SELECT statement")


class UnsafeQuery(ValueError):
    pass


def check_sql(sql: str) -> str:
    """Return the cleaned query, or raise UnsafeQuery with a reason the model can act on."""
    cleaned = sql.strip().rstrip(";").strip()
    if not cleaned:
        raise UnsafeQuery("The query is empty.")
    if ";" in cleaned:
        raise UnsafeQuery("Only one statement is allowed (remove the extra ';').")
    no_strings = re.sub(r"'(?:[^']|'')*'", "''", cleaned)  # ignore words inside string literals
    if not re.match(r"^\s*(select|with)\b", no_strings, re.IGNORECASE):
        raise UnsafeQuery("Only SELECT queries are allowed.")
    bad = FORBIDDEN.search(no_strings)
    if bad:
        raise UnsafeQuery(f"Only read-only SELECT queries are allowed (found '{bad.group(0)}').")
    return cleaned


def _jsonable(value):
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return value


def run_query(sql: str, *, conn=None, row_limit: int | None = None) -> dict:
    row_limit = row_limit or settings.sql_row_limit
    try:
        cleaned = check_sql(sql)
    except UnsafeQuery as exc:
        return {"error": str(exc)}

    own_conn = conn is None
    try:
        conn = conn or connect_readonly()
        with conn.cursor() as cur:
            cur.execute(cleaned)
            columns = [c.name for c in cur.description]
            rows = cur.fetchmany(row_limit + 1)
        conn.rollback()  # nothing to save; ends the read-only transaction
    except psycopg.errors.QueryCanceled:
        return {
            "error": f"The query took longer than {settings.sql_timeout_ms} ms and was stopped. "
            "Simplify it or add filters."
        }
    except psycopg.Error as exc:
        message = (exc.diag.message_primary if exc.diag else None) or str(exc)
        return {"error": f"Database error: {message}"}
    finally:
        if own_conn and conn is not None:
            conn.close()

    truncated = len(rows) > row_limit
    rows = rows[:row_limit]
    return {
        "columns": columns,
        "rows": [[_jsonable(v) for v in row] for row in rows],
        "row_count": len(rows),
        "truncated": truncated,
    }


def query_database(sql: str) -> str:
    return json.dumps(run_query(sql))
