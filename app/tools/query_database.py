"""Tool 2: run a read-only SQL query against the shop database.

Safety comes in layers, so one mistake doesn't open a hole:
1. The query must be a single SELECT (or WITH ... SELECT) statement.
   Comments, strings and quoted names are blanked out before the check,
   so nothing can hide a ';' or a keyword inside them.
2. It runs wrapped inside a subquery, through a server-side cursor. A
   second statement can't fit inside a subquery, and psycopg sends a
   server-side cursor's query in a form Postgres runs as one command only.
3. It runs as lookup_reader, a user that can only SELECT from the shop
   tables (see db/roles.sql).
4. The transaction is read-only and has a statement timeout.
5. At most sql_row_limit rows come back; the result says if it was cut.
   The cursor fetches only those rows, rows over MAX_ROW_BYTES are left
   out by Postgres itself (so a huge value never reaches the app), and
   long values are shortened.

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
    r"vacuum|analyze|lock|set|reset|comment|refresh|listen|notify|into)\b",
    re.IGNORECASE,
)
# Parts of a query Postgres doesn't read as code: comments, strings and quoted
# names. They are blanked out before the checks, so a keyword or ';' inside them
# doesn't count, and a quote character inside one can't hide code from the check.
# One left-to-right pass, so whichever starts first wins: a quote inside a comment
# belongs to the comment, and "--" inside a string belongs to the string.
# Postgres counts every non-ASCII character as a letter in names and $tags$.
_LETTER = r"A-Za-z_\u0080-\U0010ffff"
NOT_CODE = re.compile(
    r"--[^\r\n]*"  # line comment (Postgres ends it at \r too)
    r"|/\*.*?\*/"  # block comment (Postgres nests them; ending early only blocks more)
    rf"|(?<![{_LETTER}0-9$])\$(?P<tag>(?:[{_LETTER}][{_LETTER}0-9]*)?)\$.*?\$(?P=tag)\$"  # $$..$$, $x$..$x$
    rf"|(?<![{_LETTER}0-9$])[eE]'(?:[^'\\]|\\.|'')*'"  # E'...': backslash escapes, so before plain
    r"|'(?:[^']|'')*'"  # plain string
    r'|"(?:[^"]|"")*"',  # quoted name, e.g. "order"
    re.DOTALL,
)
MAX_CELL_CHARS = 2000  # longer text values are cut so one cell can't flood the conversation
# Rows bigger than this (as Postgres stores them) are left out before they are sent, so
# one query can't fill the app's memory: a single SELECT repeat('x', 200000000) used to
# cost the app about 430 MB. At most sql_row_limit + 1 rows of this size are fetched.
MAX_ROW_BYTES = 128_000


class QueryDatabaseArgs(BaseModel):
    sql: str = Field(description="A single PostgreSQL SELECT statement")


class UnsafeQuery(ValueError):
    pass


def check_sql(sql: str) -> str:
    """Return the cleaned query, or raise UnsafeQuery with a reason the model can act on."""
    cleaned = sql.strip().rstrip(";").strip()
    if not cleaned:
        raise UnsafeQuery("The query is empty.")
    code = NOT_CODE.sub(" ", cleaned)  # ignore words and ';' inside comments, strings and quoted names
    if ";" in code:
        raise UnsafeQuery("Only one statement is allowed (remove the extra ';').")
    if not re.match(r"^\s*(select|with)\b", code, re.IGNORECASE):
        raise UnsafeQuery("Only SELECT queries are allowed.")
    bad = FORBIDDEN.search(code)
    if bad:
        raise UnsafeQuery(f"Only read-only SELECT queries are allowed (found '{bad.group(0)}').")
    return cleaned


def _jsonable(value):
    if isinstance(value, str) and len(value) > MAX_CELL_CHARS:
        return value[:MAX_CELL_CHARS] + f"... (cut, {len(value)} characters in total)"
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
            # LIMIT 0 plans the query without running it: this only reads the column names.
            cur.execute(f"SELECT * FROM (\n{cleaned}\n) AS q LIMIT 0")
            columns = [c.name for c in cur.description]
        # A named (server-side) cursor: rows stay in Postgres until fetched.
        with conn.cursor(name="agent_query") as cur:
            cur.execute(capped_query(cleaned, len(columns)))
            fetched = cur.fetchmany(row_limit + 1)
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

    truncated = len(fetched) > row_limit
    fetched = fetched[:row_limit]
    # The last value of each row is its size; an oversized row came back as NULLs.
    rows = [row[:-1] for row in fetched if row[-1] <= MAX_ROW_BYTES]
    result = {
        "columns": columns,
        "rows": [[_jsonable(v) for v in row] for row in rows],
        "row_count": len(rows),
        "truncated": truncated,
    }
    left_out = len(fetched) - len(rows)
    if left_out:
        result["rows_left_out"] = left_out
        result["note"] = (
            f"{left_out} row(s) were over {MAX_ROW_BYTES:,} bytes and were left out. "
            "Select fewer or shorter columns, e.g. left(description, 200)."
        )
    return result


def capped_query(sql: str, column_count: int) -> str:
    """Wrap a checked query so Postgres leaves out rows over MAX_ROW_BYTES.

    Each column comes back through CASE, NULL when its row is too big, and
    the row's size is added as the last column. Columns are renamed c0, c1,
    ... inside (their real names come from the LIMIT 0 step), which also
    copes with repeated or unnamed columns. Types and order stay as they
    were. The newlines keep a trailing "-- comment" from hiding the ")".
    """
    sizes = "pg_column_size(q.*) AS row_bytes"
    if not column_count:
        return f"SELECT s.row_bytes FROM (SELECT {sizes} FROM (\n{sql}\n) AS q) AS s"
    names = ", ".join(f"c{i}" for i in range(column_count))
    capped = ", ".join(
        f"CASE WHEN s.row_bytes <= {MAX_ROW_BYTES} THEN s.c{i} END" for i in range(column_count)
    )
    return f"SELECT {capped}, s.row_bytes FROM (SELECT q.*, {sizes} FROM (\n{sql}\n) AS q({names})) AS s"


def query_database(sql: str) -> str:
    return json.dumps(run_query(sql))
