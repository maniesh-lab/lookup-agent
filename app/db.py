"""Small helpers for Postgres connections: one-off connections (owner and
read-only user) and the shared connection pool that search uses."""

import atexit
import threading

import psycopg
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

from app.core.config import settings


def plain_url(url: str) -> str:
    """psycopg wants postgresql://...; SQLAlchemy-style URLs add +psycopg."""
    return url.replace("postgresql+psycopg://", "postgresql://", 1)


def connect(url: str | None = None, *, vector: bool = False, **kwargs) -> psycopg.Connection:
    """Open a connection to the main database (as the owner user).

    vector=True teaches psycopg to send and read pgvector values.
    """
    conn = psycopg.connect(plain_url(url or settings.database_url), **kwargs)
    if vector:
        register_vector(conn)
    return conn


def connect_readonly(url: str | None = None) -> psycopg.Connection:
    """Open a connection as the read-only user, with a timeout on every query.

    The role is read-only in the database too (see db/roles.sql); the
    options below repeat the rules so they hold even if the role is
    misconfigured.
    """
    options = f"-c statement_timeout={settings.sql_timeout_ms} -c default_transaction_read_only=on"
    return psycopg.connect(
        plain_url(url or settings.readonly_database_url),
        options=options,
        connect_timeout=5,
    )


_search_pool: ConnectionPool | None = None
_search_pool_lock = threading.Lock()
_closes_at_exit = False


def search_pool() -> ConnectionPool:
    """A shared pool of owner-user connections for search, with pgvector types registered.

    Search used to open a new connection for every query (about 10 ms each,
    plus a new Postgres backend). The pool keeps a few open and checks each
    one before handing it out, so a Postgres restart costs one reconnect.
    Search runs in worker threads, so this is the thread-safe sync pool,
    created on first use (the server opens it at startup). Four connections
    are plenty: a search only holds one for its two SQL queries (a few ms)
    and hands it back before reranking. If Postgres is down, a search waits
    up to 10 s for a connection, then the tool returns an error message.
    """
    global _search_pool, _closes_at_exit
    with _search_pool_lock:
        if _search_pool is None:
            _search_pool = ConnectionPool(
                plain_url(settings.database_url),
                min_size=1,
                max_size=4,
                # autocommit: register_vector runs a query, and a pool connection
                # must not be left inside a transaction.
                kwargs={"autocommit": True},
                configure=register_vector,
                check=ConnectionPool.check_connection,
                timeout=10,
                name="search",
                open=True,
            )
            # Scripts and evals never close it themselves. Left to garbage
            # collection, the pool tries to join its threads while Python is
            # shutting down, which Python 3.14 refuses with an error.
            if not _closes_at_exit:
                atexit.register(close_search_pool)
                _closes_at_exit = True
        return _search_pool


def close_search_pool() -> None:
    """Close the search pool (at server shutdown, and between tests)."""
    global _search_pool
    with _search_pool_lock:
        if _search_pool is not None:
            _search_pool.close()
            _search_pool = None
