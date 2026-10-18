"""Small helpers for opening Postgres connections."""

import psycopg
from pgvector.psycopg import register_vector

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

    The role is read-only in the database too (see db/schema.sql); the
    options below repeat the rules so they hold even if the role is
    misconfigured.
    """
    options = f"-c statement_timeout={settings.sql_timeout_ms} -c default_transaction_read_only=on"
    return psycopg.connect(
        plain_url(url or settings.readonly_database_url),
        options=options,
        connect_timeout=5,
    )
