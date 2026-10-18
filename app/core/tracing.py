"""Optional tracing with Langfuse.

When LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set, every chat turn
becomes a trace in Langfuse, with one child observation per LLM call and
per tool call. The prompt version is stored on every trace (as its
version and in its metadata), so runs with prompt v1 and v2 can be
compared side by side.

Without the keys every function here does nothing, so the app runs the
same with or without a Langfuse account.
"""

from contextlib import contextmanager, nullcontext
from functools import lru_cache

import structlog

from app.core.config import settings

log = structlog.get_logger(__name__)


class _NoOpObservation:
    def update(self, **kwargs):
        return self


@lru_cache(maxsize=1)
def _client():
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        return None
    try:
        from langfuse import Langfuse
    except ImportError:  # pragma: no cover
        log.warning("langfuse_not_installed")
        return None
    return Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host or None,
    )


def enabled() -> bool:
    return _client() is not None


@contextmanager
def trace_turn(*, thread_id: str, question: str, prompt_version: str, correlation_id: str | None):
    """Wrap one whole chat turn."""
    client = _client()
    if client is None:
        yield _NoOpObservation()
        return

    from langfuse import propagate_attributes

    with (
        client.start_as_current_observation(
            name="chat_turn",
            as_type="agent",
            input={"question": question},
            metadata={"prompt_version": prompt_version, "correlation_id": correlation_id},
        ) as span,
        propagate_attributes(
            session_id=thread_id,
            version=prompt_version,
            trace_name="chat_turn",
            metadata={"prompt_version": prompt_version},
            tags=[f"prompt:{prompt_version}"],
        ),
    ):
        yield span


def observe_llm(*, model: str | None, messages: list[dict]):
    client = _client()
    if client is None:
        return nullcontext(_NoOpObservation())
    return client.start_as_current_observation(
        name="llm_call", as_type="generation", model=model, input=messages
    )


def observe_tool(*, name: str, arguments: dict | None):
    client = _client()
    if client is None:
        return nullcontext(_NoOpObservation())
    return client.start_as_current_observation(name=name, as_type="tool", input=arguments)


def flush() -> None:
    client = _client()
    if client is not None:
        client.flush()
