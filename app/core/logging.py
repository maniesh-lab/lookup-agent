"""Structured logging: every log line is one JSON object.

A middleware in app/main.py binds a correlation_id for each request with
structlog.contextvars, so every line logged while handling that request
(including Steadfast's own LLM-call logs) carries the same id.
"""

import logging

import structlog


def configure_logging(level: str = "INFO") -> None:
    numeric = logging.getLevelNamesMapping().get(level.upper(), logging.INFO)
    logging.basicConfig(format="%(message)s", level=numeric)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
