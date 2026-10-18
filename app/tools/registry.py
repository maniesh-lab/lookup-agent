"""One place that lists the agent's tools: name, description, argument
model and the function that runs it.

The agent gets `specs()` (what the model sees) and calls `run()` when the
model asks for a tool. Tests build a registry with fake functions.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass

import structlog
from pydantic import BaseModel, ValidationError
from steadfast import function_tool

log = structlog.get_logger(__name__)


@dataclass
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    func: Callable[..., str]
    source_type: str  # "document", "database" or "api", for the sources list


class ToolRegistry:
    def __init__(self, tools: list[Tool]):
        self.tools = {t.name: t for t in tools}

    def specs(self) -> list[dict]:
        return [function_tool(t.name, t.description, t.args_model) for t in self.tools.values()]

    def run(self, name: str, arguments: dict | None) -> str:
        """Run a tool. Problems become a message for the model, not an exception."""
        tool = self.tools.get(name)
        if tool is None:
            return json.dumps({"error": f"Unknown tool {name!r}. Available: {sorted(self.tools)}"})
        try:
            args = tool.args_model(**(arguments or {}))
        except ValidationError as exc:
            first = exc.errors()[0]
            field = ".".join(str(p) for p in first["loc"])
            return json.dumps({"error": f"Bad arguments for {name}: {field}: {first['msg']}"})
        try:
            return tool.func(**args.model_dump())
        except Exception as exc:  # a tool bug shouldn't crash the whole conversation
            log.exception("tool_failed", tool=name)
            return json.dumps({"error": f"The {name} tool failed: {type(exc).__name__}"})


def default_registry() -> ToolRegistry:
    from app.tools import external_api, query_database, search_docs

    return ToolRegistry(
        [
            Tool(
                "search_docs",
                search_docs.DESCRIPTION,
                search_docs.SearchDocsArgs,
                search_docs.search_docs,
                "document",
            ),
            Tool(
                "query_database",
                query_database.DESCRIPTION,
                query_database.QueryDatabaseArgs,
                query_database.query_database,
                "database",
            ),
            Tool(
                "convert_currency",
                external_api.DESCRIPTION,
                external_api.ConvertCurrencyArgs,
                external_api.convert_currency,
                "api",
            ),
        ]
    )
