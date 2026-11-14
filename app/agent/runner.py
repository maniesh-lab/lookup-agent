"""Run one chat turn through the graph and collect the answer and its sources."""

import asyncio
import hashlib
import json
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field

import structlog

from app.core import tracing
from app.tools.registry import ToolRegistry

log = structlog.get_logger(__name__)

NO_ANSWER = "Sorry, I couldn't finish looking that up. Please try asking in a simpler way."


class ThreadTooLong(Exception):
    """The conversation already has the most questions one thread may hold."""


def thread_ref(thread_id: str | None) -> str | None:
    """A short stand-in for a thread_id in logs and traces.

    Whoever knows a thread_id can continue that conversation and ask what was
    said in it, so the id itself is kept out of logs and Langfuse.
    """
    return hashlib.sha256(thread_id.encode()).hexdigest()[:16] if thread_id else None


class _ThreadLocks:
    """One turn at a time per conversation, in this process.

    Two questions sent at once on the same thread would both start from the
    same saved history and one turn would be lost. The second one waits for
    the first instead. Locks are dropped once nobody holds or waits for them.
    """

    def __init__(self):
        self._locks: dict[str, list] = {}  # thread_id -> [lock, holders and waiters]

    @asynccontextmanager
    async def hold(self, thread_id: str):
        entry = self._locks.setdefault(thread_id, [asyncio.Lock(), 0])
        entry[1] += 1
        try:
            async with entry[0]:
                yield
        finally:
            entry[1] -= 1
            if not entry[1]:
                del self._locks[thread_id]

    def __len__(self) -> int:
        return len(self._locks)


@dataclass
class ToolCallRecord:
    name: str
    arguments: dict | None
    result: str


@dataclass
class ChatResult:
    answer: str
    thread_id: str
    prompt_version: str
    steps: int
    tools_used: list[str] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def turn_messages(messages: list[dict]) -> list[dict]:
    """The messages added in the latest turn: everything after the last user message."""
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            return messages[i + 1 :]
    return messages


def collect_tool_calls(messages: list[dict]) -> list[ToolCallRecord]:
    results = {m["tool_call_id"]: m["content"] for m in messages if m.get("role") == "tool"}
    records = []
    for m in messages:
        for call in m.get("tool_calls") or []:
            try:
                args = json.loads(call["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args = None
            records.append(ToolCallRecord(call["function"]["name"], args, results.get(call["id"], "")))
    return records


def build_sources(records: list[ToolCallRecord], registry: ToolRegistry) -> list[dict]:
    """Turn tool results into a deduplicated list of sources for the answer."""
    sources: list[dict] = []
    seen: set[str] = set()

    def add(key: str, item: dict) -> None:
        if key not in seen:
            seen.add(key)
            sources.append(item)

    for rec in records:
        try:
            data = json.loads(rec.result)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict) or "error" in data:
            continue
        tool = registry.tools.get(rec.name)
        kind = tool.source_type if tool else rec.name
        if kind == "document":
            for hit in data.get("results", []):
                add(
                    f"doc:{hit['source']}:{hit['section']}",
                    {"type": "document", "name": hit["source"], "section": hit["section"]},
                )
        elif kind == "database":
            add(
                f"db:{(rec.arguments or {}).get('sql')}",
                {"type": "database", "name": "shop database", "query": (rec.arguments or {}).get("sql")},
            )
        elif kind == "api":
            add(
                f"api:{data.get('from')}:{data.get('to')}",
                {
                    "type": "api",
                    "name": "Frankfurter exchange-rate API",
                    "detail": f"1 {data.get('from')} = {data.get('rate')} {data.get('to')}"
                    f" (rate date {data.get('rate_date')})",
                },
            )
    return sources


class Agent:
    def __init__(
        self,
        graph,
        registry: ToolRegistry,
        *,
        prompt_version: str,
        max_steps: int,
        llm=None,
        max_turns: int = 0,
    ):
        self.graph = graph
        self.llm = llm  # kept so callers (like the eval) can read token usage
        self.registry = registry
        self.prompt_version = prompt_version
        self.max_steps = max_steps
        self.max_turns = max_turns  # questions per thread; 0 means no limit
        self.thread_locks = _ThreadLocks()

    async def ask(
        self, question: str, thread_id: str | None = None, correlation_id: str | None = None
    ) -> ChatResult:
        thread_id = thread_id or str(uuid.uuid4())
        config = {
            "configurable": {"thread_id": thread_id},
            # Each step is 2 graph nodes; leave room so our own cap triggers first.
            "recursion_limit": 2 * self.max_steps + 6,
        }
        async with self.thread_locks.hold(thread_id):
            await self._check_thread_length(config)
            return await self._run_turn(question, thread_id, config, correlation_id)

    async def _check_thread_length(self, config: dict) -> None:
        """Refuse a question once its thread has max_turns of them.

        Every earlier question and answer is sent again with each new one and
        stays in Postgres, so without a cap one thread could grow without end.
        """
        if not self.max_turns or self.graph.checkpointer is None:
            return
        snapshot = await self.graph.aget_state(config)
        asked = sum(1 for m in snapshot.values.get("messages", []) if m.get("role") == "user")
        if asked >= self.max_turns:
            raise ThreadTooLong(
                f"This conversation has reached its limit of {self.max_turns} questions. "
                "Start a new conversation to keep going."
            )

    async def _run_turn(
        self, question: str, thread_id: str, config: dict, correlation_id: str | None
    ) -> ChatResult:
        with tracing.trace_turn(
            session_id=thread_ref(thread_id),
            question=question,
            prompt_version=self.prompt_version,
            correlation_id=correlation_id,
        ) as span:
            state = await self.graph.ainvoke(
                {"messages": [{"role": "user", "content": question}], "steps": 0}, config
            )
            new = turn_messages(state["messages"])
            records = collect_tool_calls(new)
            final = new[-1] if new else {}
            answer = (final.get("content") or "").strip() if not final.get("tool_calls") else ""
            result = ChatResult(
                answer=answer or NO_ANSWER,
                thread_id=thread_id,
                prompt_version=self.prompt_version,
                steps=state.get("steps", 0),
                tools_used=[r.name for r in records],
                sources=build_sources(records, self.registry),
                tool_calls=records,
            )
            span.update(output={"answer": result.answer, "tools_used": result.tools_used})
        log.info(
            "chat_turn_done", thread=thread_ref(thread_id), steps=result.steps, tools_used=result.tools_used
        )
        return result
