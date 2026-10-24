"""The LangGraph agent with a fake LLM and fake tools (no API key needed)."""

import json

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel

from app.agent.graph import STEP_LIMIT_NOTE, build_graph
from app.agent.runner import NO_ANSWER, Agent
from app.tools.registry import Tool, ToolRegistry
from tests.conftest import FakeLLM, text_reply, tool_reply


class QueryArgs(BaseModel):
    query: str


class SqlArgs(BaseModel):
    sql: str


class FxArgs(BaseModel):
    amount: float
    from_currency: str
    to_currency: str


def fake_registry(log: list) -> ToolRegistry:
    def search_docs(query):
        log.append(("search_docs", query))
        return json.dumps(
            {
                "results": [
                    {
                        "source": "returns.md",
                        "section": "Returns Policy > Return window",
                        "text": "You can return most items within 30 days.",
                        "score": 0.9,
                    }
                ]
            }
        )

    def query_database(sql):
        log.append(("query_database", sql))
        return json.dumps({"columns": ["total"], "rows": [[173.5]], "row_count": 1, "truncated": False})

    def convert_currency(amount, from_currency, to_currency):
        log.append(("convert_currency", amount))
        return json.dumps(
            {
                "from": from_currency,
                "to": to_currency,
                "amount": amount,
                "rate": 1.4,
                "converted": round(amount * 1.4, 2),
                "rate_date": "2026-10-16",
            }
        )

    return ToolRegistry(
        [
            Tool("search_docs", "Search policies.", QueryArgs, search_docs, "document"),
            Tool("query_database", "Run SQL.", SqlArgs, query_database, "database"),
            Tool("convert_currency", "Convert money.", FxArgs, convert_currency, "api"),
        ]
    )


def make_agent(llm, registry, max_steps=4, version="v2"):
    graph = build_graph(
        llm, registry, prompt_version=version, max_steps=max_steps, checkpointer=InMemorySaver()
    )
    return Agent(graph, registry, prompt_version=version, max_steps=max_steps)


async def test_docs_question_uses_search_and_returns_sources():
    calls = []
    llm = FakeLLM(
        [
            tool_reply("search_docs", {"query": "return window"}),
            text_reply("You have 30 days to return an item.\n\nSources: returns.md"),
        ]
    )
    agent = make_agent(llm, fake_registry(calls))

    result = await agent.ask("what's your return window?")

    assert result.answer.startswith("You have 30 days")
    assert result.tools_used == ["search_docs"]
    assert result.sources == [
        {"type": "document", "name": "returns.md", "section": "Returns Policy > Return window"}
    ]
    assert calls == [("search_docs", "return window")]
    # The second LLM call saw the tool's result.
    tool_msgs = [m for m in llm.calls[1]["messages"] if m["role"] == "tool"]
    assert "30 days" in tool_msgs[0]["content"]
    # Every call starts with the system prompt and offers the tools.
    assert llm.calls[0]["messages"][0]["role"] == "system"
    assert {t["function"]["name"] for t in llm.calls[0]["tools"]} == {
        "search_docs",
        "query_database",
        "convert_currency",
    }


async def test_mixed_question_chains_database_and_api():
    calls = []
    llm = FakeLLM(
        [
            tool_reply("query_database", {"sql": "SELECT total FROM orders WHERE id = 1"}, "c1"),
            tool_reply(
                "convert_currency", {"amount": 173.5, "from_currency": "USD", "to_currency": "CAD"}, "c2"
            ),
            text_reply("173.50 USD is 242.90 CAD."),
        ]
    )
    agent = make_agent(llm, fake_registry(calls))

    result = await agent.ask("Priya's order total in CAD?")

    assert result.tools_used == ["query_database", "convert_currency"]
    assert [s["type"] for s in result.sources] == ["database", "api"]
    assert result.sources[0]["query"] == "SELECT total FROM orders WHERE id = 1"
    assert "1 USD = 1.4 CAD" in result.sources[1]["detail"]
    assert result.steps == 3


async def test_step_cap_forces_a_final_answer():
    # A model that never stops asking for tools.
    llm = FakeLLM([tool_reply("search_docs", {"query": "x"}, f"c{i}") for i in range(10)])
    agent = make_agent(llm, fake_registry([]), max_steps=2)

    result = await agent.ask("loop forever")

    last_call = llm.calls[-1]
    assert last_call["tool_choice"] == "none"
    assert last_call["messages"][-1]["content"] == STEP_LIMIT_NOTE
    assert len(llm.calls) == 3  # 2 normal steps + the forced final one
    # The scripted model still returned a tool call, so there is no text answer.
    assert result.answer == NO_ANSWER


async def test_memory_keeps_earlier_turns_in_the_same_thread():
    llm = FakeLLM([text_reply("Hi Priya!"), text_reply("Your name is Priya.")])
    agent = make_agent(llm, fake_registry([]))

    first = await agent.ask("My name is Priya.")
    second = await agent.ask("What's my name?", thread_id=first.thread_id)

    sent = [m["content"] for m in llm.calls[1]["messages"]]
    assert "My name is Priya." in sent and "Hi Priya!" in sent
    assert second.answer == "Your name is Priya."
    assert second.steps == 1  # the step counter resets every turn


async def test_new_thread_has_no_memory():
    llm = FakeLLM([text_reply("ok"), text_reply("ok")])
    agent = make_agent(llm, fake_registry([]))

    await agent.ask("My name is Priya.")
    await agent.ask("What's my name?")  # no thread_id: a new conversation

    sent = [m["content"] for m in llm.calls[1]["messages"]]
    assert "My name is Priya." not in sent


async def test_bad_tool_arguments_go_back_to_the_model():
    llm = FakeLLM([tool_reply("convert_currency", {"amount": "lots"}), text_reply("Sorry.")])
    agent = make_agent(llm, fake_registry([]))

    result = await agent.ask("convert")

    tool_msg = next(m for m in llm.calls[1]["messages"] if m["role"] == "tool")
    assert "Bad arguments for convert_currency" in tool_msg["content"]
    assert result.sources == []  # failed tools are not listed as sources


@pytest.mark.parametrize("version", ["v1", "v2"])
async def test_prompt_version_is_used(version):
    llm = FakeLLM([text_reply("ok")])
    agent = make_agent(llm, fake_registry([]), version=version)

    result = await agent.ask("hello")

    system = llm.calls[0]["messages"][0]["content"]
    assert ("Pick the right source" in system) == (version == "v2")
    assert result.prompt_version == version


# --- Stopped turns -------------------------------------------------------------

from app.agent.graph import repair_history  # noqa: E402


def call(call_id, name="search_docs"):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": "{}"}}


def test_repair_history_drops_unanswered_tool_calls():
    history = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": None, "tool_calls": [call("a")]},
        {"role": "tool", "tool_call_id": "a", "content": "result a"},
        {"role": "assistant", "content": "answer 1"},
        {"role": "user", "content": "q2"},
        # stopped here: a tool request with no result
        {"role": "assistant", "content": None, "tool_calls": [call("b")]},
        {"role": "user", "content": "q3"},
    ]
    repaired = repair_history(history)
    assert [m["role"] for m in repaired] == ["user", "assistant", "tool", "assistant", "user", "user"]
    assert repaired[1]["tool_calls"] == [call("a")]


def test_repair_history_keeps_text_and_drops_orphan_results():
    history = [
        {"role": "assistant", "content": "Let me check.", "tool_calls": [call("x")]},
        {"role": "tool", "tool_call_id": "y", "content": "orphan"},
        {"role": "user", "content": "q"},
    ]
    assert repair_history(history) == [
        {"role": "assistant", "content": "Let me check."},
        {"role": "user", "content": "q"},
    ]


async def test_conversation_continues_after_a_stopped_turn():
    import asyncio
    import time

    def slow_search(query):
        time.sleep(0.5)  # still running when the turn is stopped
        return "{}"

    registry = ToolRegistry([Tool("search_docs", "Search.", QueryArgs, slow_search, "document")])
    llm = FakeLLM([tool_reply("search_docs", {"query": "x"}, "c1"), text_reply("Second answer.")])
    agent = make_agent(llm, registry)

    thread = "stop-me"
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(agent.ask("first question", thread_id=thread), timeout=0.2)

    saved = (await agent.graph.aget_state({"configurable": {"thread_id": thread}})).values["messages"]
    assert saved[-1].get("tool_calls")  # the stop left a request with no result in memory

    result = await agent.ask("second question", thread_id=thread)

    assert result.answer == "Second answer."
    sent = llm.calls[-1]["messages"]
    assert not any(m.get("tool_calls") for m in sent)  # the unanswered request was dropped
    assert [m["content"] for m in sent if m["role"] == "user"] == ["first question", "second question"]
