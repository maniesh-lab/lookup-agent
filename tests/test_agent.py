"""The LangGraph agent with a fake LLM and fake tools (no API key needed)."""

import asyncio
import json

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel

from app.agent import create_agent
from app.agent.graph import STEP_LIMIT_NOTE, build_graph
from app.agent.runner import NO_ANSWER, Agent, ThreadTooLong, thread_ref
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


def fake_registry(log: list, search_text: str = "You can return most items within 30 days.") -> ToolRegistry:
    def search_docs(query):
        log.append(("search_docs", query))
        return json.dumps(
            {
                "results": [
                    {
                        "source": "returns.md",
                        "section": "Returns Policy > Return window",
                        "text": search_text,
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


@pytest.mark.parametrize("version", ["v1", "v2", "v3"])
async def test_prompt_version_is_used(version):
    llm = FakeLLM([text_reply("ok")])
    agent = make_agent(llm, fake_registry([]), version=version)

    result = await agent.ask("hello")

    system = llm.calls[0]["messages"][0]["content"]
    assert ("Pick the right source" in system) == (version != "v1")
    assert ("how the shop works" in system) == (version == "v3")
    assert result.prompt_version == version


def test_unknown_prompt_version_fails_at_startup():
    with pytest.raises(ValueError, match="Unknown prompt version 'v9'"):
        create_agent(llm=FakeLLM([]), registry=fake_registry([]), prompt_version="v9")


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


# --- Trimming long conversations ------------------------------------------------

from app.agent.graph import (  # noqa: E402
    OLD_RESULT_PLACEHOLDER,
    TRIM_RESULTS_OVER_CHARS,
    trim_old_tool_results,
)


def test_trim_replaces_only_earlier_turns_results():
    long_result = "x" * 500
    history = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": None, "tool_calls": [call("a")]},
        {"role": "tool", "tool_call_id": "a", "content": long_result},
        {"role": "assistant", "content": "answer 1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": None, "tool_calls": [call("b")]},
        {"role": "tool", "tool_call_id": "b", "content": long_result},
    ]
    trimmed = trim_old_tool_results(history)

    assert trimmed[2] == {"role": "tool", "tool_call_id": "a", "content": OLD_RESULT_PLACEHOLDER}
    assert trimmed[6]["content"] == long_result  # the current turn is untouched
    assert trimmed[1]["tool_calls"] == [call("a")]  # the request keeps its result
    assert [m["role"] for m in trimmed] == [m["role"] for m in history]
    assert history[2]["content"] == long_result  # the input list is not changed


def test_trim_keeps_short_results_and_single_turns():
    short = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": None, "tool_calls": [call("a")]},
        {"role": "tool", "tool_call_id": "a", "content": "{}"},
        {"role": "assistant", "content": "answer 1"},
        {"role": "user", "content": "q2"},
    ]
    short[2]["content"] = "r" * TRIM_RESULTS_OVER_CHARS
    assert trim_old_tool_results(short) == short  # too short to be worth trimming
    assert trim_old_tool_results(short[:3]) == short[:3]  # one turn: nothing is earlier
    assert trim_old_tool_results([]) == []


async def test_follow_up_sends_trimmed_history_but_saves_the_full_one():
    calls = []
    llm = FakeLLM(
        [
            tool_reply("search_docs", {"query": "return window"}, "t1"),
            text_reply("You have 30 days to return an item."),
            tool_reply("search_docs", {"query": "return window premium"}, "t2"),
            text_reply("Premium members also get 30 days."),
        ]
    )
    long_text = "You can return most items within 30 days. " + "More policy details. " * 20
    agent = make_agent(llm, fake_registry(calls, search_text=long_text))

    first = await agent.ask("what's your return window?")
    second = await agent.ask("and for premium members?", thread_id=first.thread_id)

    # Turn 2, first LLM call: turn 1's result is a placeholder, its answer is kept.
    sent = llm.calls[2]["messages"]
    old_result = next(m for m in sent if m.get("tool_call_id") == "t1")
    assert old_result["content"] == OLD_RESULT_PLACEHOLDER
    assert "You have 30 days to return an item." in [m["content"] for m in sent]
    assert any(c["id"] == "t1" for m in sent for c in m.get("tool_calls") or [])

    # Turn 2, second LLM call: its own fresh result is sent in full.
    sent = llm.calls[3]["messages"]
    assert "30 days" in next(m for m in sent if m.get("tool_call_id") == "t2")["content"]
    assert next(m for m in sent if m.get("tool_call_id") == "t1")["content"] == OLD_RESULT_PLACEHOLDER

    # The follow-up still works, and the saved conversation keeps the full result.
    assert second.answer == "Premium members also get 30 days."
    assert second.sources == [
        {"type": "document", "name": "returns.md", "section": "Returns Policy > Return window"}
    ]
    config = {"configurable": {"thread_id": first.thread_id}}
    saved = (await agent.graph.aget_state(config)).values["messages"]
    assert "30 days" in next(m for m in saved if m.get("tool_call_id") == "t1")["content"]


# --- One turn at a time per thread, and a cap on questions per thread ------------


async def test_a_thread_refuses_questions_past_its_limit():
    llm = FakeLLM([text_reply("one"), text_reply("two"), text_reply("fresh")])
    registry = fake_registry([])
    graph = build_graph(llm, registry, prompt_version="v2", max_steps=4, checkpointer=InMemorySaver())
    agent = Agent(graph, registry, prompt_version="v2", max_steps=4, max_turns=2)

    first = await agent.ask("q1")
    await agent.ask("q2", thread_id=first.thread_id)
    with pytest.raises(ThreadTooLong, match="limit of 2"):
        await agent.ask("q3", thread_id=first.thread_id)
    assert len(llm.calls) == 2  # the refused question never reached the LLM
    assert (await agent.ask("a new conversation")).answer == "fresh"


class SlowLLM:
    """Answers after a short wait and records how many calls overlapped."""

    model = "slow"

    def __init__(self):
        self.active = self.most_active = 0

    async def chat(self, messages, *, tools=None, tool_choice=None):
        self.active += 1
        self.most_active = max(self.most_active, self.active)
        await asyncio.sleep(0.05)
        self.active -= 1
        return text_reply(f"answer to {messages[-1]['content']}")


async def test_questions_on_one_thread_run_one_after_the_other():
    llm = SlowLLM()
    agent = make_agent(llm, fake_registry([]))
    await asyncio.gather(agent.ask("a", thread_id="t"), agent.ask("b", thread_id="t"))
    assert llm.most_active == 1
    state = await agent.graph.aget_state({"configurable": {"thread_id": "t"}})
    assert [m["role"] for m in state.values["messages"]] == ["user", "assistant", "user", "assistant"]
    assert len(agent.thread_locks) == 0  # nothing left behind


async def test_different_threads_still_run_side_by_side():
    llm = SlowLLM()
    agent = make_agent(llm, fake_registry([]))
    await asyncio.gather(agent.ask("a", thread_id="t1"), agent.ask("b", thread_id="t2"))
    assert llm.most_active == 2


def test_thread_ids_are_kept_out_of_logs():
    assert thread_ref(None) is None
    ref = thread_ref("1b4e28ba-2fa1-11d2-883f-0016d3cca427")
    assert len(ref) == 16 and "1b4e28ba" not in ref
    assert ref == thread_ref("1b4e28ba-2fa1-11d2-883f-0016d3cca427")
