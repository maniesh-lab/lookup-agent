"""The LangGraph agent.

The graph has two nodes and loops between them:

    START -> agent --(model asked for tools?)--> tools -> agent -> ...
                   +--(no tools, plain answer)--> END

- agent: sends the conversation to the LLM with the tool list.
- tools: runs every tool the LLM asked for and adds the results.

A step counter caps the loop: once it reaches max_steps, the agent is
told to answer with what it has and tools are switched off.

Before each LLM call, tool results from earlier turns are swapped for a
short placeholder (see trim_old_tool_results), so a long conversation
doesn't resend every old search result and SQL table with each question.

The state is a list of messages in OpenAI's format (plain dicts), the
format Steadfast uses for every provider. With a checkpointer, LangGraph
saves the state after every step under a thread_id, which is how the
agent remembers earlier turns of the same conversation.
"""

import asyncio
import json
import operator
from typing import Annotated, TypedDict

import structlog
from langgraph.graph import END, START, StateGraph

from app.agent.prompts import load_prompt
from app.core import tracing
from app.services.llm import LLM
from app.tools.registry import ToolRegistry

log = structlog.get_logger(__name__)

STEP_LIMIT_NOTE = (
    "(Step limit reached. Do not call any more tools. Answer now with the information you "
    "already have, and say clearly what you couldn't find.)"
)


def repair_history(messages: list[dict]) -> list[dict]:
    """Drop half-finished tool steps before sending the conversation to the LLM.

    If a turn is stopped partway (the user pressed Stop, the server restarted,
    a provider timed out), the saved conversation can end up with a tool
    request that never got its result, or a result whose request is gone.
    Providers reject a conversation like that, so every later question in
    the thread would fail. This keeps only complete request/result pairs.
    The saved history itself is left as it is; only what is sent is cleaned.
    """
    answered = {m.get("tool_call_id") for m in messages if m.get("role") == "tool"}
    repaired: list[dict] = []
    kept_calls: set[str] = set()
    for message in messages:
        if message.get("role") == "assistant" and message.get("tool_calls"):
            calls = [c for c in message["tool_calls"] if c.get("id") in answered]
            kept_calls.update(c["id"] for c in calls)
            if calls:
                repaired.append({**message, "tool_calls": calls})
            elif message.get("content"):
                repaired.append({"role": "assistant", "content": message["content"]})
        elif message.get("role") == "tool":
            if message.get("tool_call_id") in kept_calls:
                repaired.append(message)
        else:
            repaired.append(message)
    return repaired


OLD_RESULT_PLACEHOLDER = (
    "(Result from an earlier question, left out to save space. "
    "Call the tool again if you need these details.)"
)
# Results this short (a one-row SQL answer, an exchange rate) cost a few dozen
# tokens and are handy for follow-ups, so they are always sent in full.
TRIM_RESULTS_OVER_CHARS = 300


def trim_old_tool_results(messages: list[dict]) -> list[dict]:
    """Swap tool results from earlier turns for a short placeholder.

    Every message in a thread is sent again with each new question, and
    tool results are by far the biggest part (search hits, SQL tables).
    Left alone, a long conversation grows until it hits the model's token
    limit and uses up the free-tier token budget. The answers the agent
    gave in earlier turns stay, so follow-up questions still have the
    facts; only the raw results behind them are left out. The current turn
    (everything after the last user message) is sent untouched, because
    the agent is still working with those results. Short results are kept
    too (see TRIM_RESULTS_OVER_CHARS).

    Results are replaced, never removed, so every tool request keeps its
    matching result (the same rule as repair_history: providers reject a
    request without a result). The saved history is left as it is; only
    what is sent is trimmed.
    """
    last_user = max((i for i, m in enumerate(messages) if m.get("role") == "user"), default=-1)
    trimmed: list[dict] = []
    for i, message in enumerate(messages):
        content = message.get("content")
        if (
            i < last_user
            and message.get("role") == "tool"
            and isinstance(content, str)
            and len(content) > TRIM_RESULTS_OVER_CHARS
        ):
            trimmed.append({**message, "content": OLD_RESULT_PLACEHOLDER})
        else:
            trimmed.append(message)
    return trimmed


class AgentState(TypedDict):
    # operator.add means "append": each node returns only its new messages.
    messages: Annotated[list[dict], operator.add]
    # No reducer: a node's value replaces the old one. Reset to 0 every turn.
    steps: int


def build_graph(
    llm: LLM,
    registry: ToolRegistry,
    *,
    prompt_version: str,
    max_steps: int,
    checkpointer=None,
):
    tool_specs = registry.specs() or None  # providers reject an empty tool list

    async def agent_node(state: AgentState) -> dict:
        steps = state.get("steps", 0)
        history = trim_old_tool_results(repair_history(state["messages"]))
        messages = [{"role": "system", "content": load_prompt(prompt_version)}, *history]

        if steps >= max_steps:
            messages.append({"role": "user", "content": STEP_LIMIT_NOTE})
            response = await llm.chat(messages, tools=tool_specs, tool_choice="none")
        else:
            response = await llm.chat(messages, tools=tool_specs)

        log.info(
            "agent_step",
            step=steps + 1,
            tool_calls=[c.name for c in response.tool_calls],
            finish_reason=response.finish_reason,
        )
        return {"messages": [response.to_message()], "steps": steps + 1}

    async def tools_node(state: AgentState) -> dict:
        last = state["messages"][-1]
        results = []
        for call in last.get("tool_calls", []):
            name = call["function"]["name"]
            raw = call["function"].get("arguments") or "{}"
            try:
                args = json.loads(raw)
            except json.JSONDecodeError:
                args = None
            with tracing.observe_tool(name=name, arguments=args) as obs:
                # Tools are ordinary (blocking) functions; run them in a thread
                # so the web server stays responsive.
                output = await asyncio.to_thread(registry.run, name, args)
                obs.update(output=output)
            log.info("tool_called", tool=name, arguments=args, result_chars=len(output))
            results.append({"role": "tool", "tool_call_id": call["id"], "content": output})
        return {"messages": results}

    def route_after_agent(state: AgentState) -> str:
        last = state["messages"][-1]
        if last.get("tool_calls") and state.get("steps", 0) <= max_steps:
            return "tools"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route_after_agent, ["tools", END])
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=checkpointer)
