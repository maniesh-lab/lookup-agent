"""The Steadfast-based LLM client, tracing switch and agent wiring (no network)."""

import pytest
from steadfast import LLMResponse

from app.agent import create_agent
from app.core import tracing
from app.core.config import settings
from app.services import llm as llm_module
from app.services.llm import SteadfastLLM, build_llm
from tests.conftest import FakeLLM, text_reply


class FakeSteadfastClient:
    def __init__(self):
        self.calls = []

    async def chat(self, provider, messages, **kwargs):
        self.calls.append((provider, messages, kwargs))
        return LLMResponse(text="hi", finish_reason="stop", model="m", provider=provider)


async def test_steadfast_llm_passes_tools_and_fallbacks():
    client = FakeSteadfastClient()
    llm = SteadfastLLM(client, "groq", ["openai"], "m")

    response = await llm.chat([{"role": "user", "content": "x"}], tools=[{"t": 1}], tool_choice="none")

    provider, _messages, kwargs = client.calls[0]
    assert response.text == "hi"
    assert provider == "groq"
    assert kwargs["tools"] == [{"t": 1}] and kwargs["tool_choice"] == "none"
    assert kwargs["fallbacks"] == ["openai"]
    assert kwargs["temperature"] == settings.llm_temperature


def test_build_llm_registers_main_provider_and_fallback(monkeypatch):
    monkeypatch.setattr(settings, "groq_api_key", "gsk_test")
    monkeypatch.setattr(settings, "openai_api_key", "sk_test")
    monkeypatch.setattr(settings, "llm_provider", "groq")

    llm = build_llm()

    assert llm.provider == "groq"
    assert llm.fallbacks == ["openai"]
    assert llm.model == settings.llm_model


def test_build_llm_without_a_key_explains_what_to_do(monkeypatch):
    monkeypatch.setattr(settings, "groq_api_key", None)
    monkeypatch.setattr(settings, "openai_api_key", None)
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        build_llm()


def test_tracing_is_off_without_keys():
    tracing._client.cache_clear()
    assert tracing.enabled() is False
    with tracing.trace_turn(thread_id="t", question="q", prompt_version="v2", correlation_id=None) as span:
        span.update(output="ignored")
    with tracing.observe_tool(name="search_docs", arguments={}) as obs:
        obs.update(output="ignored")
    tracing.flush()


async def test_create_agent_wires_the_pieces(monkeypatch):
    from langgraph.checkpoint.memory import InMemorySaver

    from app.tools.registry import ToolRegistry

    agent = create_agent(
        checkpointer=InMemorySaver(),
        llm=FakeLLM([text_reply("hello")]),
        registry=ToolRegistry([]),
        prompt_version="v1",
    )
    result = await agent.ask("hi")
    assert result.answer == "hello" and result.prompt_version == "v1"
    assert llm_module  # module imports cleanly
