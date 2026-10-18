"""The LLM client, built on Steadfast (github.com/maniesh-lab/steadfast).

Steadfast gives every call retries with backoff, a per-provider rate
limit, a fallback provider, token counting and correlation-id logging,
and it speaks one message/tool format for Groq, OpenAI and Anthropic.

The agent only depends on the small `LLM` protocol below, so tests can
pass in a fake that returns scripted replies without any API key.
"""

from typing import Any, Protocol

from steadfast import LLMResponse, SteadfastClient

from app.core import tracing
from app.core.config import settings


class LLM(Protocol):
    model: str | None

    async def chat(
        self, messages: list[dict], *, tools: list[dict] | None = None, tool_choice: Any = None
    ) -> LLMResponse: ...


class SteadfastLLM:
    def __init__(self, client: SteadfastClient, provider: str, fallbacks: list[str], model: str):
        self.client = client
        self.provider = provider
        self.fallbacks = fallbacks
        self.model = model

    async def chat(self, messages, *, tools=None, tool_choice=None) -> LLMResponse:
        with tracing.observe_llm(model=self.model, messages=messages) as obs:
            response = await self.client.chat(
                self.provider,
                messages,
                tools=tools,
                tool_choice=tool_choice,
                fallbacks=self.fallbacks or None,
                temperature=settings.llm_temperature,
            )
            obs.update(
                output=response.to_message(),
                model=response.model,
                usage_details={
                    "input": response.usage.input_tokens,
                    "output": response.usage.output_tokens,
                },
            )
        return response


def build_llm() -> SteadfastLLM:
    """Register the configured providers. The main one is LLM_PROVIDER; the
    other one (if its key is set) becomes the fallback."""
    client = SteadfastClient()
    registered: list[str] = []

    if settings.groq_api_key:
        from steadfast.providers.groq_provider import GroqProvider

        client.register(
            GroqProvider(api_key=settings.groq_api_key, model=settings.llm_model),
            rate=settings.llm_calls_per_minute,
            period=60,
        )
        registered.append("groq")
    if settings.openai_api_key:
        from steadfast.providers.openai_provider import OpenAIProvider

        client.register(
            OpenAIProvider(api_key=settings.openai_api_key, model=settings.openai_model),
            rate=settings.llm_calls_per_minute,
            period=60,
        )
        registered.append("openai")

    if settings.llm_provider not in registered:
        raise RuntimeError(
            f"LLM_PROVIDER is {settings.llm_provider!r} but its API key isn't set. "
            "Add GROQ_API_KEY or OPENAI_API_KEY to .env."
        )
    fallbacks = [name for name in registered if name != settings.llm_provider]
    model = settings.llm_model if settings.llm_provider == "groq" else settings.openai_model
    return SteadfastLLM(client, settings.llm_provider, fallbacks, model)
