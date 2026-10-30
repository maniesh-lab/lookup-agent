from app.agent.graph import build_graph
from app.agent.prompts import load_prompt
from app.agent.runner import Agent, ChatResult
from app.core.config import settings


def create_agent(*, checkpointer=None, llm=None, registry=None, prompt_version=None) -> Agent:
    """Wire the real pieces together (tests pass fakes for llm and registry)."""
    if llm is None:
        from app.services.llm import build_llm

        llm = build_llm()
    if registry is None:
        from app.tools.registry import default_registry

        registry = default_registry()
    version = prompt_version or settings.prompt_version
    load_prompt(version)  # fail at startup, not at the first question, if the version is wrong
    graph = build_graph(
        llm,
        registry,
        prompt_version=version,
        max_steps=settings.max_agent_steps,
        checkpointer=checkpointer,
    )
    return Agent(graph, registry, prompt_version=version, max_steps=settings.max_agent_steps, llm=llm)


__all__ = ["Agent", "ChatResult", "create_agent"]
