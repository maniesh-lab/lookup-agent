"""
Chat with the agent in the terminal (no web server needed).

    python -m scripts.ask "What's your return window?"
    python -m scripts.ask            # interactive; the conversation keeps its memory

Memory here lasts while the script runs. The web API (POST /chat) saves
conversations in Postgres instead.
"""

import asyncio
import sys

from langgraph.checkpoint.memory import InMemorySaver

from app.agent import create_agent
from app.core.config import settings
from app.core.logging import configure_logging


def show(result) -> None:
    print(f"\n{result.answer}\n")
    if result.tools_used:
        print(f"  tools: {', '.join(result.tools_used)}  |  steps: {result.steps}")
    for s in result.sources:
        print(f"  source: {s['type']}: {s['name']}" + (f" ({s['section']})" if s.get("section") else ""))
    print()


async def main() -> None:
    configure_logging("WARNING")
    agent = create_agent(checkpointer=InMemorySaver())
    print(f"lookup-agent (prompt {settings.prompt_version}, provider {settings.llm_provider})")

    if len(sys.argv) > 1:
        show(await agent.ask(" ".join(sys.argv[1:])))
        return

    thread_id = None
    while True:
        try:
            question = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question in {"", "exit", "quit"}:
            break
        result = await agent.ask(question, thread_id=thread_id)
        thread_id = result.thread_id
        show(result)


if __name__ == "__main__":
    asyncio.run(main())
