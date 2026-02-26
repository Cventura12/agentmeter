# Run: python examples/openai_agents_example.py
# Requires: pip install agentmeter[openai-agents]

from __future__ import annotations

import asyncio
import os

from agentmeter import configure
from agentmeter.integrations import setup_openai_agents_tracing


async def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY before running this example.")

    from agents import Agent, Runner

    configure(log_path="./agentmeter_logs/openai_agents.jsonl")
    setup_openai_agents_tracing("openai-agents-demo")

    agent = Agent(
        name="assistant",
        instructions="Answer briefly and clearly.",
    )
    result = await Runner.run(agent, "Give one sentence on why tracing cost matters.")
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
