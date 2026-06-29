#!/usr/bin/env python3
"""Basic usage examples for Claude Agent."""

import asyncio

from claude_agent import Agent, Settings, create_llm_provider
from claude_agent.memory.conversation import ConversationMemory
from claude_agent.tools.registry import ToolRegistry


async def main() -> None:
    # 1. Load settings (from .env or environment)
    settings = Settings()  # type: ignore[call-arg]
    print(f"Provider: {settings.llm_provider}")
    print(f"Model: {settings.openai_model}")

    # 2. Create LLM provider
    llm = create_llm_provider(settings)

    # 3. Set up tools
    tools = ToolRegistry()
    tools.discover_builtin()
    print(f"Tools: {tools.list_names()}")

    # 4. Set up memory
    memory = ConversationMemory(
        max_tokens=settings.conversation_max_tokens,
        system_prompt=settings.system_prompt,
    )

    # 5. Create agent
    agent = Agent(
        llm=llm,
        tools=tools,
        memory=memory,
        settings=settings,
    )

    # 6. Run some tasks
    tasks = [
        "What is 2 + 2?",
        "Write a Python function that calculates Fibonacci numbers.",
        "Search for the latest Python version and tell me what it is.",
    ]

    for task in tasks:
        print(f"\n{'=' * 60}")
        print(f"Task: {task}")
        print("=" * 60)
        try:
            result = await agent.run(task)
            print(f"\nResult:\n{result}")
        except Exception as e:
            print(f"Error: {e}")

        # Reset for next task
        agent.reset()


if __name__ == "__main__":
    asyncio.run(main())
