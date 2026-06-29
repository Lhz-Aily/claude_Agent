#!/usr/bin/env python3
"""Example: Creating and registering a custom tool."""

import asyncio
from typing import Any

from claude_agent import Agent, Settings, create_llm_provider
from claude_agent.memory.conversation import ConversationMemory
from claude_agent.tools.base import BaseTool, ToolError
from claude_agent.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Step 1: Define a custom tool
# ---------------------------------------------------------------------------

class WeatherTool(BaseTool):
    """A mock weather tool — replace with a real API call."""

    name = "get_weather"
    description = "Get the current weather for a city. Returns temperature and conditions."

    # JSON Schema for the tool's parameters (used by the LLM to generate arguments)
    parameters = {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "City name, e.g. 'Beijing' or 'San Francisco'.",
            },
            "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "Temperature unit (default: celsius).",
                "default": "celsius",
            },
        },
        "required": ["city"],
    }

    async def execute(self, city: str, unit: str = "celsius") -> str:
        # In a real tool you'd call a weather API here
        # For this example we return mock data
        weather_data: dict[str, dict[str, Any]] = {
            "beijing": {"temp": 28, "condition": "Sunny", "humidity": 45},
            "san francisco": {"temp": 18, "condition": "Foggy", "humidity": 80},
            "tokyo": {"temp": 22, "condition": "Cloudy", "humidity": 65},
        }

        key = city.lower()
        if key not in weather_data:
            raise ToolError(f"No weather data for city: {city}")

        data = weather_data[key]
        temp = data["temp"]
        if unit == "fahrenheit":
            temp = temp * 9 / 5 + 32

        return (
            f"Weather in {city.title()}: {data['condition']}, "
            f"{temp:.1f}°{'F' if unit == 'fahrenheit' else 'C'}, "
            f"humidity {data['humidity']}%"
        )


# ---------------------------------------------------------------------------
# Step 2: Use it
# ---------------------------------------------------------------------------

async def main() -> None:
    settings = Settings()  # type: ignore[call-arg]
    llm = create_llm_provider(settings)

    # Register both built-in and custom tools
    tools = ToolRegistry()
    tools.discover_builtin()
    tools.register(WeatherTool())  # <-- register custom tool
    print(f"Registered tools: {tools.list_names()}")

    memory = ConversationMemory(
        max_tokens=settings.conversation_max_tokens,
        system_prompt=(
            "You are a helpful assistant. Use the get_weather tool "
            "when the user asks about weather."
        ),
    )

    agent = Agent(llm=llm, tools=tools, memory=memory, settings=settings)

    # Ask about weather — the Agent should automatically call get_weather
    result = await agent.run("What's the weather like in Tokyo today?")
    print(f"\n{result}")


if __name__ == "__main__":
    asyncio.run(main())
