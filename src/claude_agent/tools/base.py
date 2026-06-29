"""Abstract base class for tools."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseTool(ABC):
    """Minimal interface that every tool must implement.

    Each tool describes itself with a name, description, and JSON Schema
    for its parameters. The `execute` method carries out the actual work.
    """

    name: str = ""
    description: str = ""
    parameters: dict[str, Any] = {}

    @abstractmethod
    async def execute(self, **kwargs: Any) -> str:
        """Execute the tool with the given keyword arguments.

        Returns:
            A string result (shown to the LLM as the tool output).
        """
        ...

    def to_openai_schema(self) -> dict[str, Any]:
        """Convert tool to OpenAI function-calling schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolError(Exception):
    """Raised when a tool encounters an execution error."""
