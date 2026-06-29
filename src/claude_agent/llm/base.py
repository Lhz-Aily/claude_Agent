"""Abstract base class for LLM providers."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any

from ..core.types import Message, Role, ToolCall


class BaseLLMProvider(ABC):
    """Abstract interface for any LLM backend.

    Subclasses implement the specifics of calling each provider's API
    and normalizing the response into our internal Message format.
    """

    def __init__(self, model: str, temperature: float = 0.7, max_tokens: int = 4096) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    async def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> Message:
        """Send messages to the LLM and return the assistant response.

        Args:
            messages: The full conversation history.
            tools: Optional list of tool definitions (OpenAI function format).

        Returns:
            A Message with role=ASSISTANT. If the LLM wanted to call tools,
            `tool_calls` will be populated.
        """
        ...

    # ------------------------------------------------------------------
    # Shared helpers for subclasses
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_tool_calls_from_openai(response: Any) -> list[ToolCall]:
        """Parse tool calls from an OpenAI chat completion response."""
        tool_calls: list[ToolCall] = []
        msg = response.choices[0].message
        if msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append(
                    ToolCall(id=tc.id, name=tc.function.name, arguments=args)
                )
        return tool_calls

    @staticmethod
    def _parse_tool_calls_from_anthropic(response: Any) -> list[ToolCall]:
        """Parse tool-use blocks from an Anthropic message response."""
        tool_calls: list[ToolCall] = []
        for block in response.content:
            if block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=block.input,
                    )
                )
        return tool_calls

    @staticmethod
    def _convert_messages_for_anthropic(
        messages: list[Message],
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """Convert internal messages to Anthropic format.

        Anthropic uses a different structure: a system prompt string plus
        a list of user/assistant/tool_result message dicts.

        Returns:
            (system_prompt, api_messages)
        """
        system_prompt: str | None = None
        api_messages: list[dict[str, Any]] = []

        for msg in messages:
            if msg.role == Role.SYSTEM:
                system_prompt = msg.content or ""
                continue

            if msg.role == Role.TOOL:
                # Anthropic expects tool_result blocks inside a user message
                api_messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg.tool_call_id,
                            "content": msg.content or "",
                        }
                    ],
                })
                continue

            if msg.role == Role.ASSISTANT and msg.tool_calls:
                # Anthropic: assistant content with tool_use blocks
                content_blocks: list[dict[str, Any]] = []
                if msg.content:
                    content_blocks.append({"type": "text", "text": msg.content})
                for tc in msg.tool_calls:
                    content_blocks.append({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.arguments,
                    })
                api_messages.append({"role": "assistant", "content": content_blocks})
                continue

            # Plain user / assistant message
            role = msg.role.value
            api_messages.append({"role": role, "content": [{"type": "text", "text": msg.content or ""}]})

        return system_prompt, api_messages

    @staticmethod
    def _convert_tools_for_anthropic(
        tools: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Convert OpenAI-format tool defs to Anthropic format."""
        anthropic_tools: list[dict[str, Any]] = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool["function"]
                anthropic_tools.append({
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {
                        "type": "object",
                        "properties": {},
                    }),
                })
        return anthropic_tools
