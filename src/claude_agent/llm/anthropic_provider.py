"""Anthropic Claude LLM provider."""

from __future__ import annotations

from typing import Any

from anthropic import AsyncAnthropic

from ..core.types import Message, Role
from ..utils.logger import get_logger
from .base import BaseLLMProvider


class AnthropicProvider(BaseLLMProvider):
    """LLM provider backed by the Anthropic API (Claude Sonnet, Opus, etc.)."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-6-20250514",
        api_key: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> None:
        super().__init__(model=model, temperature=temperature, max_tokens=max_tokens)
        self._client = AsyncAnthropic(api_key=api_key)
        self._log = get_logger(__name__)

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> Message:
        """Call the Anthropic Messages API."""
        system_prompt, api_messages = self._convert_messages_for_anthropic(messages)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": api_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        if system_prompt:
            kwargs["system"] = [{"type": "text", "text": system_prompt}]

        if tools:
            kwargs["tools"] = self._convert_tools_for_anthropic(tools)

        self._log.debug(f"Calling {self.model} with {len(messages)} messages")

        response = await self._client.messages.create(**kwargs)

        # Extract text and tool-use blocks
        text_parts: list[str] = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)

        text_content = "\n".join(text_parts) if text_parts else None
        tool_calls = self._parse_tool_calls_from_anthropic(response)

        self._log.debug(
            f"Response: content={text_content!r}, tool_calls={len(tool_calls)}"
        )

        return Message(
            role=Role.ASSISTANT,
            content=text_content,
            tool_calls=tool_calls or None,
        )
