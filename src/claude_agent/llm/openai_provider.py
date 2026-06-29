"""OpenAI LLM provider."""

from __future__ import annotations

from typing import Any

from openai import AsyncOpenAI

from ..core.types import Message, Role
from ..utils.logger import get_logger
from .base import BaseLLMProvider


class OpenAIProvider(BaseLLMProvider):
    """LLM provider backed by the OpenAI API (GPT-4o, GPT-4.1, etc.)."""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> None:
        super().__init__(model=model, temperature=temperature, max_tokens=max_tokens)
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._log = get_logger(__name__)

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> Message:
        """Call the OpenAI chat completion API."""
        openai_messages = [m.to_openai_format() for m in messages]

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": openai_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        self._log.debug(f"Calling {self.model} with {len(messages)} messages")

        response = await self._client.chat.completions.create(**kwargs)

        msg = response.choices[0].message
        tool_calls = self._parse_tool_calls_from_openai(response)

        self._log.debug(
            f"Response: content={msg.content!r}, tool_calls={len(tool_calls)}"
        )

        return Message(
            role=Role.ASSISTANT,
            content=msg.content,
            tool_calls=tool_calls or None,
        )
