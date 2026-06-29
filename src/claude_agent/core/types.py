"""Core type definitions for the Agent framework."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class Role(str, Enum):
    """Message role in a conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Message(BaseModel):
    """A single message in the conversation."""

    role: Role
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None  # for tool result messages
    name: str | None = None  # tool name for tool result messages
    id: str = Field(default_factory=lambda: f"msg_{uuid.uuid4().hex[:12]}")

    @field_validator("content", mode="before")
    @classmethod
    def coerce_null_to_empty(cls, v: str | None) -> str | None:
        """Anthropic returns None content on tool_use messages — keep as None."""
        return v

    def to_openai_format(self) -> dict[str, Any]:
        """Convert to OpenAI API message format."""
        msg: dict[str, Any] = {"role": self.role.value}
        if self.content is not None:
            msg["content"] = self.content
        if self.tool_calls:
            msg["tool_calls"] = [tc.to_openai_format() for tc in self.tool_calls]
        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id
        if self.name:
            msg["name"] = self.name
        return msg

    def to_anthropic_format(self) -> dict[str, Any]:
        """Convert to Anthropic API message format."""
        msg: dict[str, Any] = {"role": self.role.value}
        if self.content is not None:
            msg["content"] = self.content
        return msg


class ToolCall(BaseModel):
    """Represents a request from the LLM to call a tool."""

    id: str = Field(default_factory=lambda: f"call_{uuid.uuid4().hex[:12]}")
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)

    def to_openai_format(self) -> dict[str, Any]:
        """Convert to OpenAI tool_call format."""
        import json

        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments, ensure_ascii=False),
            },
        }


class ToolResult(BaseModel):
    """The result of executing a tool call."""

    tool_call_id: str
    name: str
    content: str
    success: bool = True
    error: str | None = None

    def to_message(self) -> Message:
        """Convert this result back into a tool-response Message."""
        return Message(
            role=Role.TOOL,
            content=self.content,
            tool_call_id=self.tool_call_id,
            name=self.name,
        )


class AgentStep(BaseModel):
    """A single step in the Agent's reasoning loop."""

    step_number: int
    thought: str | None = None  # LLM's reasoning before acting
    tool_calls: list[ToolCall] | None = None
    tool_results: list[ToolResult] | None = None
    final_answer: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def is_final(self) -> bool:
        """Whether this step ends the loop (final answer given)."""
        return self.final_answer is not None


class AgentState(str, Enum):
    """Current state of the Agent."""

    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    OBSERVING = "observing"
    DONE = "done"
    ERROR = "error"
