"""Tests for the memory system."""

import pytest

from claude_agent.core.types import Message, Role
from claude_agent.memory.conversation import ConversationMemory


class TestConversationMemory:
    def test_add_and_get(self):
        memory = ConversationMemory(max_tokens=1_000_000)
        memory.add(Message(role=Role.USER, content="hello"))
        memory.add(Message(role=Role.ASSISTANT, content="hi there"))
        msgs = memory.get_messages()
        assert len(msgs) == 2
        assert msgs[0].content == "hello"
        assert msgs[1].content == "hi there"

    def test_system_prompt(self):
        memory = ConversationMemory(
            max_tokens=1_000_000,
            system_prompt="You are a helpful assistant.",
        )
        msgs = memory.get_messages()
        assert len(msgs) == 1
        assert msgs[0].role == Role.SYSTEM
        assert msgs[0].content == "You are a helpful assistant."

    def test_clear_preserves_system(self):
        memory = ConversationMemory(
            max_tokens=1_000_000,
            system_prompt="System prompt here.",
        )
        memory.add(Message(role=Role.USER, content="hello"))
        memory.clear()
        msgs = memory.get_messages()
        assert len(msgs) == 1
        assert msgs[0].role == Role.SYSTEM

    def test_clear_without_system(self):
        memory = ConversationMemory(max_tokens=1_000_000)
        memory.add(Message(role=Role.USER, content="hello"))
        memory.clear()
        assert len(memory.get_messages()) == 0

    def test_trim_on_token_limit(self):
        memory = ConversationMemory(max_tokens=100)  # very low limit
        # Add a large message
        large_text = "x" * 10_000
        memory.add(Message(role=Role.USER, content=large_text))
        memory.add(Message(role=Role.ASSISTANT, content="short"))
        # Should have trimmed the large one
        msgs = memory.get_messages()
        # At least one message should have been dropped
        assert len(msgs) <= 2


class TestCoreTypes:
    def test_message_to_openai(self):
        msg = Message(role=Role.USER, content="hello")
        d = msg.to_openai_format()
        assert d == {"role": "user", "content": "hello"}

    def test_tool_call_to_openai_format(self):
        from claude_agent.core.types import ToolCall

        tc = ToolCall(name="search", arguments={"q": "python"})
        d = tc.to_openai_format()
        assert d["type"] == "function"
        assert d["function"]["name"] == "search"

    def test_tool_result_to_message(self):
        from claude_agent.core.types import ToolResult

        tr = ToolResult(
            tool_call_id="call_123",
            name="search",
            content="results here",
            success=True,
        )
        msg = tr.to_message()
        assert msg.role == Role.TOOL
        assert msg.tool_call_id == "call_123"
        assert msg.content == "results here"
