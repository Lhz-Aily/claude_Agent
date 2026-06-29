"""Tests for the Agent core loop (with a mock LLM)."""

from __future__ import annotations

from typing import Any

import pytest

from claude_agent.core.agent import Agent
from claude_agent.core.types import AgentState, Message, Role
from claude_agent.llm.base import BaseLLMProvider
from claude_agent.memory.conversation import ConversationMemory
from claude_agent.tools.base import BaseTool
from claude_agent.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Mock LLM that returns controlled responses
# ---------------------------------------------------------------------------

class MockLLM(BaseLLMProvider):
    """Returns pre-programmed responses for testing."""

    def __init__(self, responses: list[Message] | None = None):
        super().__init__(model="mock", temperature=0, max_tokens=100)
        self.responses = responses or []
        self.call_count = 0
        self.last_messages: list[Message] = []

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> Message:
        self.last_messages = messages
        if self.call_count < len(self.responses):
            resp = self.responses[self.call_count]
            self.call_count += 1
            return resp
        # Default: return a text response
        self.call_count += 1
        return Message(role=Role.ASSISTANT, content="Mock final answer.")


# ---------------------------------------------------------------------------
# Simple echo tool for testing
# ---------------------------------------------------------------------------

class EchoTool(BaseTool):
    name = "echo"
    description = "Echo back the input."
    parameters = {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "Text to echo."},
        },
        "required": ["text"],
    }

    async def execute(self, text: str) -> str:
        return f"ECHO: {text}"


# ---------------------------------------------------------------------------
# Settings stub
# ---------------------------------------------------------------------------

class StubSettings:
    agent_max_iterations: int = 5
    agent_temperature: float = 0.0
    agent_max_tokens: int = 100
    conversation_max_tokens: int = 1_000_000
    system_prompt: str = "You are a test agent."
    log_level: str = "WARNING"
    llm_provider: str = "mock"
    openai_model: str = "mock"
    openai_api_key: str = ""
    openai_base_url: str = ""
    anthropic_model: str = "mock"
    anthropic_api_key: str = ""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAgentWithMockLLM:
    @pytest.mark.asyncio
    async def test_simple_text_response(self):
        """Agent returns the LLM's answer directly when no tool calls."""
        mock_llm = MockLLM(responses=[
            Message(role=Role.ASSISTANT, content="Hello! How can I help?")
        ])
        tools = ToolRegistry()
        memory = ConversationMemory(system_prompt="You are helpful.")
        settings = StubSettings()

        agent = Agent(llm=mock_llm, tools=tools, memory=memory, settings=settings)
        result = await agent.run("Hi!")

        assert result == "Hello! How can I help?"
        assert agent.state == AgentState.DONE
        assert mock_llm.call_count == 1

    @pytest.mark.asyncio
    async def test_single_tool_call_then_answer(self):
        """Agent calls a tool, observes the result, then gives a final answer."""
        from claude_agent.core.types import ToolCall

        mock_llm = MockLLM(responses=[
            # First call: LLM wants to use echo
            Message(
                role=Role.ASSISTANT,
                content="Let me echo that.",
                tool_calls=[
                    ToolCall(name="echo", arguments={"text": "hello world"})
                ],
            ),
            # Second call: LLM sees result and gives final answer
            Message(
                role=Role.ASSISTANT,
                content="The echo returned: ECHO: hello world. That worked!",
            ),
        ])

        tools = ToolRegistry()
        tools.register(EchoTool())
        memory = ConversationMemory(system_prompt="You are helpful.")
        settings = StubSettings()

        agent = Agent(llm=mock_llm, tools=tools, memory=memory, settings=settings)
        result = await agent.run("Echo hello world")

        assert "ECHO: hello world" in result
        assert agent.state == AgentState.DONE
        assert mock_llm.call_count == 2

    @pytest.mark.asyncio
    async def test_max_iterations_reached(self):
        """Agent stops after max_iterations and forces a final answer."""
        from claude_agent.core.types import ToolCall

        # LLM keeps asking for tool calls forever
        infinite_responses = []
        for _ in range(10):
            infinite_responses.append(
                Message(
                    role=Role.ASSISTANT,
                    content="Let me check...",
                    tool_calls=[
                        ToolCall(name="echo", arguments={"text": "again"})
                    ],
                )
            )

        mock_llm = MockLLM(responses=infinite_responses)
        tools = ToolRegistry()
        tools.register(EchoTool())
        memory = ConversationMemory(system_prompt="You are helpful.")
        settings = StubSettings()
        settings.agent_max_iterations = 3  # small limit

        agent = Agent(llm=mock_llm, tools=tools, memory=memory, settings=settings)
        result = await agent.run("test")

        # Should terminate after 3 iterations + 1 force-final call
        assert agent.state == AgentState.DONE
        assert mock_llm.call_count >= 3

    @pytest.mark.asyncio
    async def test_reset_clears_memory(self):
        """reset() clears conversation and sets state to IDLE."""
        mock_llm = MockLLM(responses=[
            Message(role=Role.ASSISTANT, content="First answer.")
        ])
        tools = ToolRegistry()
        memory = ConversationMemory(system_prompt="You are helpful.")
        settings = StubSettings()

        agent = Agent(llm=mock_llm, tools=tools, memory=memory, settings=settings)
        await agent.run("First question")

        agent.reset()
        assert agent.state == AgentState.IDLE
        msgs = memory.get_messages()
        assert len(msgs) == 1  # only system prompt remains
