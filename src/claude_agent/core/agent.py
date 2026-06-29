"""Core Agent — ReAct loop implementation."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, AsyncIterator

from ..core.types import AgentState, Message, Role, ToolCall
from ..utils.logger import get_logger

if TYPE_CHECKING:
    from ..config.settings import Settings
    from ..llm.base import BaseLLMProvider
    from ..memory.conversation import ConversationMemory
    from ..tools.registry import ToolRegistry


class Agent:
    """A ReAct-pattern AI Agent.

    Core loop:
    1. Receive user input
    2. Send to LLM with available tools
    3. If LLM returns tool calls → execute them → feed results back
    4. Repeat until final answer or max iterations reached
    """

    def __init__(
        self,
        llm: BaseLLMProvider,
        tools: ToolRegistry,
        memory: ConversationMemory,
        settings: Settings,
    ) -> None:
        self._llm = llm
        self._tools = tools
        self._memory = memory
        self._settings = settings
        self._log = get_logger(__name__)
        self._state = AgentState.IDLE

        # Lazy imports to avoid circular dependency
        self._executor: Any = None

    @property
    def state(self) -> AgentState:
        return self._state

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self, user_input: str) -> str:
        """Run the agent on a single user input and return the final answer."""
        from ..core.executor import Executor

        self._executor = Executor(self._tools)
        self._state = AgentState.THINKING

        # Add user message to memory
        self._memory.add(Message(role=Role.USER, content=user_input))

        iteration = 0
        max_iter = self._settings.agent_max_iterations

        while iteration < max_iter:
            iteration += 1
            self._log.info(f"🔄 Iteration {iteration}/{max_iter}")

            # --- Step 1: Call LLM ---
            self._state = AgentState.THINKING
            response = await self._call_llm()

            # --- Step 2: No tool calls → final answer ---
            if not response.tool_calls:
                self._state = AgentState.DONE
                answer = response.content or ""
                self._memory.add(
                    Message(role=Role.ASSISTANT, content=answer)
                )
                return answer

            # --- Step 3: Execute tool calls ---
            self._state = AgentState.ACTING
            tool_results = await self._executor.execute_all(response.tool_calls)

            # --- Step 4: Observe results ---
            self._state = AgentState.OBSERVING

            # Add assistant message (with tool calls) to memory
            self._memory.add(
                Message(
                    role=Role.ASSISTANT,
                    content=response.content,
                    tool_calls=response.tool_calls,
                )
            )

            # Add tool result messages to memory
            for tr in tool_results:
                self._memory.add(tr.to_message())

        # Max iterations reached
        self._state = AgentState.ERROR
        self._log.warning(f"⚠️ Max iterations ({max_iter}) reached — forcing summary.")
        return await self._force_final_answer()

    async def run_stream(self, user_input: str) -> AsyncIterator[str]:
        """Run the agent with streaming output.

        Yields tokens of the final answer (streaming intermediate
        states and tool-call status are also yielded as rich markup).
        """
        from ..core.executor import Executor

        self._executor = Executor(self._tools)
        self._memory.add(Message(role=Role.USER, content=user_input))

        iteration = 0
        max_iter = self._settings.agent_max_iterations

        while iteration < max_iter:
            iteration += 1
            yield f"\n[bold cyan]🔄 Iteration {iteration}/{max_iter}[/bold cyan]\n"

            # --- Call LLM ---
            self._state = AgentState.THINKING
            response = await self._call_llm()

            if not response.tool_calls:
                self._state = AgentState.DONE
                answer = response.content or ""
                self._memory.add(
                    Message(role=Role.ASSISTANT, content=answer)
                )
                yield answer
                return

            # --- Tool calls ---
            self._state = AgentState.ACTING
            tool_names = [tc.name for tc in response.tool_calls]
            yield f"[dim]🔧 Calling: {', '.join(tool_names)}[/dim]\n"

            tool_results = await self._executor.execute_all(response.tool_calls)

            self._state = AgentState.OBSERVING
            for tr in tool_results:
                status = "✅" if tr.success else "❌"
                preview = tr.content[:200] + ("..." if len(tr.content) > 200 else "")
                yield f"[dim]{status} {tr.name}: {preview}[/dim]\n"

            self._memory.add(
                Message(
                    role=Role.ASSISTANT,
                    content=response.content,
                    tool_calls=response.tool_calls,
                )
            )
            for tr in tool_results:
                self._memory.add(tr.to_message())

        self._state = AgentState.ERROR
        answer = await self._force_final_answer()
        yield answer

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _call_llm(self) -> Message:
        """Call the LLM with the current conversation and tool definitions."""
        messages = self._memory.get_messages()
        tool_defs = self._tools.to_openai_schema() if self._tools.list_all() else None

        try:
            return await self._llm.chat(messages, tools=tool_defs)
        except Exception as exc:
            self._log.error(f"LLM call failed: {exc}")
            self._state = AgentState.ERROR
            # Return error as text
            return Message(
                role=Role.ASSISTANT,
                content=f"❌ LLM error: {exc}",
            )

    async def _force_final_answer(self) -> str:
        """Ask the LLM to summarize based on the conversation so far."""
        self._memory.add(
            Message(
                role=Role.USER,
                content=(
                    "You have reached the maximum number of steps. "
                    "Please provide your best final answer based on what "
                    "you have observed so far. Do NOT call any more tools."
                ),
            )
        )
        response = await self._call_llm()
        answer = response.content or "Unable to produce a final answer."
        self._memory.add(Message(role=Role.ASSISTANT, content=answer))
        self._state = AgentState.DONE
        return answer

    def reset(self) -> None:
        """Reset the agent memory and state for a new conversation."""
        self._memory.clear()
        self._state = AgentState.IDLE
