"""Action executor — runs tool calls and collects results."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..core.types import ToolCall, ToolResult

if TYPE_CHECKING:
    from ..tools.registry import ToolRegistry


class Executor:
    """Executes tool calls and returns results.

    Handles:
    - Looking up tools in the registry
    - Running tool execute() methods
    - Converting results / errors into ToolResult objects
    """

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    async def execute(self, tool_call: ToolCall) -> ToolResult:
        """Execute a single tool call and return the result."""
        tool = self._registry.get(tool_call.name)
        if tool is None:
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                content=f"Error: Unknown tool '{tool_call.name}'. "
                f"Available: {', '.join(self._registry.list_names())}",
                success=False,
                error=f"Tool '{tool_call.name}' not registered.",
            )

        try:
            output = await tool.execute(**tool_call.arguments)
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                content=str(output),
                success=True,
            )
        except Exception as exc:
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                content=f"Error executing '{tool_call.name}': {exc}",
                success=False,
                error=str(exc),
            )

    async def execute_all(self, tool_calls: list[ToolCall]) -> list[ToolResult]:
        """Execute multiple tool calls (sequentially for safety)."""
        results: list[ToolResult] = []
        for tc in tool_calls:
            result = await self.execute(tc)
            results.append(result)
        return results
