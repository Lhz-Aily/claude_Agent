"""Tool registry — stores and discovers tools."""

from __future__ import annotations

from typing import Any

from ..utils.logger import get_logger
from .base import BaseTool


class ToolRegistry:
    """Holds all available tools and provides lookup / schema generation.

    Tools are registered by name. The registry can also auto-discover
    tools from the `builtin` and `custom` packages.
    """

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}
        self._log = get_logger(__name__)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def register(self, tool: BaseTool) -> None:
        """Add a tool to the registry."""
        self._tools[tool.name] = tool
        self._log.debug(f"Registered tool: {tool.name}")

    def unregister(self, name: str) -> None:
        """Remove a tool by name."""
        self._tools.pop(name, None)

    def get(self, name: str) -> BaseTool | None:
        """Look up a tool by name."""
        return self._tools.get(name)

    def list_all(self) -> list[BaseTool]:
        """Return all registered tools."""
        return list(self._tools.values())

    def list_names(self) -> list[str]:
        """Return names of all registered tools."""
        return list(self._tools.keys())

    # ------------------------------------------------------------------
    # Schema helpers
    # ------------------------------------------------------------------

    def to_openai_schema(self) -> list[dict[str, Any]]:
        """All tools as OpenAI function-calling definitions."""
        return [t.to_openai_schema() for t in self._tools.values()]

    # ------------------------------------------------------------------
    # Auto-discovery
    # ------------------------------------------------------------------

    def discover_builtin(self) -> None:
        """Import and register all built-in tools."""
        from .builtin.file_tools import ReadFileTool, WriteFileTool
        from .builtin.web_tools import HttpGetTool, HttpPostTool
        from .builtin.shell_tools import ShellTool
        from .builtin.python_tools import PythonTool
        from .builtin.search_tools import WebSearchTool

        for tool_cls in [
            ReadFileTool,
            WriteFileTool,
            HttpGetTool,
            HttpPostTool,
            ShellTool,
            PythonTool,
            WebSearchTool,
        ]:
            self.register(tool_cls())

        self._log.info(f"Loaded {len(self._tools)} built-in tools.")

    def discover_custom(self, custom_dir: str = "") -> None:
        """Discover custom tools from the `tools/custom/` directory.

        Each `.py` file in the directory is imported; any subclass
        of BaseTool found inside is registered.
        """
        import importlib
        import pkgutil
        from pathlib import Path

        from . import custom as custom_pkg

        if not custom_dir:
            custom_dir = str(Path(custom_pkg.__path__[0]))

        custom_path = Path(custom_dir)
        if not custom_path.exists():
            self._log.info("No custom tools directory found; skipping.")
            return

        count = 0
        for finder, name, ispkg in pkgutil.iter_modules([str(custom_path)]):
            if ispkg:
                continue
            try:
                module = importlib.import_module(
                    f"claude_agent.tools.custom.{name}"
                )
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (
                        isinstance(attr, type)
                        and issubclass(attr, BaseTool)
                        and attr is not BaseTool
                    ):
                        self.register(attr())
                        count += 1
            except Exception as exc:
                self._log.warning(f"Failed to load custom tool '{name}': {exc}")

        self._log.info(f"Loaded {count} custom tool(s).")
