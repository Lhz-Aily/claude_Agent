"""Tests for the tools system."""

import pytest

from claude_agent.tools.base import BaseTool, ToolError
from claude_agent.tools.registry import ToolRegistry
from claude_agent.tools.builtin.file_tools import ReadFileTool, WriteFileTool
from claude_agent.tools.builtin.python_tools import PythonTool


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
        return f"echo: {text}"


# ---------------------------------------------------------------------------
# Tool Registry
# ---------------------------------------------------------------------------

class TestToolRegistry:
    def test_register_and_get(self):
        registry = ToolRegistry()
        tool = EchoTool()
        registry.register(tool)
        assert registry.get("echo") is tool
        assert "echo" in registry.list_names()

    def test_unregister(self):
        registry = ToolRegistry()
        registry.register(EchoTool())
        registry.unregister("echo")
        assert registry.get("echo") is None

    def test_to_openai_schema(self):
        registry = ToolRegistry()
        registry.register(EchoTool())
        schemas = registry.to_openai_schema()
        assert len(schemas) == 1
        assert schemas[0]["type"] == "function"
        assert schemas[0]["function"]["name"] == "echo"

    def test_discover_builtin(self):
        registry = ToolRegistry()
        registry.discover_builtin()
        names = registry.list_names()
        assert "read_file" in names
        assert "write_file" in names
        assert "http_get" in names
        assert "shell" in names
        assert "python" in names
        assert "web_search" in names


# ---------------------------------------------------------------------------
# Echo Tool
# ---------------------------------------------------------------------------

class TestEchoTool:
    @pytest.mark.asyncio
    async def test_execute(self):
        tool = EchoTool()
        result = await tool.execute(text="hello")
        assert result == "echo: hello"


# ---------------------------------------------------------------------------
# Python Tool
# ---------------------------------------------------------------------------

class TestPythonTool:
    @pytest.mark.asyncio
    async def test_simple_execution(self):
        tool = PythonTool()
        result = await tool.execute(code="print('hello world')")
        assert "hello world" in result

    @pytest.mark.asyncio
    async def test_error_handling(self):
        tool = PythonTool()
        result = await tool.execute(code="raise ValueError('test error')")
        assert "ValueError" in result or "test error" in result


# ---------------------------------------------------------------------------
# Read File Tool
# ---------------------------------------------------------------------------

class TestReadFileTool:
    @pytest.mark.asyncio
    async def test_read_existing_file(self, tmp_path):
        tool = ReadFileTool()
        file = tmp_path / "test.txt"
        file.write_text("hello from test")
        result = await tool.execute(path=str(file))
        assert "hello from test" in result

    @pytest.mark.asyncio
    async def test_read_nonexistent_file(self):
        tool = ReadFileTool()
        with pytest.raises(ToolError, match="File not found"):
            await tool.execute(path="/nonexistent/file_12345.txt")


# ---------------------------------------------------------------------------
# Write File Tool
# ---------------------------------------------------------------------------

class TestWriteFileTool:
    @pytest.mark.asyncio
    async def test_write_file(self, tmp_path):
        tool = WriteFileTool()
        file = tmp_path / "subdir" / "output.txt"
        result = await tool.execute(path=str(file), content="test content")
        assert "Successfully wrote" in result
        assert file.read_text() == "test content"
