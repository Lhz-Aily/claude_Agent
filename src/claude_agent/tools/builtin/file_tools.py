"""File system tools — read and write files."""

from __future__ import annotations

from pathlib import Path

from ..base import BaseTool, ToolError


class ReadFileTool(BaseTool):
    name = "read_file"
    description = "Read the contents of a file. Use to inspect text files, logs, configs, etc."
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute or relative path to the file.",
            },
            "encoding": {
                "type": "string",
                "description": "File encoding (default: utf-8).",
                "default": "utf-8",
            },
        },
        "required": ["path"],
    }

    async def execute(self, path: str, encoding: str = "utf-8") -> str:
        p = Path(path).expanduser().resolve()
        if not p.exists():
            raise ToolError(f"File not found: {p}")
        if p.stat().st_size > 10 * 1024 * 1024:  # 10 MB limit
            raise ToolError(f"File too large ({p.stat().st_size} bytes); use a smaller file.")
        content = p.read_text(encoding=encoding)
        return content


class WriteFileTool(BaseTool):
    name = "write_file"
    description = "Write content to a file. Creates parent directories if needed."
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute or relative path to the file.",
            },
            "content": {
                "type": "string",
                "description": "Text content to write.",
            },
            "encoding": {
                "type": "string",
                "description": "File encoding (default: utf-8).",
                "default": "utf-8",
            },
        },
        "required": ["path", "content"],
    }

    async def execute(self, path: str, content: str, encoding: str = "utf-8") -> str:
        p = Path(path).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding=encoding)
        return f"Successfully wrote {len(content)} characters to {p}"
