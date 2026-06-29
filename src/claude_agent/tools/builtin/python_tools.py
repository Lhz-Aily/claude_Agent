"""Python execution tool — runs Python code in a sandboxed-ish env."""

from __future__ import annotations

import asyncio
import sys
import traceback
from io import StringIO

from ..base import BaseTool, ToolError


class PythonTool(BaseTool):
    name = "python"
    description = (
        "Execute a snippet of Python code and capture print() output. "
        "Use for calculations, data processing, or quick scripting. "
        "⚠️ Has access to your Python environment — avoid running untrusted code."
    )
    parameters = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python code to execute. Use print() to output results.",
            },
            "timeout_sec": {
                "type": "integer",
                "description": "Max execution time in seconds (default: 30).",
                "default": 30,
            },
        },
        "required": ["code"],
    }

    async def execute(self, code: str, timeout_sec: int = 30) -> str:
        # We run in a subprocess to get true isolation and timeout
        cmd = [sys.executable, "-c", code]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout_sec
            )
            parts: list[str] = []
            if stdout:
                parts.append(stdout.decode("utf-8", errors="replace"))
            if stderr:
                parts.append(f"[stderr]\n{stderr.decode('utf-8', errors='replace')}")
            if proc.returncode != 0:
                parts.append(f"[exit code: {proc.returncode}]")
            return "\n".join(parts) if parts else "(no output)"
        except asyncio.TimeoutError:
            raise ToolError(f"Python code timed out after {timeout_sec}s")
