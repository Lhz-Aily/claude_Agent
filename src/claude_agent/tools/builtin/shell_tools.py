"""Shell execution tool — runs system commands safely."""

from __future__ import annotations

import asyncio
import os

from ..base import BaseTool, ToolError


class ShellTool(BaseTool):
    name = "shell"
    description = (
        "Execute a shell command and return its output (stdout + stderr). "
        "Use for file operations, git commands, pip installs, etc. "
        "⚠️ This tool can modify your system — review commands carefully."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute.",
            },
            "working_dir": {
                "type": "string",
                "description": "Directory to run the command in (default: current).",
            },
            "timeout_sec": {
                "type": "integer",
                "description": "Max execution time in seconds (default: 60).",
                "default": 60,
            },
        },
        "required": ["command"],
    }

    # Blocklist of dangerous patterns (user must explicitly approve)
    DANGEROUS_PATTERNS = [
        "rm -rf /",
        "mkfs.",
        "dd if=",
        "> /dev/sda",
        "chmod 777 /",
    ]

    async def execute(
        self,
        command: str,
        working_dir: str | None = None,
        timeout_sec: int = 60,
    ) -> str:
        # Basic safety check
        for pattern in self.DANGEROUS_PATTERNS:
            if pattern in command:
                raise ToolError(
                    f"Command blocked — matches dangerous pattern '{pattern}'."
                )

        cwd = working_dir or os.getcwd()
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout_sec
            )
            output_parts: list[str] = []
            if stdout:
                output_parts.append(f"[stdout]\n{stdout.decode('utf-8', errors='replace')}")
            if stderr:
                output_parts.append(f"[stderr]\n{stderr.decode('utf-8', errors='replace')}")
            output_parts.append(f"\n[exit code: {proc.returncode}]")
            return "\n".join(output_parts)
        except asyncio.TimeoutError:
            raise ToolError(f"Command timed out after {timeout_sec}s: {command}")
        except FileNotFoundError as exc:
            raise ToolError(f"Command not found: {exc}") from exc
