"""Web tools — HTTP GET and POST requests."""

from __future__ import annotations

import httpx

from ..base import BaseTool, ToolError


class HttpGetTool(BaseTool):
    name = "http_get"
    description = "Send an HTTP GET request to a URL. Use to fetch data from APIs or web pages."
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL to fetch (e.g., https://api.example.com/data).",
            },
            "headers": {
                "type": "object",
                "description": "Optional HTTP headers as key-value pairs.",
                "additionalProperties": {"type": "string"},
            },
        },
        "required": ["url"],
    }

    async def execute(
        self, url: str, headers: dict[str, str] | None = None
    ) -> str:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            try:
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                content_type = response.headers.get("content-type", "")
                # Truncate very large responses
                text = response.text[:50_000]
                return text
            except httpx.HTTPError as exc:
                raise ToolError(f"HTTP GET failed for '{url}': {exc}") from exc


class HttpPostTool(BaseTool):
    name = "http_post"
    description = "Send an HTTP POST request with a JSON body."
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL to post to.",
            },
            "body": {
                "type": "object",
                "description": "JSON body to send.",
            },
            "headers": {
                "type": "object",
                "description": "Optional HTTP headers.",
                "additionalProperties": {"type": "string"},
            },
        },
        "required": ["url", "body"],
    }

    async def execute(
        self,
        url: str,
        body: dict,
        headers: dict[str, str] | None = None,
    ) -> str:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            try:
                response = await client.post(url, json=body, headers=headers)
                response.raise_for_status()
                return response.text[:50_000]
            except httpx.HTTPError as exc:
                raise ToolError(f"HTTP POST failed for '{url}': {exc}") from exc
