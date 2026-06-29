"""Web search tool — uses DuckDuckGo instant answers (no API key needed)."""

from __future__ import annotations

import urllib.parse

import httpx

from ..base import BaseTool, ToolError


class WebSearchTool(BaseTool):
    name = "web_search"
    description = (
        "Search the web and return snippets. "
        "Use for finding current information, facts, or documentation."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query.",
            },
            "num_results": {
                "type": "integer",
                "description": "Number of results to return (max 10).",
                "default": 5,
            },
        },
        "required": ["query"],
    }

    async def execute(self, query: str, num_results: int = 5) -> str:
        num_results = min(num_results, 10)

        # Use DuckDuckGo Instant Answer API (non-commercial, no key needed)
        encoded = urllib.parse.quote(query)
        url = f"https://api.duckduckgo.com/?q={encoded}&format=json&no_html=1"

        async with httpx.AsyncClient(timeout=15.0) as client:
            try:
                response = await client.get(url)
                response.raise_for_status()
                data = response.json()
            except httpx.HTTPError as exc:
                raise ToolError(f"Search request failed: {exc}") from exc

        lines: list[str] = []

        # Abstract / instant answer
        abstract = data.get("AbstractText", "")
        if abstract:
            lines.append(f"📌 Summary: {abstract}")
            lines.append("")

        # Related topics
        related = data.get("RelatedTopics", [])
        results_count = 0
        for item in related:
            if isinstance(item, dict) and "Text" in item:
                text = item["Text"]
                url_ref = item.get("FirstURL", "")
                lines.append(f"- {text}")
                if url_ref:
                    lines.append(f"  🔗 {url_ref}")
                results_count += 1
                if results_count >= num_results:
                    break

        if not lines:
            return f"No results found for '{query}'."

        return "\n".join(lines)
