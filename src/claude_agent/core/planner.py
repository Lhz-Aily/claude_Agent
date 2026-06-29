"""Task planner — breaks complex tasks into sub-steps."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from ..llm.base import BaseLLMProvider


class PlanStep(BaseModel):
    """A single step in an execution plan."""

    index: int
    description: str
    tool_name: str | None = None
    depends_on: list[int] = Field(default_factory=list)


class Plan(BaseModel):
    """An execution plan composed of multiple steps."""

    goal: str
    steps: list[PlanStep]
    reasoning: str = ""


class Planner:
    """Generates execution plans for complex user requests.

    The planner asks the LLM to decompose a high-level goal
    into a sequence of concrete, executable steps.
    """

    SYSTEM_PROMPT = """You are a task planner. Given a user's goal, break it down into
a short sequence of concrete, actionable steps (3-7 steps typically).

Return your plan as a JSON object with this exact structure:
{
  "reasoning": "Brief explanation of your approach",
  "steps": [
    {
      "index": 1,
      "description": "What to do in this step",
      "tool_name": null
    }
  ]
}

Rules:
- Each step must be a single, concrete action.
- Use "tool_name" only when you are certain a specific tool is needed; otherwise leave it null.
- "depends_on" lists the indices of steps that must complete before this one.
- Keep steps actionable — avoid vague instructions."""

    def __init__(self, llm: BaseLLMProvider) -> None:
        self._llm = llm

    async def plan(self, goal: str) -> Plan:
        """Generate a plan for the given goal."""
        from ..core.types import Message, Role

        messages = [
            Message(role=Role.SYSTEM, content=self.SYSTEM_PROMPT),
            Message(role=Role.USER, content=f"Goal: {goal}\n\nGenerate a plan."),
        ]

        response = await self._llm.chat(messages, tools=None)
        content = response.content or ""

        # Try to parse JSON from the response
        import json

        try:
            # Handle possible markdown code fences
            content = content.strip()
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1])
            data = json.loads(content)
            return Plan(
                goal=goal,
                reasoning=data.get("reasoning", ""),
                steps=[PlanStep(**s) for s in data.get("steps", [])],
            )
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            # Fallback: treat as a single-step plan
            return Plan(
                goal=goal,
                reasoning=f"Plan parsing failed ({e}), executing directly.",
                steps=[PlanStep(index=1, description=goal)],
            )
