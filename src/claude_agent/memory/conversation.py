"""Conversation memory — stores the current session's message history."""

from __future__ import annotations

import tiktoken

from ..core.types import Message, Role
from ..utils.logger import get_logger
from .base import BaseMemory


class ConversationMemory(BaseMemory):
    """In-memory conversation history with sliding-window token management.

    When the total token count exceeds `max_tokens`, older messages
    (except the system prompt) are dropped to make room.
    """

    def __init__(
        self,
        max_tokens: int = 128_000,
        system_prompt: str = "",
    ) -> None:
        self._messages: list[Message] = []
        self._max_tokens = max_tokens
        self._log = get_logger(__name__)

        if system_prompt:
            self._messages.append(Message(role=Role.SYSTEM, content=system_prompt))

        try:
            self._enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self._enc = None  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, message: Message) -> None:
        """Append a message and trim if over the token limit."""
        self._messages.append(message)
        self._maybe_trim()

    def get_messages(self) -> list[Message]:
        """Return a copy of all messages."""
        return list(self._messages)

    def clear(self) -> None:
        """Reset to just the system prompt (if any)."""
        system = self._messages[0] if self._messages and self._messages[0].role == Role.SYSTEM else None
        self._messages = [system] if system else []

    @property
    def token_count(self) -> int:
        """Estimated total token count of all messages."""
        return self._count_tokens(self._messages)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _count_tokens(self, messages: list[Message]) -> int:
        """Estimate token count for a list of messages."""
        if self._enc is None:
            # Fallback: ~4 chars per token
            return sum(
                len(m.content or "") // 4 for m in messages
            )
        total = 0
        for m in messages:
            if m.content:
                total += len(self._enc.encode(m.content))
            # Rough overhead per message
            total += 4
        return total

    def _maybe_trim(self) -> None:
        """Remove oldest non-system messages until under the token limit."""
        while self.token_count > self._max_tokens and len(self._messages) > 1:
            # Never remove the system prompt at index 0
            idx = 1 if (self._messages[0].role == Role.SYSTEM) else 0
            if idx >= len(self._messages):
                break
            removed = self._messages.pop(idx)
            self._log.debug(
                f"Trimmed message (tokens={self.token_count}): "
                f"{removed.role.value} | {str(removed.content)[:60]}..."
            )
