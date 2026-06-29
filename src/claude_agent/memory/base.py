"""Abstract base class for memory implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..core.types import Message


class BaseMemory(ABC):
    """Abstract interface for memory backends."""

    @abstractmethod
    def add(self, message: Message) -> None:
        """Store a message."""
        ...

    @abstractmethod
    def get_messages(self) -> list[Message]:
        """Retrieve all messages."""
        ...

    @abstractmethod
    def clear(self) -> None:
        """Remove all messages."""
        ...
