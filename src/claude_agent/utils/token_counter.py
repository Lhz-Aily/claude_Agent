"""Token counting utilities using tiktoken."""

from __future__ import annotations


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Count the number of tokens in a text string.

    Args:
        text: The text to count.
        model: Model name for encoding selection (default: gpt-4o).

    Returns:
        Approximate token count.
    """
    try:
        import tiktoken

        try:
            enc = tiktoken.encoding_for_model(model)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except ImportError:
        # Fallback: ~4 characters per token
        return len(text) // 4


def count_message_tokens(messages: list[dict[str, str]]) -> int:
    """Count total tokens across a list of message dicts."""
    total = 0
    for msg in messages:
        content = msg.get("content", "") or ""
        total += count_tokens(content)
        # Overhead for role, formatting
        total += 4
    return total
