"""Factory for creating LLM providers from configuration."""

from __future__ import annotations

from ..utils.logger import get_logger
from .base import BaseLLMProvider


def create_llm_provider(settings: Any) -> BaseLLMProvider:
    """Build the configured LLM provider.

    Args:
        settings: A Settings instance (duck-typed; any object with the
                  required attributes).

    Returns:
        A concrete BaseLLMProvider instance.

    Raises:
        ValueError: If the configured LLM_PROVIDER is not supported.
    """
    log = get_logger(__name__)
    provider_name = settings.llm_provider.lower()

    if provider_name == "openai":
        from .openai_provider import OpenAIProvider

        log.info(f"Using OpenAI provider: {settings.openai_model}")
        return OpenAIProvider(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url or None,
            temperature=settings.agent_temperature,
            max_tokens=settings.agent_max_tokens,
        )

    elif provider_name == "anthropic":
        from .anthropic_provider import AnthropicProvider

        log.info(f"Using Anthropic provider: {settings.anthropic_model}")
        return AnthropicProvider(
            model=settings.anthropic_model,
            api_key=settings.anthropic_api_key,
            temperature=settings.agent_temperature,
            max_tokens=settings.agent_max_tokens,
        )

    else:
        raise ValueError(
            f"Unsupported LLM provider: '{settings.llm_provider}'. "
            "Use 'openai' or 'anthropic'."
        )
