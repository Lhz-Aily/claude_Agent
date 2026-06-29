"""Application settings loaded from environment variables and .env file."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All configurable parameters for the Claude Agent.

    Values are read from environment variables (or a .env file) first,
    with the defaults below.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM Provider ---
    llm_provider: str = "openai"

    # --- OpenAI ---
    openai_api_key: str = "sk-placeholder"
    openai_model: str = "gpt-4o"
    openai_base_url: str = ""

    # --- Anthropic ---
    anthropic_api_key: str = "sk-ant-placeholder"
    anthropic_model: str = "claude-sonnet-4-6-20250514"

    # --- Agent ---
    agent_max_iterations: int = 10
    agent_temperature: float = 0.7
    agent_max_tokens: int = 4096

    # --- Memory ---
    memory_dir: str = "./data/memory"
    vector_store_dir: str = "./data/vector_store"
    conversation_max_tokens: int = 128_000

    # --- System Prompt ---
    system_prompt: str = (
        "You are Claude Agent, a helpful AI assistant with access to tools. "
        "Follow this process for each user request:\n"
        "1. Understand the user's goal.\n"
        "2. If you need information or need to take action, use the appropriate tool.\n"
        "3. Observe the tool result and decide if more actions are needed.\n"
        "4. When you have enough information, provide a clear, well-structured final answer.\n"
        "Be thorough, accurate, and helpful. When you encounter errors, "
        "try an alternative approach before giving up."
    )

    # --- Logging ---
    log_level: str = "INFO"

    # ------------------------------------------------------------------
    # Computed / helper
    # ------------------------------------------------------------------

    @property
    def memory_path(self) -> Path:
        """Resolved memory directory."""
        p = Path(self.memory_dir).resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def vector_store_path(self) -> Path:
        """Resolved vector store directory."""
        p = Path(self.vector_store_dir).resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p
