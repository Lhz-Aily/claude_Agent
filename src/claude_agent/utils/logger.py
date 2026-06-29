"""Centralized logging configuration using Rich."""

from __future__ import annotations

import logging
import sys

from rich.console import Console
from rich.logging import RichHandler


_console: Console | None = None
_loggers: dict[str, logging.Logger] = {}


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for the given module name.

    Uses RichHandler for beautiful console output. The log level is
    controlled by the LOG_LEVEL env var (default: INFO).
    """
    global _console, _loggers

    if name in _loggers:
        return _loggers[name]

    if _console is None:
        _console = Console(stderr=True)

    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = RichHandler(
            console=_console,
            show_time=False,
            show_path=False,
            rich_tracebacks=True,
            markup=True,
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)

    # Respect LOG_LEVEL env var
    import os

    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    logger.setLevel(getattr(logging, level_name, logging.INFO))

    # Don't propagate to root logger
    logger.propagate = False

    _loggers[name] = logger
    return logger


def configure_root_logger(level: str = "INFO") -> None:
    """Configure the root logger. Call once at startup."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True, markup=True)],
    )
