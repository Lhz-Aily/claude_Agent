"""CLI entry point for Claude Agent."""

from __future__ import annotations

import asyncio
import sys

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

# Load .env before anything else
load_dotenv()

from .config.settings import Settings
from .core.agent import Agent
from .llm.factory import create_llm_provider
from .memory.conversation import ConversationMemory
from .tools.registry import ToolRegistry
from .utils.logger import configure_root_logger, get_logger

console = Console()
log = get_logger("cli")


# ---------------------------------------------------------------------------
# Helper — build the agent with all its dependencies
# ---------------------------------------------------------------------------

def build_agent(settings: Settings | None = None) -> Agent:
    """Wire up the full agent stack."""
    if settings is None:
        settings = Settings()  # type: ignore[call-arg]

    configure_root_logger(settings.log_level)

    # LLM
    llm = create_llm_provider(settings)

    # Tools
    tools = ToolRegistry()
    tools.discover_builtin()
    tools.discover_custom()

    # Memory
    memory = ConversationMemory(
        max_tokens=settings.conversation_max_tokens,
        system_prompt=settings.system_prompt,
    )

    return Agent(llm=llm, tools=tools, memory=memory, settings=settings)


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

@click.group()
@click.version_option(version="0.1.0", prog_name="claude-agent")
def cli() -> None:
    """Claude Agent — A modular, multi-provider Python AI Agent.

    Use 'chat' for interactive mode or 'run' for a one-shot task.
    """


@cli.command()
@click.option(
    "--provider",
    default=None,
    type=click.Choice(["openai", "anthropic"]),
    help="LLM provider to use.",
)
@click.option("--model", default=None, help="Override the model name.")
def chat(provider: str | None, model: str | None) -> None:
    """Start an interactive chat session with the Agent."""
    settings = Settings()  # type: ignore[call-arg]
    if provider:
        settings.llm_provider = provider
    if model:
        if settings.llm_provider == "openai":
            settings.openai_model = model
        else:
            settings.anthropic_model = model

    agent = build_agent(settings)

    console.print(
        Panel.fit(
            f"[bold green]Claude Agent v0.1.0[/bold green]\n"
            f"Provider: [cyan]{settings.llm_provider}[/cyan]  |  "
            f"Model: [cyan]{settings.openai_model if settings.llm_provider == 'openai' else settings.anthropic_model}[/cyan]  |  "
            f"Max iterations: [cyan]{settings.agent_max_iterations}[/cyan]\n"
            f"Type [yellow]/exit[/yellow] to quit, [yellow]/reset[/yellow] to clear history, [yellow]/help[/yellow] for commands.",
            title="🚀 Ready",
            border_style="blue",
        )
    )

    while True:
        try:
            user_input = console.input("[bold green]You ›[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/dim]")
            return

        if not user_input:
            continue

        # Built-in slash commands
        if user_input.startswith("/"):
            _handle_slash_command(user_input, agent, settings)
            continue

        # Run agent
        console.print()
        with console.status("[bold yellow]Thinking…[/bold yellow]", spinner="dots"):
            try:
                result = asyncio.run(agent.run(user_input))
            except Exception as exc:
                log.error(f"Agent error: {exc}")
                result = f"❌ Error: {exc}"

        console.print(
            Panel(
                Markdown(result),
                title="[bold blue]Agent[/bold blue]",
                border_style="blue",
            )
        )
        console.print()


@cli.command()
@click.argument("task")
@click.option(
    "--provider",
    default=None,
    type=click.Choice(["openai", "anthropic"]),
    help="LLM provider.",
)
@click.option("--model", default=None, help="Override the model name.")
@click.option(
    "--stream/--no-stream",
    default=False,
    help="Stream the response token-by-token.",
)
def run(task: str, provider: str | None, model: str | None, stream: bool) -> None:
    """Run a single task and exit."""
    settings = Settings()  # type: ignore[call-arg]
    if provider:
        settings.llm_provider = provider
    if model:
        if settings.llm_provider == "openai":
            settings.openai_model = model
        else:
            settings.anthropic_model = model

    agent = build_agent(settings)

    async def _run() -> None:
        if stream:
            async for chunk in agent.run_stream(task):
                console.print(chunk, end="")
            console.print()
        else:
            with console.status("[bold yellow]Working…[/bold yellow]", spinner="dots"):
                result = await agent.run(task)
            console.print(Markdown(result))

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted.[/dim]")
    except Exception as exc:
        log.error(f"Error: {exc}")
        sys.exit(1)


@cli.command()
def version() -> None:
    """Print version info."""
    from . import __version__

    console.print(f"[bold]Claude Agent[/bold] v{__version__}")


# ---------------------------------------------------------------------------
# Slash command handler
# ---------------------------------------------------------------------------

def _handle_slash_command(cmd: str, agent: Agent, settings: Settings) -> None:
    """Process built-in slash commands in interactive mode."""
    parts = cmd.split()
    command = parts[0].lower()

    if command == "/exit" or command == "/quit":
        console.print("[dim]Goodbye![/dim]")
        sys.exit(0)

    elif command == "/reset":
        agent.reset()
        console.print("[dim]🔄 Conversation history cleared.[/dim]")

    elif command == "/help":
        console.print(
            Panel(
                "[bold]Available commands:[/bold]\n\n"
                "  [yellow]/exit[/yellow]    — Quit the session\n"
                "  [yellow]/reset[/yellow]   — Clear conversation history\n"
                "  [yellow]/help[/yellow]    — Show this help\n"
                "  [yellow]/tools[/yellow]   — List available tools\n"
                "  [yellow]/state[/yellow]   — Show agent state\n\n"
                "[bold]Tips:[/bold]\n"
                "  • Type any question or task naturally.\n"
                "  • The Agent will use tools automatically when needed.\n"
                "  • Use [cyan]Ctrl+C[/cyan] to interrupt generation.",
                title="Help",
                border_style="green",
            )
        )

    elif command == "/tools":
        tool_names = agent._tools.list_names()
        if tool_names:
            tools_list = "\n".join(f"  • [cyan]{n}[/cyan]" for n in tool_names)
            console.print(f"[bold]Available tools:[/bold]\n{tools_list}")
        else:
            console.print("[dim]No tools registered.[/dim]")

    elif command == "/state":
        console.print(f"Agent state: [yellow]{agent.state.value}[/yellow]")

    else:
        console.print(f"[dim]Unknown command: {command}. Type /help for available commands.[/dim]")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cli()
