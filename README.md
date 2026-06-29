# Claude Agent

A modular, multi-provider Python AI Agent framework following the **ReAct** (Reasoning + Acting) paradigm.

## Features

- 🔄 **ReAct Loop** — Thought → Action → Observation → Answer
- 🔌 **Multi-Provider** — OpenAI (GPT-4o) and Anthropic (Claude) supported
- 🛠️ **Extensible Tools** — 6 built-in tools + custom tool auto-discovery
- 🧠 **Memory System** — Conversation history with sliding window + vector storage
- ⚡ **Streaming** — Real-time token streaming in CLI
- 🎨 **Rich Output** — Beautiful terminal UI with Rich
- 🔧 **Fully Typed** — Comprehensive type hints and Pydantic models

## Quick Start

### 1. Install

```bash
cd E:\Agent\Claude_Agent
pip install -e .
```

### 2. Configure

Copy `.env.example` to `.env` and set your API key:

```bash
cp .env.example .env
```

Edit `.env`:
```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-key-here
```

### 3. Run

```bash
# Interactive chat
claude-agent chat

# Single task
claude-agent run "Calculate 128 * 45 and explain the steps"

# With streaming
claude-agent run "Search for Python 3.13 features" --stream
```

## Architecture

```
src/claude_agent/
├── core/           # Agent loop, planner, executor, types
├── llm/            # LLM abstraction (OpenAI + Anthropic)
├── tools/          # Tool system (base, registry, builtin, custom)
├── memory/         # Conversation + vector memory
├── config/         # Pydantic Settings
└── utils/          # Logging, token counting
```

## Built-in Tools

| Tool | Description |
|------|-------------|
| `read_file` | Read file contents |
| `write_file` | Write content to file |
| `http_get` | HTTP GET request |
| `http_post` | HTTP POST request |
| `shell` | Execute shell commands |
| `python` | Run Python code snippets |
| `web_search` | Search the web (DuckDuckGo) |

## Custom Tools

Create a `.py` file in `src/claude_agent/tools/custom/`:

```python
from claude_agent.tools.base import BaseTool

class MyTool(BaseTool):
    name = "my_tool"
    description = "Does something useful."
    parameters = {
        "type": "object",
        "properties": {
            "input": {"type": "string", "description": "..."},
        },
        "required": ["input"],
    }

    async def execute(self, input: str) -> str:
        return f"Result: {input}"
```

The tool is auto-discovered on startup.

## Usage as a Library

```python
import asyncio
from claude_agent import Agent, Settings, create_llm_provider
from claude_agent.memory.conversation import ConversationMemory
from claude_agent.tools.registry import ToolRegistry

async def main():
    settings = Settings()
    llm = create_llm_provider(settings)

    tools = ToolRegistry()
    tools.discover_builtin()

    memory = ConversationMemory(
        max_tokens=settings.conversation_max_tokens,
        system_prompt=settings.system_prompt,
    )

    agent = Agent(llm=llm, tools=tools, memory=memory, settings=settings)
    answer = await agent.run("What can you do?")
    print(answer)

asyncio.run(main())
```

## Testing

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT
