# LangGraph ReAct Agent

A simple ReAct (Reasoning and Acting) agent built with LangGraph and Google Gemini API.

## Setup

1. Install dependencies:
```bash
uv sync
```

2. Copy the environment file and add your API key:
```bash
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
```

3. Run the agent:
```bash
uv run python main.py
```

## Project Structure

```
langgraph-react-agent/
├── pyproject.toml           # Project configuration
├── src/
│   └── react_agent/
│       ├── __init__.py      # Package exports
│       ├── agent.py         # ReAct agent graph
│       ├── state.py         # Agent state definition
│       └── tools.py         # Tool definitions
├── main.py                  # Entry point
└── .env.example             # Environment variables template
```

## Tools

- **search_web**: Mock web search tool
- **calculator**: Mathematical expression evaluator
