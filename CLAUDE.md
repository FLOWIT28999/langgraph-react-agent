# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LangGraph ReAct Agent - A Python implementation of the ReAct (Reasoning and Acting) pattern using LangGraph and Google Gemini API. The agent reasons about tasks and executes tools (web search, calculator) to accomplish goals.

## Commands

```bash
# Install dependencies
uv sync

# Run the agent
uv run python main.py
```

No build, test, or lint commands are configured.

## Architecture

### ReAct Loop Pattern

The agent implements a reasoning/acting loop using LangGraph's StateGraph:

```
Entry → agent_node → should_continue
                        ├→ "tools" → tools_node → agent_node (loop)
                        └→ END
```

1. **agent_node**: Calls Gemini LLM with bound tools, processes messages
2. **should_continue**: Routes to tools if AIMessage has tool_calls, otherwise ends
3. **tools_node**: Executes selected tools, returns results to agent

### Key Files

- `src/react_agent/agent.py` - Core graph definition with LLM initialization and routing logic
- `src/react_agent/state.py` - AgentState TypedDict with messages list using `add_messages` reducer
- `src/react_agent/tools.py` - Tool definitions decorated with `@tool` (search_web, calculator)
- `main.py` - Entry point demonstrating agent usage

### Tool Integration

Tools are LangChain `@tool` decorated functions with docstrings for auto-documentation. The LLM is bound with tools via `bind_tools()` and decides which to call based on the query.

## Environment

Requires `GOOGLE_API_KEY` environment variable set in `.env` file.
