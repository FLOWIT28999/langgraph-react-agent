"""Tools for the ReAct agent."""

from langchain_core.tools import tool


@tool
def search_web(query: str) -> str:
    """Search the web for information.

    Args:
        query: The search query string.

    Returns:
        Mock search results as a string.
    """
    # Mock implementation for testing purposes
    mock_results = {
        "langgraph": "LangGraph is a library for building stateful, multi-actor applications with LLMs.",
        "react agent": "ReAct (Reasoning and Acting) is a paradigm that combines reasoning and action in LLM agents.",
        "gemini api": "Google Gemini API provides access to Google's multimodal AI models.",
    }

    query_lower = query.lower()
    for keyword, result in mock_results.items():
        if keyword in query_lower:
            return f"Search results for '{query}': {result}"

    return f"Search results for '{query}': No specific results found. This is a mock search tool."


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression.

    Args:
        expression: A mathematical expression to evaluate (e.g., "2 + 2", "10 * 5").

    Returns:
        The result of the calculation as a string.
    """
    try:
        # Only allow safe mathematical operations
        allowed_chars = set("0123456789+-*/(). ")
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in expression. Only numbers and basic operators (+, -, *, /, parentheses) are allowed."

        result = eval(expression)  # noqa: S307
        return f"Result: {expression} = {result}"
    except ZeroDivisionError:
        return "Error: Division by zero"
    except Exception as e:
        return f"Error: Could not evaluate expression. {e}"
