"""ReAct Agent implementation using LangGraph."""

import os
from typing import Literal

from langchain_core.messages import AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from react_agent.state import AgentState
from react_agent.tools import calculator, search_web

# Define available tools
tools = [search_web, calculator]


def create_llm() -> ChatGoogleGenerativeAI:
    """Create and configure the LLM instance."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set")

    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=api_key,
        temperature=0,
    )


def agent_node(state: AgentState) -> dict:
    """Process the current state and generate a response.

    Args:
        state: Current agent state containing messages.

    Returns:
        Updated state with new AI message.
    """
    llm = create_llm()
    llm_with_tools = llm.bind_tools(tools)
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    """Determine whether to continue with tools or end the conversation.

    Args:
        state: Current agent state.

    Returns:
        "tools" if the last message has tool calls, otherwise END.
    """
    last_message = state["messages"][-1]

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"

    return END


def create_agent_graph() -> StateGraph:
    """Create the ReAct agent graph.

    Returns:
        Compiled StateGraph for the ReAct agent.
    """
    # Create the graph
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))

    # Set entry point
    graph.set_entry_point("agent")

    # Add conditional edge from agent
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            END: END,
        },
    )

    # Add edge from tools back to agent
    graph.add_edge("tools", "agent")

    return graph.compile()
