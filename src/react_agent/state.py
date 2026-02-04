"""Agent state definition for the ReAct agent."""

from typing import Annotated

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """State of the ReAct agent.

    Attributes:
        messages: List of messages in the conversation, managed by add_messages reducer.
    """

    messages: Annotated[list, add_messages]
