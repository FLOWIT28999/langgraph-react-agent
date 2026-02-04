"""Main entry point for the ReAct Agent."""

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from react_agent.agent import create_agent_graph


def main() -> None:
    """Run the ReAct agent with a sample query."""
    # Load environment variables
    load_dotenv()

    # Create the agent graph
    agent = create_agent_graph()

    # Example queries to test the agent
    test_queries = [
        "What is LangGraph?",
        "Calculate 25 * 4 + 10",
        "Search for information about the ReAct agent pattern",
    ]

    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"Query: {query}")
        print("=" * 50)

        # Run the agent
        result = agent.invoke({"messages": [HumanMessage(content=query)]})

        # Print the final response
        final_message = result["messages"][-1]
        print(f"\nAgent Response:\n{final_message.content}")


if __name__ == "__main__":
    main()
