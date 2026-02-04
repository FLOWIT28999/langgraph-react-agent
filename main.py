"""Main entry point for the ReAct Agent."""

import argparse
import sys

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from react_agent.agent import create_agent_graph


def print_message(msg, verbose: bool = False) -> None:
    """Print a message with appropriate formatting."""
    if isinstance(msg, HumanMessage):
        print(f"\n[사용자] {msg.content}")
    elif isinstance(msg, AIMessage):
        if msg.tool_calls and verbose:
            print(f"\n[도구 호출]")
            for tc in msg.tool_calls:
                print(f"  → {tc['name']}({tc['args']})")
        if msg.content:
            print(f"\n[에이전트] {msg.content}")
    elif isinstance(msg, ToolMessage) and verbose:
        print(f"\n[도구 결과] {msg.content[:100]}{'...' if len(msg.content) > 100 else ''}")


def run_single_query(agent, query: str, verbose: bool = False) -> str:
    """Run a single query and return the response."""
    print(f"\n{'='*60}")
    print(f"질문: {query}")
    print("=" * 60)

    result = agent.invoke({"messages": [HumanMessage(content=query)]})

    if verbose:
        print("\n--- 전체 메시지 흐름 ---")
        for msg in result["messages"]:
            print_message(msg, verbose=True)

    final_message = result["messages"][-1]
    if not verbose:
        print(f"\n응답: {final_message.content}")

    return final_message.content


def run_streaming(agent, query: str) -> str:
    """Run a query with streaming output to show progress."""
    print(f"\n{'='*60}")
    print(f"질문: {query}")
    print("=" * 60)

    print("\n[처리 중...]")

    final_content = ""
    for step in agent.stream({"messages": [HumanMessage(content=query)]}):
        for node_name, output in step.items():
            if node_name == "agent":
                for msg in output.get("messages", []):
                    if isinstance(msg, AIMessage):
                        if msg.tool_calls:
                            for tc in msg.tool_calls:
                                print(f"  → 도구 호출: {tc['name']}")
                        if msg.content:
                            final_content = msg.content
            elif node_name == "tools":
                for msg in output.get("messages", []):
                    if isinstance(msg, ToolMessage):
                        print(f"  ← 도구 결과 수신")

    print(f"\n[응답]\n{final_content}")
    return final_content


def run_interactive(agent, verbose: bool = False, streaming: bool = False) -> None:
    """Run the agent in interactive chat mode."""
    print("\n" + "=" * 60)
    print("  LangGraph ReAct Agent - 대화형 모드")
    print("=" * 60)
    print("종료하려면 'quit', 'exit', 또는 'q'를 입력하세요.")
    print("스트리밍 모드:", "켜짐" if streaming else "꺼짐")
    print("상세 모드:", "켜짐" if verbose else "꺼짐")
    print("-" * 60)

    while True:
        try:
            query = input("\n질문: ").strip()

            if not query:
                continue

            if query.lower() in ["quit", "exit", "q"]:
                print("\n대화를 종료합니다.")
                break

            if query.lower() == "/help":
                print("\n사용 가능한 명령어:")
                print("  /help     - 도움말 표시")
                print("  /verbose  - 상세 모드 토글")
                print("  /stream   - 스트리밍 모드 토글")
                print("  quit      - 종료")
                continue

            if query.lower() == "/verbose":
                verbose = not verbose
                print(f"상세 모드: {'켜짐' if verbose else '꺼짐'}")
                continue

            if query.lower() == "/stream":
                streaming = not streaming
                print(f"스트리밍 모드: {'켜짐' if streaming else '꺼짐'}")
                continue

            if streaming:
                run_streaming(agent, query)
            else:
                run_single_query(agent, query, verbose=verbose)

        except KeyboardInterrupt:
            print("\n\n대화를 종료합니다.")
            break


def run_demo(agent) -> None:
    """Run demo queries to showcase agent capabilities."""
    demo_queries = [
        ("웹 검색", "LangGraph에 대해 검색해줘"),
        ("계산", "157 * 23 + 89를 계산해줘"),
        ("일반 대화", "안녕! 오늘 기분이 어때?"),
    ]

    print("\n" + "=" * 60)
    print("  LangGraph ReAct Agent - 데모 모드")
    print("=" * 60)

    for category, query in demo_queries:
        print(f"\n[데모: {category}]")
        run_streaming(agent, query)
        print()


def main() -> None:
    """Run the ReAct agent."""
    parser = argparse.ArgumentParser(
        description="LangGraph ReAct Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python main.py                      # 대화형 모드
  python main.py -q "질문 내용"        # 단일 질문
  python main.py --demo               # 데모 실행
  python main.py -q "질문" --verbose  # 상세 출력
  python main.py --stream             # 스트리밍 모드로 대화
        """,
    )
    parser.add_argument(
        "-q", "--query",
        type=str,
        help="실행할 단일 질문",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="상세 출력 모드 (도구 호출 과정 표시)",
    )
    parser.add_argument(
        "-s", "--stream",
        action="store_true",
        help="스트리밍 모드 (실시간 진행 상황 표시)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="데모 쿼리 실행",
    )

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Create the agent graph
    try:
        agent = create_agent_graph()
    except ValueError as e:
        print(f"오류: {e}")
        print("GOOGLE_API_KEY가 .env 파일에 설정되어 있는지 확인하세요.")
        sys.exit(1)

    # Execute based on mode
    if args.demo:
        run_demo(agent)
    elif args.query:
        if args.stream:
            run_streaming(agent, args.query)
        else:
            run_single_query(agent, args.query, verbose=args.verbose)
    else:
        run_interactive(agent, verbose=args.verbose, streaming=args.stream)


if __name__ == "__main__":
    main()
