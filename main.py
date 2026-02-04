"""Main entry point for the ReAct Agent."""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from react_agent.agent import create_agent_graph


class ConversationHistory:
    """Manages conversation history for multi-turn interactions."""

    def __init__(self):
        self.messages: list = []
        self.start_time = datetime.now()

    def add_human_message(self, content: str) -> None:
        """Add a human message to history."""
        self.messages.append(HumanMessage(content=content))

    def add_ai_messages(self, messages: list) -> None:
        """Add AI response messages to history."""
        self.messages.extend(messages)

    def get_messages(self) -> list:
        """Get all messages in history."""
        return self.messages

    def clear(self) -> None:
        """Clear conversation history."""
        self.messages = []
        self.start_time = datetime.now()

    def export_to_file(self, filepath: str) -> None:
        """Export conversation to a JSON file."""
        export_data = {
            "start_time": self.start_time.isoformat(),
            "export_time": datetime.now().isoformat(),
            "messages": [],
        }

        for msg in self.messages:
            msg_data = {
                "type": type(msg).__name__,
                "content": msg.content if hasattr(msg, "content") else "",
            }
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                msg_data["tool_calls"] = msg.tool_calls
            export_data["messages"].append(msg_data)

        Path(filepath).write_text(json.dumps(export_data, ensure_ascii=False, indent=2))
        print(f"대화 내용이 {filepath}에 저장되었습니다.")


def print_separator(char: str = "=", length: int = 60) -> None:
    """Print a separator line."""
    print(char * length)


def print_message(msg, verbose: bool = False) -> None:
    """Print a message with appropriate formatting."""
    if isinstance(msg, HumanMessage):
        print(f"\n🧑 [사용자] {msg.content}")
    elif isinstance(msg, AIMessage):
        if msg.tool_calls and verbose:
            print("\n🔧 [도구 호출]")
            for tc in msg.tool_calls:
                print(f"   → {tc['name']}({tc['args']})")
        if msg.content:
            print(f"\n🤖 [에이전트] {msg.content}")
    elif isinstance(msg, ToolMessage) and verbose:
        content = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
        print(f"\n📋 [도구 결과] {content}")


def run_single_query(agent, query: str, history: ConversationHistory, verbose: bool = False) -> str:
    """Run a single query with conversation history."""
    print(f"\n{'='*60}")
    print(f"📝 질문: {query}")
    print("=" * 60)

    history.add_human_message(query)
    result = agent.invoke({"messages": history.get_messages()})

    # Get new messages (after the human message we just added)
    new_messages = result["messages"][len(history.messages) :]
    history.add_ai_messages(new_messages)

    if verbose:
        print("\n--- 메시지 흐름 ---")
        for msg in new_messages:
            print_message(msg, verbose=True)
    else:
        final_message = result["messages"][-1]
        print(f"\n💬 응답: {final_message.content}")

    return result["messages"][-1].content


def run_streaming(agent, query: str, history: ConversationHistory) -> str:
    """Run a query with streaming output."""
    print(f"\n{'='*60}")
    print(f"📝 질문: {query}")
    print("=" * 60)
    print("\n⏳ [처리 중...]")

    history.add_human_message(query)
    final_content = ""
    new_messages = []

    for step in agent.stream({"messages": history.get_messages()}):
        for node_name, output in step.items():
            if node_name == "agent":
                for msg in output.get("messages", []):
                    new_messages.append(msg)
                    if isinstance(msg, AIMessage):
                        if msg.tool_calls:
                            for tc in msg.tool_calls:
                                print(f"   🔧 도구 호출: {tc['name']}")
                        if msg.content:
                            final_content = msg.content
            elif node_name == "tools":
                for msg in output.get("messages", []):
                    new_messages.append(msg)
                    if isinstance(msg, ToolMessage):
                        print("   📋 도구 결과 수신")

    history.add_ai_messages(new_messages)
    print(f"\n💬 [응답]\n{final_content}")
    return final_content


def run_interactive(agent, verbose: bool = False, streaming: bool = False) -> None:
    """Run the agent in interactive chat mode with conversation history."""
    history = ConversationHistory()

    print("\n" + "=" * 60)
    print("  🤖 LangGraph ReAct Agent - 대화형 모드")
    print("=" * 60)
    print("\n📌 명령어:")
    print("   /help     - 도움말 표시")
    print("   /verbose  - 상세 모드 토글")
    print("   /stream   - 스트리밍 모드 토글")
    print("   /clear    - 대화 기록 초기화")
    print("   /history  - 대화 기록 보기")
    print("   /export   - 대화 내용 저장")
    print("   /quit     - 종료")
    print("-" * 60)
    print(f"스트리밍: {'켜짐' if streaming else '꺼짐'} | 상세 모드: {'켜짐' if verbose else '꺼짐'}")
    print("-" * 60)

    while True:
        try:
            query = input("\n🧑 질문: ").strip()

            if not query:
                continue

            # Command handling
            if query.startswith("/"):
                cmd = query.lower()

                if cmd == "/quit" or cmd == "/exit" or cmd == "/q":
                    print("\n👋 대화를 종료합니다.")
                    break

                elif cmd == "/help":
                    print("\n📌 사용 가능한 명령어:")
                    print("   /help     - 이 도움말 표시")
                    print("   /verbose  - 상세 모드 토글 (도구 호출 과정 표시)")
                    print("   /stream   - 스트리밍 모드 토글 (실시간 진행 표시)")
                    print("   /clear    - 대화 기록 초기화")
                    print("   /history  - 현재 대화 기록 보기")
                    print("   /export   - 대화 내용을 파일로 저장")
                    print("   /quit     - 대화 종료")
                    continue

                elif cmd == "/verbose":
                    verbose = not verbose
                    print(f"🔧 상세 모드: {'켜짐' if verbose else '꺼짐'}")
                    continue

                elif cmd == "/stream":
                    streaming = not streaming
                    print(f"⏳ 스트리밍 모드: {'켜짐' if streaming else '꺼짐'}")
                    continue

                elif cmd == "/clear":
                    history.clear()
                    print("🗑️ 대화 기록이 초기화되었습니다.")
                    continue

                elif cmd == "/history":
                    if not history.messages:
                        print("📭 대화 기록이 비어있습니다.")
                    else:
                        print(f"\n📜 대화 기록 ({len(history.messages)}개 메시지):")
                        print("-" * 40)
                        for msg in history.messages:
                            print_message(msg, verbose=True)
                    continue

                elif cmd.startswith("/export"):
                    parts = cmd.split()
                    filename = parts[1] if len(parts) > 1 else f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    history.export_to_file(filename)
                    continue

                else:
                    print(f"❓ 알 수 없는 명령어: {cmd}")
                    print("   /help 로 사용 가능한 명령어를 확인하세요.")
                    continue

            # Run query
            if streaming:
                run_streaming(agent, query, history)
            else:
                run_single_query(agent, query, history, verbose=verbose)

        except KeyboardInterrupt:
            print("\n\n👋 대화를 종료합니다.")
            break


def run_demo(agent) -> None:
    """Run demo queries to showcase agent capabilities."""
    history = ConversationHistory()

    demo_queries = [
        ("🔍 웹 검색", "LangGraph에 대해 검색해줘"),
        ("🧮 계산기", "157 * 23 + 89를 계산해줘"),
        ("🕐 시간 확인", "지금 몇 시야?"),
        ("💬 일반 대화", "안녕! 너는 어떤 도구들을 사용할 수 있어?"),
    ]

    print("\n" + "=" * 60)
    print("  🎮 LangGraph ReAct Agent - 데모 모드")
    print("=" * 60)

    for category, query in demo_queries:
        print(f"\n[{category}]")
        run_streaming(agent, query, history)
        history.clear()  # Reset for each demo
        print()


def main() -> None:
    """Run the ReAct agent."""
    parser = argparse.ArgumentParser(
        description="🤖 LangGraph ReAct Agent",
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
        print(f"❌ 오류: {e}")
        print("💡 GOOGLE_API_KEY가 .env 파일에 설정되어 있는지 확인하세요.")
        sys.exit(1)

    # Execute based on mode
    if args.demo:
        run_demo(agent)
    elif args.query:
        history = ConversationHistory()
        if args.stream:
            run_streaming(agent, args.query, history)
        else:
            run_single_query(agent, args.query, history, verbose=args.verbose)
    else:
        run_interactive(agent, verbose=args.verbose, streaming=args.stream)


if __name__ == "__main__":
    main()
