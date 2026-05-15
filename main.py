"""
LODESTONE — main.py
====================
Entry point. Runs the multi-turn conversation loop.

Handles:
  1. Initial state setup for each session
  2. Processing user input through the graph
  3. Interrupt detection and resume (clarification cycle)
  4. Rendering the final response and follow-up suggestions
  5. Graceful exit

Interrupt/Resume cycle:
  When the Clarity Agent flags a query as ambiguous:
    1. Graph pauses at human_feedback_node
    2. main.py catches the interrupt, prints the clarification question
    3. User types their answer
    4. main.py calls graph.invoke() with the new input + Command(resume=answer)
    5. Graph resumes from human_feedback_node, routes back to normalizer
    6. Processing continues normally

Thread ID:
  Each session gets a unique thread_id. LangGraph's MemorySaver uses this
  to store/retrieve the checkpoint for that conversation.
  This is how multi-turn context works — same thread_id = same conversation.

Usage:
    python main.py
    python main.py --debug      (enables DEBUG logging)
    python main.py --no-hf      (disables HuggingFace spell correction)
"""

import argparse
import logging
import sys
import uuid
from typing import Any, Optional

from langgraph.types import Command


def setup_logging(debug: bool = False) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        level=level,
        stream=sys.stdout,
    )
    # Silence noisy third-party loggers unless in debug mode
    if not debug:
        for name in ("httpx", "httpcore", "openai", "anthropic", "groq", "tavily"):
            logging.getLogger(name).setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LODESTONE — Multi-Agent Business Research Assistant"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG logging",
    )
    parser.add_argument(
        "--no-hf",
        action="store_true",
        help="Disable HuggingFace spell correction (faster startup)",
    )
    parser.add_argument(
        "--thread-id",
        type=str,
        default=None,
        help="Resume a specific conversation thread by ID",
    )
    return parser.parse_args()


def print_banner() -> None:
    print("\n" + "═" * 60)
    print("  LODESTONE — Business Research Assistant")
    print("  Multi-Agent System | LangGraph + Tavily")
    print("═" * 60)
    print("  Type your query. Type 'exit' or 'quit' to stop.")
    print("  Type 'new' to start a fresh conversation.")
    print("═" * 60 + "\n")


def print_response(response: str, followups: list[str]) -> None:
    """Render the final response and follow-up suggestions."""
    print("\n" + "─" * 60)
    print("  LODESTONE")
    print("─" * 60)
    print(response)

    if followups:
        print("\n" + "─" * 40)
        print("  Suggested follow-ups:")
        for i, q in enumerate(followups, 1):
            print(f"  {i}. {q}")
    print("─" * 60 + "\n")


def get_initial_state() -> dict:
    """Build the minimal initial state for a new conversation."""
    return {
        "conversation_history": [],
        "current_turn": 0,
        "raw_input": "",
        "normalized_input": "",
        "clarity_status": "",
        "clarification_question": None,
        "sub_intents": [],
        "entity_memory": {
            "company_name": None,
            "ticker": None,
            "last_turn": None,
        },
        "research_results": [],
        "confidence_score": 0.0,
        "source_conflict": False,
        "validation_result": "",
        "research_attempts": 0,
        "validator_notes": None,
        "final_response": "",
        "suggested_followups": [],
        "assembled_research": {},
    }


def run_conversation_turn(
    graph,
    user_input: str,
    thread_id: str,
    initial_state: Optional[dict] = None,
) -> tuple[str, list[str]]:
    """
    Process one user input through the full graph.

    Handles the interrupt/resume cycle internally.

    Returns:
        (final_response, suggested_followups)
    """
    config = {"configurable": {"thread_id": thread_id}}

    # For the first turn, we need to pass the full initial state + input
    # For subsequent turns, LangGraph loads state from the checkpoint
    invoke_input = {"raw_input": user_input}
    if initial_state:
        invoke_input = {**initial_state, "raw_input": user_input}

    logger.debug(f"Invoking graph | thread={thread_id} | input={repr(user_input[:60])}")

    # ── Interrupt/resume loop ─────────────────────────────────────────────────
    # The graph may interrupt multiple times (unlikely, but handle it).
    max_interrupts = 3
    interrupt_count = 0

    result = None

    while interrupt_count < max_interrupts:
        try:
            # stream() lets us detect interrupts mid-execution
            final_values = None

            for event in graph.stream(
                invoke_input,
                config=config,
                stream_mode="values",
            ):
                final_values = event

            # No interrupt occurred — execution completed
            result = final_values
            break

        except Exception as e:
            # Check if this is a LangGraph interrupt
            error_name = type(e).__name__
            if "GraphInterrupt" in error_name or "Interrupt" in error_name:
                interrupt_count += 1

                # Extract the interrupt value (the clarification question)
                interrupt_value = _extract_interrupt_value(e)
                question = interrupt_value.get("question", "Could you clarify?") \
                    if isinstance(interrupt_value, dict) else str(interrupt_value)

                print(f"\n  🔍 {question}")
                print("  → ", end="", flush=True)

                try:
                    user_clarification = input().strip()
                except (EOFError, KeyboardInterrupt):
                    user_clarification = "cancel"

                if not user_clarification or user_clarification.lower() in ("cancel", "exit", "quit"):
                    return "Query cancelled.", []

                # Resume graph with the user's clarification
                invoke_input = Command(resume=user_clarification)
                logger.debug(f"Resuming with: {repr(user_clarification[:60])}")

            else:
                # Not an interrupt — real error
                logger.error(f"Graph execution error: {e}", exc_info=True)
                return (
                    f"I encountered an error while processing your request: {str(e)[:200]}",
                    [],
                )

    if result is None:
        return "Processing did not complete. Please try again.", []

    final_response   = result.get("final_response", "")
    suggested_followups = result.get("suggested_followups", [])

    return final_response, suggested_followups


def _extract_interrupt_value(exc: Exception) -> Any:
    """
    Extract the interrupt value from a GraphInterrupt exception.
    LangGraph stores it differently across versions — handle both.
    """
    # LangGraph >= 0.2: exception has .interrupts attribute
    if hasattr(exc, "interrupts"):
        interrupts = exc.interrupts
        if interrupts:
            interrupt_obj = interrupts[0]
            if hasattr(interrupt_obj, "value"):
                return interrupt_obj.value
            return interrupt_obj

    # Fallback: try args
    if exc.args:
        return exc.args[0]

    return {}


def main() -> None:
    args = parse_args()
    setup_logging(args.debug)

    # Disable HF model if requested (faster startup for demos)
    if args.no_hf:
        import config
        config.MODELS["normalizer_hf"] = None  # type: ignore[assignment]
        logger.info("HuggingFace spell correction disabled")

    print_banner()

    # Import graph after logging is set up (graph.py logs on import)
    from graph import lodestone_graph as graph

    # Session state
    thread_id     = args.thread_id or str(uuid.uuid4())
    initial_state = get_initial_state()
    first_turn    = True

    if args.thread_id:
        print(f"  Resuming conversation: {thread_id}\n")
        first_turn = False
    else:
        print(f"  Session ID: {thread_id}\n")

    # ── Main conversation loop ─────────────────────────────────────────────────
    while True:
        try:
            print("You: ", end="", flush=True)
            user_input = input().strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n  Session ended.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit", "bye"):
            print("\n  Session ended. Goodbye.\n")
            break

        if user_input.lower() == "new":
            thread_id     = str(uuid.uuid4())
            initial_state = get_initial_state()
            first_turn    = True
            print(f"\n  New session started: {thread_id}\n")
            continue

        if user_input.lower() == "thread":
            print(f"\n  Current session ID: {thread_id}\n")
            continue

        # Process the turn
        print("  Processing...", flush=True)

        response, followups = run_conversation_turn(
            graph=graph,
            user_input=user_input,
            thread_id=thread_id,
            initial_state=initial_state if first_turn else None,
        )

        print_response(response, followups)

        first_turn = False


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    main()
