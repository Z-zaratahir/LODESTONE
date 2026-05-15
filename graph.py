"""
LODESTONE — graph.py
=====================
Builds and compiles the LangGraph state machine.

This file is the architectural blueprint made executable.
Every node, edge, conditional branch, and interrupt defined here corresponds
directly to a box or arrow in the system diagram.

Graph structure:
  START
    → normalizer_node          (pre-processing, no LLM)
    → clarity_agent            (LLM: intent + entity extraction)
        ├── [needs_clarification] → INTERRUPT → (user responds) → clarity_agent
        └── [clear]            → research_agent
    → research_agent           (tool call + LLM assembly)
        ├── [confidence ≥ 6]   → synthesis_agent
        └── [confidence < 6]   → validator_agent
    → validator_agent          (LLM: quality check + source triangulation)
        ├── [sufficient OR max_attempts] → synthesis_agent
        └── [insufficient]     → research_agent  (loop, max 3 times)
    → synthesis_agent          (LLM: final response)
    → END

Interrupt mechanism:
  LangGraph's interrupt_before is set on the human_feedback node.
  When clarity_status = "needs_clarification", the graph routes to
  human_feedback which calls interrupt() — this halts execution and
  returns control to main.py, which surfaces the clarification question
  to the user, collects their response, then resumes the graph with
  the new input merged into state.

State persistence:
  MemorySaver checkpointer saves state between interrupt/resume cycles.
  This is what allows multi-turn conversations — the graph "remembers"
  where it paused and what was in state when it halted.
"""

import logging
from typing import Literal

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt

from state import AgentState
from agents.clarity   import run_clarity_agent
from agents.research  import run_research_agent
from agents.validator import run_validator_agent
from agents.synthesis import run_synthesis_agent
from config import CONFIDENCE_THRESHOLD, MAX_RESEARCH_RETRIES

logger = logging.getLogger(__name__)


# ── Normalizer node ───────────────────────────────────────────────────────────

def normalizer_node(state: AgentState) -> dict:
    """
    Pre-processing node. Runs before any LLM.
    Calls the two-layer normalizer (rules + optional HF spell correction).
    Also increments current_turn and appends raw input to conversation history.
    """
    from tools.normalizer import normalize

    raw = state.get("raw_input", "")
    normalized = normalize(raw)

    current_turn = state.get("current_turn", 0) + 1

    logger.info(f"[NORMALIZER] Turn {current_turn} | Raw: {repr(raw[:60])} → Normalized: {repr(normalized[:60])}")

    # Add user message to conversation history
    history_entry = [{
        "role": "user",
        "content": normalized,   # store normalized version (LLMs read this)
        "turn": current_turn,
        "raw": raw,              # keep raw for auditability
    }]

    return {
        "raw_input":          raw,
        "normalized_input":   normalized,
        "current_turn":       current_turn,
        "conversation_history": history_entry,  # appended via operator.add
        # Reset per-turn fields
        "clarity_status":     "",
        "clarification_question": None,
        "sub_intents":        [],
        "research_results":   [],
        "confidence_score":   0.0,
        "source_conflict":    False,
        "validation_result":  "",
        "validator_notes":    None,
        "final_response":     "",
        "suggested_followups": [],
        "_research_summary":  {},
    }


# ── Human feedback node (interrupt point) ─────────────────────────────────────

def human_feedback_node(state: AgentState) -> dict:
    """
    Interrupt node. Called when clarity_status = "needs_clarification".

    interrupt() halts graph execution here. main.py detects the halt,
    surfaces the clarification_question to the user, collects their
    response, and calls graph.invoke() again with the new input.

    The new input is merged into state as raw_input, and the graph
    resumes from the normalizer_node (re-normalizes → re-runs clarity).

    Note: We don't just resume at clarity_agent because the user's
    clarification may itself need normalization.
    """
    question = state.get("clarification_question", "Could you clarify your question?")

    logger.info(f"[INTERRUPT] Halting for clarification: {question}")

    # This call halts the graph. The value passed to interrupt() is returned
    # to the caller of graph.stream() / graph.invoke() as the interrupt value.
    user_response = interrupt({
        "question": question,
        "turn": state.get("current_turn", 1),
    })

    # When the graph resumes, user_response contains the user's clarification.
    # We inject it as the new raw_input so normalizer re-processes it.
    logger.info(f"[INTERRUPT] Resumed with: {repr(str(user_response)[:60])}")

    return {
        "raw_input": str(user_response),
    }


# ── Conditional routing functions ─────────────────────────────────────────────

def route_after_clarity(state: AgentState) -> Literal["human_feedback", "research_agent"]:
    """
    Route after Clarity Agent:
      - needs_clarification → human_feedback (interrupt)
      - clear               → research_agent
    """
    status = state.get("clarity_status", "needs_clarification")

    if status == "needs_clarification":
        logger.info("[ROUTE] clarity → human_feedback (interrupt)")
        return "human_feedback"

    logger.info("[ROUTE] clarity → research_agent")
    return "research_agent"


def route_after_research(state: AgentState) -> Literal["validator_agent", "synthesis_agent"]:
    """
    Route after Research Agent:
      - confidence < THRESHOLD → validator_agent (quality check needed)
      - confidence ≥ THRESHOLD → synthesis_agent (good enough to synthesize)
    """
    confidence = state.get("confidence_score", 0.0)

    if confidence < CONFIDENCE_THRESHOLD:
        logger.info(f"[ROUTE] research → validator (confidence={confidence} < {CONFIDENCE_THRESHOLD})")
        return "validator_agent"

    logger.info(f"[ROUTE] research → synthesis (confidence={confidence} ≥ {CONFIDENCE_THRESHOLD})")
    return "synthesis_agent"


def route_after_validator(
    state: AgentState,
) -> Literal["research_agent", "synthesis_agent"]:
    """
    Route after Validator Agent:
      - insufficient + attempts < max → research_agent (retry loop)
      - sufficient OR max attempts    → synthesis_agent
    """
    validation_result  = state.get("validation_result", "sufficient")
    research_attempts  = state.get("research_attempts", 1)

    if validation_result == "insufficient" and research_attempts < MAX_RESEARCH_RETRIES:
        logger.info(
            f"[ROUTE] validator → research (insufficient, attempt {research_attempts}/{MAX_RESEARCH_RETRIES})"
        )
        return "research_agent"

    if research_attempts >= MAX_RESEARCH_RETRIES:
        logger.info(f"[ROUTE] validator → synthesis (max attempts reached: {research_attempts})")
    else:
        logger.info("[ROUTE] validator → synthesis (sufficient)")

    return "synthesis_agent"


def route_after_human_feedback(state: AgentState) -> Literal["normalizer_node"]:
    """
    After human feedback, always re-normalize and re-run clarity.
    The user's clarification may itself need cleanup.
    """
    return "normalizer_node"


# ── Graph construction ────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """
    Construct the LangGraph StateGraph.
    Returns the compiled graph with MemorySaver checkpointer.

    The checkpointer is what enables:
      1. Interrupt/resume (state is persisted across halts)
      2. Multi-turn conversations (state survives between top-level calls)
      3. Debugging (you can inspect state at any checkpoint)
    """
    builder = StateGraph(AgentState)

    # ── Add nodes ─────────────────────────────────────────────────────────────
    builder.add_node("normalizer_node",    normalizer_node)
    builder.add_node("clarity_agent",      run_clarity_agent)
    builder.add_node("human_feedback",     human_feedback_node)
    builder.add_node("research_agent",     run_research_agent)
    builder.add_node("validator_agent",    run_validator_agent)
    builder.add_node("synthesis_agent",    run_synthesis_agent)

    # ── Add edges ─────────────────────────────────────────────────────────────
    builder.add_edge(START, "normalizer_node")
    builder.add_edge("normalizer_node", "clarity_agent")

    # Clarity → conditional (interrupt OR research)
    builder.add_conditional_edges(
        "clarity_agent",
        route_after_clarity,
        {
            "human_feedback": "human_feedback",
            "research_agent": "research_agent",
        }
    )

    # Human feedback → back to normalizer (re-normalize the clarification)
    builder.add_edge("human_feedback", "normalizer_node")

    # Research → conditional (validator OR synthesis)
    builder.add_conditional_edges(
        "research_agent",
        route_after_research,
        {
            "validator_agent": "validator_agent",
            "synthesis_agent": "synthesis_agent",
        }
    )

    # Validator → conditional (research loop OR synthesis)
    builder.add_conditional_edges(
        "validator_agent",
        route_after_validator,
        {
            "research_agent": "research_agent",
            "synthesis_agent": "synthesis_agent",
        }
    )

    # Synthesis → END
    builder.add_edge("synthesis_agent", END)

    # ── Compile with memory checkpointer ─────────────────────────────────────
    checkpointer = MemorySaver()
    graph = builder.compile(checkpointer=checkpointer)

    logger.info("[GRAPH] Compiled successfully")
    return graph


# ── Module-level compiled graph (singleton) ───────────────────────────────────
# Import this in main.py: `from graph import lodestone_graph`

lodestone_graph = build_graph()
