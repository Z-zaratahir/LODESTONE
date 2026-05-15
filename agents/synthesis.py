"""
LODESTONE — agents/synthesis.py
==================================
Stage 6: The Synthesis Agent node — the only agent that talks to the user.

Responsibilities:
  1. Read all research findings from state
  2. Read conversation history for context-aware responses
  3. Call LLM to generate a structured, user-friendly response
  4. Extract suggested follow-up questions from the response
  5. Append the final response to conversation history
  6. Return final_response and suggested_followups to state

Design decision — Synthesis is the ONLY user-facing node:
  Every other agent writes to state. Only Synthesis writes final_response.
  This creates a clean separation: agents reason internally, Synthesis speaks.
  It also means the conversation history always contains the exact text
  the user saw — no intermediate agent outputs pollute it.

Follow-up question extraction:
  The synthesis prompt asks the LLM to embed follow-up questions in a
  ```followups ... ``` code block at the end of its response.
  We parse this out, store it in suggested_followups, and strip it from
  the user-visible response. This keeps the response clean while giving
  the UI something structured to render as buttons.
"""

import json
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from state import AgentState
from prompts import synthesis_system_prompt, synthesis_prompt
from memory.entity_store import get_company_for_search
from config import MODELS

logger = logging.getLogger(__name__)


def _get_llm():
    model = MODELS.get("primary", "llama3-8b-8192")

    try:
        from langchain_groq import ChatGroq
        from config import GROQ_API_KEY
        if GROQ_API_KEY:
            return ChatGroq(model=model, temperature=0.3)  # slight creativity for synthesis
    except (ImportError, Exception):
        pass

    try:
        from langchain_openai import ChatOpenAI
        from config import OPENAI_API_KEY
        if OPENAI_API_KEY:
            return ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    except (ImportError, Exception):
        pass

    try:
        from langchain_anthropic import ChatAnthropic
        from config import ANTHROPIC_API_KEY
        if ANTHROPIC_API_KEY:
            return ChatAnthropic(model="claude-haiku-20240307", temperature=0.3)
    except (ImportError, Exception):
        pass

    raise RuntimeError("No LLM provider configured. Check your .env file.")


def run_synthesis_agent(state: AgentState) -> dict[str, Any]:
    """
    Synthesis Agent node function.

    Reads all accumulated research state, generates the final user-facing
    response, and appends it to conversation history.
    """
    current_turn     = state.get("current_turn", 1)
    entity_memory    = state.get("entity_memory") or {}
    sub_intents      = state.get("sub_intents") or []
    research_summary = state.get("_research_summary") or {}
    validator_notes  = state.get("validator_notes")
    source_conflict  = state.get("source_conflict", False)
    confidence_score = state.get("confidence_score", 0.0)
    conversation_history = state.get("conversation_history") or []

    company = entity_memory.get("company_name", "the company")
    company_search_str = get_company_for_search(entity_memory) or company

    # Get original query from history
    original_query = _extract_original_query(conversation_history, current_turn)

    logger.info(
        f"[SYNTHESIS] Turn {current_turn} | "
        f"Company: {company_search_str} | "
        f"Confidence: {confidence_score}"
    )

    # ── LLM synthesis call ────────────────────────────────────────────────────
    llm = _get_llm()

    prompt_text = synthesis_prompt(
        original_query=original_query,
        company=company_search_str,
        sub_intents=sub_intents,
        research_summary=research_summary,
        validator_notes=validator_notes,
        source_conflict=source_conflict,
        confidence_score=confidence_score,
        conversation_history=conversation_history,
    )

    try:
        response = llm.invoke([
            SystemMessage(content=synthesis_system_prompt()),
            HumanMessage(content=prompt_text),
        ])
        raw_output = response.content.strip()
        logger.debug(f"[SYNTHESIS] Raw output length: {len(raw_output)} chars")
    except Exception as e:
        logger.error(f"[SYNTHESIS] LLM call failed: {e}")
        raw_output = _fallback_response(company, research_summary, sub_intents)

    # ── Extract follow-up questions ───────────────────────────────────────────
    final_response, suggested_followups = _extract_followups(raw_output)

    logger.info(
        f"[SYNTHESIS] Response: {len(final_response)} chars | "
        f"Follow-ups: {len(suggested_followups)}"
    )

    # Append final response to conversation history (operator.add will merge)
    history_addition = [{
        "role": "assistant",
        "content": final_response,
        "turn": current_turn,
    }]

    return {
        "final_response":     final_response,
        "suggested_followups": suggested_followups,
        "conversation_history": history_addition,
    }


# ── Follow-up extraction ──────────────────────────────────────────────────────

def _extract_followups(raw_response: str) -> tuple[str, list[str]]:
    """
    Extract the ```followups ... ``` block from the LLM response.

    Returns:
        (clean_response_text, list_of_followup_strings)

    The LLM is instructed to put follow-ups in a ```followups ... ``` block.
    We extract it, parse the JSON array inside, and strip it from the response.
    """
    pattern = re.compile(
        r'```followups\s*\n(.*?)\n```',
        re.DOTALL | re.IGNORECASE,
    )
    match = pattern.search(raw_response)

    if not match:
        # No block found — return raw response with no follow-ups
        return raw_response.strip(), []

    followups_raw = match.group(1).strip()
    clean_response = raw_response[:match.start()].strip()

    try:
        followups = json.loads(followups_raw)
        if isinstance(followups, list):
            return clean_response, [str(f) for f in followups[:3]]
    except json.JSONDecodeError:
        # Parse failed — just skip follow-ups
        logger.warning("[SYNTHESIS] Follow-up JSON parse failed")

    return clean_response, []


def _extract_original_query(history: list[dict], current_turn: int) -> str:
    """Extract the user's message for the current turn."""
    for msg in reversed(history):
        if msg.get("role") == "user" and msg.get("turn") == current_turn:
            return msg.get("content", "")
    for msg in reversed(history):
        if msg.get("role") == "user":
            return msg.get("content", "")
    return "your research query"


def _fallback_response(company: str, research_summary: dict, sub_intents: list) -> str:
    """Minimal fallback if LLM synthesis fails entirely."""
    intent_summaries = research_summary.get("intent_summaries", {})
    parts = [f"Here is what I found about **{company}**:\n"]

    for intent in sub_intents:
        summary = intent_summaries.get(intent, "")
        if summary and summary != "No relevant results found":
            parts.append(f"**{intent.title()}**: {summary[:400]}\n")

    if len(parts) == 1:
        parts.append("I was unable to retrieve sufficient information at this time.")

    return "\n".join(parts)
