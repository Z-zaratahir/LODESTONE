"""
LODESTONE — agents/validator.py
=================================
Stage 5: The Validator Agent node.

Responsibilities:
  1. Assess quality and completeness of research results
  2. Perform source triangulation — do sources agree?
  3. Identify specific gaps (fed back to Research Agent on retry)
  4. Enforce the max-retry hard cap (attempt 3 → always route to Synthesis)
  5. Return validation_result + validator_notes to state

The Validator is what separates LODESTONE from a naive "search and summarize"
pipeline. Most systems just dump search results into Synthesis. The Validator
asks: "Is this good enough to actually answer the question?"

Source triangulation detail:
  The Validator receives source_conflict (bool) from the Research Agent.
  It independently checks source_agreement and compares against the flag.
  If both agree there's a conflict, the note goes into validator_notes
  AND gets passed to Synthesis with a caveat instruction.

Design decision — why the LLM decides sufficiency, not just a rule:
  A rule like "at least 2 sources = sufficient" misses cases where
  the sources are all about a different company (a name collision),
  or all repeat the same single fact. The LLM reads the actual content
  and makes a judgment — the rule-based confidence score is a guide,
  not the verdict.
"""

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage

from state import AgentState
from prompts import validator_prompt
from config import MODELS, MAX_RESEARCH_RETRIES

logger = logging.getLogger(__name__)


def _get_llm():
    """Load Groq LLM client (free tier — llama3-8b-8192)."""
    from config import GROQ_API_KEY
    model = MODELS.get("primary", "llama3-8b-8192")

    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to your .env file. "
            "Get a free key at https://console.groq.com"
        )

    try:
        from langchain_groq import ChatGroq
        return ChatGroq(model=model, temperature=0)
    except ImportError:
        raise RuntimeError(
            "langchain-groq is not installed. Run: pip install langchain-groq"
        )


def run_validator_agent(state: AgentState) -> dict[str, Any]:
    """
    Validator Agent node function.

    Reads research results from state, calls LLM to assess quality,
    enforces the max-retry cap, and returns routing decision to state.
    """
    current_turn     = state.get("current_turn", 1)
    research_attempts = state.get("research_attempts", 1)
    confidence_score  = state.get("confidence_score", 0.0)
    source_conflict   = state.get("source_conflict", False)
    entity_memory     = state.get("entity_memory") or {}
    sub_intents       = state.get("sub_intents") or []
    research_results  = state.get("research_results") or []
    research_summary  = state.get("_research_summary") or {}

    company = entity_memory.get("company_name") or "Unknown Company"

    # Get original query from conversation history
    history = state.get("conversation_history") or []
    original_query = _extract_original_query(history, current_turn)

    logger.info(
        f"[VALIDATOR] Turn {current_turn} | Attempt {research_attempts} | "
        f"Company: {company} | Confidence: {confidence_score}"
    )

    # ── Hard cap check: attempt 3 always passes ───────────────────────────────
    if research_attempts > MAX_RESEARCH_RETRIES:
        logger.info(
            f"[VALIDATOR] Max attempts ({MAX_RESEARCH_RETRIES}) reached — "
            "forcing sufficient to break loop"
        )
        return {
            "validation_result": "sufficient",
            "validator_notes": (
                f"Maximum research attempts reached ({MAX_RESEARCH_RETRIES}). "
                "Proceeding with available data. Some gaps may exist."
            ),
        }

    # ── LLM quality assessment ────────────────────────────────────────────────
    llm = _get_llm()

    prompt_text = validator_prompt(
        company=company,
        sub_intents=sub_intents,
        research_summary=research_summary,
        confidence_score=confidence_score,
        source_conflict=source_conflict,
        attempt_number=research_attempts,
        original_query=original_query,
    )

    try:
        response = llm.invoke([HumanMessage(content=prompt_text)])
        raw_output = response.content.strip()
        logger.debug(f"[VALIDATOR] LLM response: {raw_output[:300]}")
    except Exception as e:
        logger.error(f"[VALIDATOR] LLM call failed: {e}")
        # Fail safe: if we can't validate, assume sufficient and move on
        return {
            "validation_result": "sufficient",
            "validator_notes": f"Validator LLM call failed: {e}",
        }

    parsed = _parse_validator_response(raw_output)

    validation_result = parsed["validation_result"]
    quality_score     = parsed.get("quality_score", 5)
    source_agreement  = parsed.get("source_agreement", "unknown")
    notes             = parsed.get("validator_notes")
    reasoning         = parsed.get("reasoning", "")

    # Enrich notes with source agreement context if conflicting
    if source_agreement == "conflicting" and notes:
        notes = f"[SOURCE CONFLICT] {notes}"
    elif source_agreement == "conflicting" and not notes:
        notes = "Sources contain conflicting information — verify key figures with official reports."

    logger.info(
        f"[VALIDATOR] Result: {validation_result} | "
        f"quality={quality_score} | agreement={source_agreement} | "
        f"reasoning={reasoning[:100]}"
    )

    return {
        "validation_result": validation_result,
        "validator_notes":   notes,
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_original_query(history: list[dict], current_turn: int) -> str:
    """Extract the most recent user message for this turn."""
    for msg in reversed(history):
        if msg.get("role") == "user" and msg.get("turn") == current_turn:
            return msg.get("content", "")
    # Fallback: latest user message
    for msg in reversed(history):
        if msg.get("role") == "user":
            return msg.get("content", "")
    return "research query"


def _parse_validator_response(raw: str) -> dict:
    """Parse Validator LLM JSON response with fallback."""
    clean = raw.strip()

    if "```" in clean:
        start = clean.find("{")
        end   = clean.rfind("}") + 1
        if start != -1 and end > start:
            clean = clean[start:end]

    start = clean.find("{")
    end   = clean.rfind("}") + 1
    if start != -1 and end > start:
        clean = clean[start:end]

    try:
        data = json.loads(clean)
        return {
            "validation_result": data.get("validation_result", "sufficient"),
            "quality_score":     data.get("quality_score", 5),
            "source_agreement":  data.get("source_agreement", "unknown"),
            "validator_notes":   data.get("validator_notes"),
            "reasoning":         data.get("reasoning", ""),
        }
    except json.JSONDecodeError as e:
        logger.warning(f"[VALIDATOR] JSON parse failed: {e}")
        # Conservative fallback: pass through to Synthesis
        return {
            "validation_result": "sufficient",
            "quality_score":     5,
            "source_agreement":  "unknown",
            "validator_notes":   "Validator parse error — proceeding with available data",
            "reasoning":         "JSON parse failed",
        }
