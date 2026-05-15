"""
LODESTONE — agents/clarity.py
==============================
Stage 3: The Clarity Agent node.

Responsibilities (in execution order):
  1. Call entity_store.extract_entity() to update entity memory
  2. Check if memory is fresh enough to resolve pronouns
  3. Call the LLM with the clarity prompt
  4. Parse the JSON response
  5. Return partial state dict with clarity decision + sub_intents

The Clarity Agent is the only node that can trigger an INTERRUPT.
When it does, the graph halts and surfaces clarification_question to the user.
The interrupt/resume cycle is handled in graph.py — this node just sets the flag.

Design decision — entity extraction BEFORE LLM call:
  We run the rule-based entity extractor first. If it finds a company,
  we pass that to the LLM as confirmed context. This means the LLM doesn't
  need to guess at entity resolution — it just confirms or rejects what
  the extractor found. Cheaper, faster, more reliable.
"""

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from state import AgentState, EntityMemory
from prompts import clarity_prompt
from memory.entity_store import extract_entity, is_memory_fresh
from config import MODELS, MAX_HISTORY_MESSAGES

logger = logging.getLogger(__name__)


def _get_llm():
    """
    Load Groq LLM client (free tier — no credit card required).
    Model: llama3-8b-8192 via api.groq.com
    """
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


def run_clarity_agent(state: AgentState) -> dict[str, Any]:
    """
    Clarity Agent node function.

    Called by LangGraph when the graph reaches the clarity_agent node.
    Reads from state, returns a partial state dict to merge.

    Flow:
      normalized_input
        → entity extraction (rule-based, free)
        → freshness check (is old memory still valid?)
        → LLM clarity assessment
        → JSON parse
        → return partial state
    """
    logger.info(f"[CLARITY] Turn {state['current_turn']} — starting")

    normalized = state.get("normalized_input", "")
    current_memory: EntityMemory = state.get("entity_memory") or {
        "company_name": None,
        "ticker": None,
        "last_turn": None,
    }
    current_turn = state.get("current_turn", 1)
    conversation_history = state.get("conversation_history", [])

    # ── Step 1: Rule-based entity extraction ─────────────────────────────────
    updated_memory = extract_entity(
        normalized_input=normalized,
        current_memory=current_memory,
        current_turn=current_turn,
    )

    # ── Step 2: Freshness check ────────────────────────────────────────────────
    # If entity was confirmed long ago and input uses pronouns,
    # treat memory as stale — force re-extraction via LLM
    memory_to_use = updated_memory
    if not updated_memory.get("company_name") and current_memory.get("company_name"):
        # Entity extractor found nothing new — check if old memory is fresh
        if is_memory_fresh(current_memory, current_turn, max_gap=5):
            memory_to_use = current_memory
            logger.info(
                f"[CLARITY] Using fresh memory: '{current_memory['company_name']}' "
                f"from turn {current_memory.get('last_turn')}"
            )
        else:
            logger.info("[CLARITY] Memory stale — will rely on LLM to flag ambiguity")
            memory_to_use = updated_memory

    # ── Step 3: LLM call ──────────────────────────────────────────────────────
    llm = _get_llm()
    prompt_text = clarity_prompt(
        normalized_input=normalized,
        entity_memory=memory_to_use,
        conversation_history=conversation_history,
        current_turn=current_turn,
    )

    logger.debug(f"[CLARITY] Sending prompt to LLM ({len(prompt_text)} chars)")

    try:
        response = llm.invoke([HumanMessage(content=prompt_text)])
        raw_output = str(response.content).strip()
        logger.debug(f"[CLARITY] Raw LLM response: {raw_output[:300]}")
    except Exception as e:
        logger.error(f"[CLARITY] LLM call failed: {e}")
        # Fail safe: treat as needing clarification rather than guessing
        return {
            "clarity_status": "needs_clarification",
            "clarification_question": "I had trouble processing your request. Could you rephrase it?",
            "sub_intents": [],
            "entity_memory": memory_to_use,
        }

    # ── Step 4: JSON parse ────────────────────────────────────────────────────
    parsed = _parse_clarity_response(raw_output)

    # ── Step 5: Override: if entity memory has a company but LLM says unclear,
    #    double-check — the extractor is more reliable for company detection.
    if (
        parsed["clarity_status"] == "needs_clarification"
        and memory_to_use.get("company_name")
    ):
        logger.info(
            "[CLARITY] LLM flagged unclear but entity memory has a company — "
            f"checking if query is truly ambiguous..."
        )
        # Only override if LLM's own reasoning mentions the company was found
        reasoning = parsed.get("reasoning", "").lower()
        company_lower = memory_to_use["company_name"].lower()
        if company_lower in reasoning or "found" in reasoning or "memory" in reasoning:
            logger.info("[CLARITY] Memory override: treating as clear")
            parsed["clarity_status"] = "clear"
            parsed["clarification_question"] = None

    clarity_status = parsed["clarity_status"]
    sub_intents    = parsed.get("sub_intents", [])
    clarification  = parsed.get("clarification_question")
    reasoning      = parsed.get("reasoning", "")

    logger.info(
        f"[CLARITY] Result: status={clarity_status} | "
        f"intents={sub_intents} | "
        f"company={memory_to_use.get('company_name')} | "
        f"reasoning={reasoning}"
    )

    # Build the clarification message for conversation history if needed
    history_addition = []
    if clarity_status == "needs_clarification" and clarification:
        history_addition = [{
            "role": "assistant",
            "content": clarification,
            "turn": current_turn,
        }]

    return {
        "clarity_status": clarity_status,
        "clarification_question": clarification if clarity_status == "needs_clarification" else None,
        "sub_intents": sub_intents,
        "entity_memory": memory_to_use,
        "conversation_history": history_addition,  # appends via operator.add
    }


# ── JSON parsing with fallback ────────────────────────────────────────────────

def _parse_clarity_response(raw: str) -> dict:
    """
    Parse the LLM's JSON response with robust fallback handling.

    Handles common LLM formatting failures:
      - Markdown fences (```json ... ```)
      - Leading/trailing explanation text
      - Partial JSON with missing fields
    """
    # Strip markdown fences if present
    clean = raw.strip()
    if clean.startswith("```"):
        lines = clean.split("\n")
        # Remove first and last fence lines
        inner_lines = []
        in_block = False
        for line in lines:
            if line.startswith("```") and not in_block:
                in_block = True
                continue
            elif line.startswith("```") and in_block:
                break
            elif in_block:
                inner_lines.append(line)
        clean = "\n".join(inner_lines)

    # Find JSON object in response (in case LLM added explanation text)
    start = clean.find("{")
    end   = clean.rfind("}") + 1
    if start != -1 and end > start:
        clean = clean[start:end]

    try:
        data = json.loads(clean)
    except json.JSONDecodeError as e:
        logger.error(f"[CLARITY] JSON parse failed: {e}. Raw: {raw[:200]}")
        # Fallback: assume needs clarification (safe default)
        return {
            "clarity_status": "needs_clarification",
            "sub_intents": [],
            "clarification_question": "Could you clarify which company you're asking about?",
            "reasoning": "JSON parse failed — defaulting to clarification",
        }

    # Ensure required fields exist with defaults
    return {
        "clarity_status": data.get("clarity_status", "needs_clarification"),
        "sub_intents":    data.get("sub_intents", []),
        "clarification_question": data.get("clarification_question"),
        "reasoning":      data.get("reasoning", ""),
    }
