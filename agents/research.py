"""
LODESTONE — agents/research.py
================================
Stage 4: The Research Agent node.

Responsibilities:
  1. Read confirmed entity from state (company name + ticker)
  2. Read sub_intents from state (set by Clarity Agent)
  3. Call tools/search.py → run_research() for parallel Tavily searches
  4. Call LLM to assemble raw results into structured summaries
  5. Return research_results, confidence_score, source_conflict to state

Design decision — two-step research:
  Step A (tool call): Tavily fetches raw results in parallel — fast, factual,
                      no LLM involved. search.py handles this entirely.
  Step B (LLM call):  LLM assembles the raw snippets into per-intent summaries
                      and extracts key facts. This separation means:
                        - The LLM never needs to search the web itself
                        - The LLM only summarizes text it was explicitly given
                        - Hallucination risk is minimized

The confidence_score comes from search.py's scoring algorithm (source volume,
diversity, recency) — NOT from the LLM. LLMs are unreliable at self-assessing
confidence. Math is better.
"""

import json
import logging
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from state import AgentState
from tools.search import run_research
from prompts import research_system_prompt, research_assembly_prompt
from memory.entity_store import get_company_for_search
from config import MODELS, RESEARCH_INTENTS

logger = logging.getLogger(__name__)


def _get_llm():
    """Load Groq LLM client (free tier — llama3-8b-8192)."""
    from config import GROQ_API_KEY
    model = MODELS.get("research", "llama3-8b-8192")

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


def run_research_agent(state: AgentState) -> dict[str, Any]:
    """
    Research Agent node function.

    Reads sub_intents and entity_memory from state.
    Runs parallel Tavily searches via tools/search.py.
    Assembles results into structured summaries via LLM.
    Returns updated research fields for state.
    """
    current_turn     = state.get("current_turn", 1)
    entity_memory    = state.get("entity_memory") or {}
    sub_intents      = state.get("sub_intents") or RESEARCH_INTENTS
    validator_notes  = state.get("validator_notes")
    research_attempts = state.get("research_attempts", 0) + 1

    # Get the best search string for this company
    company_search_str = get_company_for_search(entity_memory)
    if not company_search_str:
        logger.error("[RESEARCH] No company in entity memory — cannot research")
        return {
            "research_results": [],
            "confidence_score": 0.0,
            "source_conflict": False,
            "research_attempts": research_attempts,
        }

    logger.info(
        f"[RESEARCH] Turn {current_turn} | Attempt {research_attempts} | "
        f"Company: {company_search_str} | Intents: {sub_intents}"
    )

    # ── Step A: Parallel Tavily search ────────────────────────────────────────
    try:
        raw_results, confidence_score, source_conflict = run_research(
            company=company_search_str,
            intents=sub_intents,
            extra_context=validator_notes,
        )
    except Exception as e:
        logger.error(f"[RESEARCH] Search tool failed: {e}")
        return {
            "research_results": [],
            "confidence_score": 0.0,
            "source_conflict": False,
            "research_attempts": research_attempts,
        }

    logger.info(
        f"[RESEARCH] Search complete — "
        f"confidence={confidence_score} | conflict={source_conflict} | "
        f"{len(raw_results)} intent results"
    )

    # ── Step B: LLM assembly ──────────────────────────────────────────────────
    llm = _get_llm()

    assembly_prompt = research_assembly_prompt(
        company=company_search_str,
        sub_intents=sub_intents,
        research_results=[dict(r) for r in raw_results],
        validator_notes=validator_notes,
        attempt_number=research_attempts,
    )

    try:
        response = llm.invoke([
            SystemMessage(content=research_system_prompt()),
            HumanMessage(content=assembly_prompt),
        ])
        raw_output = response.content.strip()
        logger.debug(f"[RESEARCH] LLM assembly response: {raw_output[:400]}")
    except Exception as e:
        logger.error(f"[RESEARCH] LLM assembly failed: {e}")
        # Fallback: use raw Tavily summaries directly
        raw_output = _build_fallback_summary(raw_results)

    structured_summary = _parse_research_response(raw_output)

    # Enrich raw_results with the structured summary for downstream agents
    # We keep both: raw_results for the Validator (needs raw_hits),
    # and structured_summary as a clean dict in state
    state_update = {
        "research_results": [dict(r) for r in raw_results],
        "confidence_score": confidence_score,
        "source_conflict":  source_conflict,
        "research_attempts": research_attempts,
        # Store structured summary in validator_notes temporarily
        # so Synthesis Agent can access it — overwritten by Validator later
        "_research_summary": structured_summary,
    }

    logger.info(
        f"[RESEARCH] Complete — "
        f"confidence={confidence_score} | "
        f"key_facts={len(structured_summary.get('key_facts', []))} | "
        f"gaps={len(structured_summary.get('data_gaps', []))}"
    )

    return state_update


# ── Response parsing ──────────────────────────────────────────────────────────

def _parse_research_response(raw: str) -> dict:
    """Parse LLM's JSON assembly response with fallback."""
    clean = raw.strip()

    # Strip markdown fences
    if "```" in clean:
        start = clean.find("{")
        end   = clean.rfind("}") + 1
        if start != -1 and end > start:
            clean = clean[start:end]

    try:
        data = json.loads(clean)
        return {
            "intent_summaries": data.get("intent_summaries", {}),
            "key_facts":        data.get("key_facts", []),
            "data_gaps":        data.get("data_gaps", []),
            "source_urls":      data.get("source_urls", []),
        }
    except json.JSONDecodeError as e:
        logger.warning(f"[RESEARCH] JSON parse failed: {e} — using fallback")
        return {
            "intent_summaries": {"general": clean[:1000]},
            "key_facts":        [],
            "data_gaps":        ["Structured parsing failed — raw output used"],
            "source_urls":      [],
        }


def _build_fallback_summary(raw_results: list) -> dict:
    """Build a minimal summary dict directly from Tavily results if LLM fails."""
    intent_summaries = {}
    source_urls = []

    for r in raw_results:
        intent = r.get("intent", "general")
        summary = r.get("summary", "No data retrieved.")
        sources = r.get("sources", [])
        intent_summaries[intent] = summary[:800]
        source_urls.extend(sources[:3])

    return {
        "intent_summaries": intent_summaries,
        "key_facts": [],
        "data_gaps": [],
        "source_urls": list(set(source_urls)),
    }
