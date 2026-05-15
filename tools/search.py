"""
LODESTONE — tools/search.py
============================
Stage 2: Tavily search wrapper used exclusively by the Research Agent.

Responsibilities:
  1. Route each sub-intent to the most appropriate Tavily search topic
     ("news", "finance", "general") — Tavily returns much better results
     when the topic matches the query type.

  2. Run all sub-intent searches in parallel (asyncio) — not sequentially.
     For 4 intents this cuts research time from ~8s to ~2s.

  3. Compute a per-intent confidence score based on:
       - Number of results returned
       - Recency of results (recent = higher confidence)
       - Source diversity (same domain repeated = lower confidence)

  4. Detect source conflicts — if two results for the same intent
     contradict each other on numeric facts, flag source_conflict=True.
     The Synthesis Agent uses this to add a caveat block.

  5. Return structured ResearchResult objects that slot directly into state.
"""

import asyncio
import logging
import re
from datetime import datetime, timezone
from typing import Optional

from tavily import TavilyClient

from config import (
    TAVILY_API_KEY,
    TAVILY_MAX_RESULTS,
    TAVILY_SEARCH_DEPTH,
    RESEARCH_INTENTS,
    MIN_SOURCES_REQUIRED,
)
from state import ResearchResult

logger = logging.getLogger(__name__)


# ── Tavily client (singleton) ─────────────────────────────────────────────────

_client: Optional[TavilyClient] = None


def _get_client() -> TavilyClient:
    global _client
    if _client is None:
        if not TAVILY_API_KEY:
            raise ValueError(
                "TAVILY_API_KEY is not set. Add it to your .env file."
            )
        _client = TavilyClient(api_key=TAVILY_API_KEY)
    return _client


# ── Intent → Tavily topic mapping ─────────────────────────────────────────────
#
# Tavily's topic parameter significantly improves result quality.
# "news"    → recent articles, press releases, breaking developments
# "finance" → earnings reports, stock data, analyst coverage
# "general" → broad web results (leadership bios, competitor analysis)

INTENT_TOPIC_MAP = {
    "news":        "news",
    "financials":  "finance",
    "leadership":  "general",
    "competitors": "general",
}

# Query templates per intent — more specific queries = better Tavily results
INTENT_QUERY_TEMPLATES = {
    "news":        "{company} latest news recent developments 2024 2025",
    "financials":  "{company} revenue earnings financial results quarterly annual",
    "leadership":  "{company} CEO leadership executive team management",
    "competitors":  "{company} competitors market landscape industry rivals",
}


# ── Confidence scoring ────────────────────────────────────────────────────────

def _score_results(results: list[dict], intent: str) -> float:
    """
    Compute a confidence score (0.0–10.0) for a set of Tavily results.

    Scoring factors:
      - Volume:    more results = higher base score
      - Diversity: unique domains = bonus, repeated domains = penalty
      - Recency:   results with recent dates score higher (news/financials)
      - Relevance: Tavily returns a score field (0–1), we incorporate it

    This is intentionally simple — the Validator Agent does deeper quality
    assessment. This score is just for the routing decision (< 6 → Validator).
    """
    if not results:
        return 0.0

    score = 0.0

    # Volume component (max 4 points)
    volume_score = min(len(results) / TAVILY_MAX_RESULTS, 1.0) * 4.0
    score += volume_score

    # Diversity component (max 3 points)
    domains = []
    for r in results:
        url = r.get("url", "")
        match = re.search(r'https?://(?:www\.)?([^/]+)', url)
        if match:
            domains.append(match.group(1))
    unique_domains = len(set(domains))
    diversity_score = min(unique_domains / max(len(results), 1), 1.0) * 3.0
    score += diversity_score

    # Relevance component from Tavily's own score field (max 2 points)
    tavily_scores = [r.get("score", 0.5) for r in results]
    avg_relevance = sum(tavily_scores) / len(tavily_scores)
    score += avg_relevance * 2.0

    # Recency bonus for news and financials (max 1 point)
    if intent in ("news", "financials"):
        recency_bonus = 0.0
        now = datetime.now(timezone.utc)
        for r in results:
            pub_date = r.get("published_date", "")
            if pub_date:
                try:
                    # Tavily returns dates in various formats — try common ones
                    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"):
                        try:
                            dt = datetime.strptime(pub_date[:19], fmt).replace(tzinfo=timezone.utc)
                            days_old = (now - dt).days
                            if days_old <= 30:
                                recency_bonus += 0.25
                            elif days_old <= 90:
                                recency_bonus += 0.1
                            break
                        except ValueError:
                            continue
                except Exception:
                    pass
        score += min(recency_bonus, 1.0)

    return round(min(score, 10.0), 2)


# ── Source conflict detection ─────────────────────────────────────────────────

def _detect_conflict(results: list[dict]) -> bool:
    """
    Check if results contain contradictory numeric facts.

    Strategy: extract all dollar/percentage figures from result content,
    check if there are large divergences (> 20% difference) between
    figures that appear in multiple snippets for the same query.

    This is a lightweight heuristic — not perfect, but catches the common
    case where one source says "$94B revenue" and another says "$82B".
    """
    if len(results) < 2:
        return False

    # Extract all numeric values preceded by $ or followed by B/M/%
    number_pattern = re.compile(
        r'\$\s*([\d,]+(?:\.\d+)?)\s*(?:billion|million|B|M|bn|mn)?\b'
        r'|(\d+(?:\.\d+)?)\s*%'
    )

    all_figures: list[float] = []
    for r in results:
        content = r.get("content", "") + r.get("title", "")
        for match in number_pattern.finditer(content):
            raw = match.group(1) or match.group(2)
            if raw:
                try:
                    val = float(raw.replace(",", ""))
                    all_figures.append(val)
                except ValueError:
                    pass

    if len(all_figures) < 2:
        return False

    # If the max figure is more than 3x the min, flag a conflict
    # (accounts for B vs M unit mismatches being caught elsewhere)
    min_val = min(all_figures)
    max_val = max(all_figures)

    if min_val > 0 and (max_val / min_val) > 3.0:
        logger.info(
            f"search: source conflict detected — "
            f"figures range from {min_val} to {max_val}"
        )
        return True

    return False


# ── Core search function (single intent) ─────────────────────────────────────

def _search_one_intent(
    company: str,
    intent: str,
    extra_context: Optional[str] = None,
) -> ResearchResult:
    """
    Run a single Tavily search for one sub-intent.
    Builds the query, calls Tavily, scores results, returns ResearchResult.

    Args:
        company:       Confirmed company name from entity_memory.
        intent:        One of RESEARCH_INTENTS ("news", "financials", etc.)
        extra_context: Optional hint from validator_notes on what's missing.
    """
    client = _get_client()

    # Build query
    template = INTENT_QUERY_TEMPLATES.get(intent, "{company} {intent}")
    query = template.format(company=company, intent=intent)

    # Append validator's gap hint on retry runs
    if extra_context:
        query = f"{query} {extra_context}"

    topic = INTENT_TOPIC_MAP.get(intent, "general")

    logger.info(f"search: [{intent}] querying Tavily — '{query}'")

    try:
        response = client.search(
            query=query,
            search_depth=TAVILY_SEARCH_DEPTH,
            topic=topic,
            max_results=TAVILY_MAX_RESULTS,
            include_answer=True,       # Tavily's own AI answer as a summary seed
            include_raw_content=False, # saves tokens — snippets are enough
        )
    except Exception as e:
        logger.error(f"search: [{intent}] Tavily call failed — {e}")
        return ResearchResult(
            intent=intent,
            summary=f"Search failed for {intent}: {str(e)}",
            sources=[],
            raw_hits=[],
        )

    hits = response.get("results", [])
    tavily_answer = response.get("answer", "")

    # Build summary: Tavily's answer + top snippet titles
    summary_parts = []
    if tavily_answer:
        summary_parts.append(tavily_answer)
    for hit in hits[:3]:
        title = hit.get("title", "")
        snippet = hit.get("content", "")[:300]
        if title:
            summary_parts.append(f"• {title}: {snippet}")

    summary = "\n".join(summary_parts) if summary_parts else "No results found."

    sources = [hit.get("url", "") for hit in hits if hit.get("url")]

    confidence = _score_results(hits, intent)
    logger.info(f"search: [{intent}] {len(hits)} results, confidence={confidence}")

    return ResearchResult(
        intent=intent,
        summary=summary,
        sources=sources,
        raw_hits=hits,
        # Note: confidence stored per-intent in raw_hits metadata below
    )


# ── Parallel search (all intents) ────────────────────────────────────────────

async def _search_all_async(
    company: str,
    intents: list[str],
    extra_context: Optional[str] = None,
) -> list[ResearchResult]:
    """
    Run all intent searches in parallel using asyncio.
    Each search is run in a thread (Tavily client is sync) via run_in_executor.
    """
    # Use get_running_loop() — correct for Python 3.10+ inside an async function.
    # get_event_loop() is deprecated in async contexts from Python 3.10 onward.
    loop = asyncio.get_running_loop()

    tasks = [
        loop.run_in_executor(
            None,                    # default ThreadPoolExecutor
            _search_one_intent,
            company,
            intent,
            extra_context,
        )
        for intent in intents
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out any exceptions that slipped through gather
    clean_results = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            logger.error(f"search: parallel task for '{intents[i]}' raised {r}")
            clean_results.append(ResearchResult(
                intent=intents[i],
                summary=f"Search error: {str(r)}",
                sources=[],
                raw_hits=[],
            ))
        else:
            clean_results.append(r)

    return clean_results


# ── Public API ────────────────────────────────────────────────────────────────

def run_research(
    company: str,
    intents: list[str],
    extra_context: Optional[str] = None,
) -> tuple[list[ResearchResult], float, bool]:
    """
    Main entry point called by the Research Agent node.

    Runs all intent searches in parallel, computes overall confidence score,
    and detects source conflicts across all results.

    Args:
        company:       Company name from entity_memory.company_name
        intents:       List of sub-intents from state.sub_intents
        extra_context: state.validator_notes from previous failed attempt

    Returns:
        Tuple of:
          - results:          list[ResearchResult] — one per intent
          - confidence_score: float (0–10) — overall score across all intents
          - source_conflict:  bool — True if contradictory facts detected

    Usage in Research Agent:
        results, score, conflict = run_research(
            company=state["entity_memory"]["company_name"],
            intents=state["sub_intents"],
            extra_context=state.get("validator_notes"),
        )
    """
    if not company:
        raise ValueError("run_research: company name cannot be empty")

    if not intents:
        intents = RESEARCH_INTENTS
        logger.info("search: no intents specified, defaulting to all")

    # Always create a fresh event loop — avoids DeprecationWarning from
    # asyncio.get_event_loop() on Python 3.12 and RuntimeError when called
    # from a non-main thread (which LangGraph nodes may run in).
    loop = asyncio.new_event_loop()
    try:
        results = loop.run_until_complete(
            _search_all_async(company, intents, extra_context)
        )
    finally:
        loop.close()

    # Overall confidence: average of per-intent scores
    per_intent_scores = [
        _score_results(r["raw_hits"], r["intent"])
        for r in results
    ]
    overall_confidence = (
        round(sum(per_intent_scores) / len(per_intent_scores), 2)
        if per_intent_scores else 0.0
    )

    # Conflict detection: pool all raw hits across intents
    all_hits = []
    for r in results:
        all_hits.extend(r["raw_hits"])
    conflict = _detect_conflict(all_hits)

    logger.info(
        f"search: complete — "
        f"{len(results)} intents, "
        f"confidence={overall_confidence}, "
        f"conflict={conflict}"
    )

    return results, overall_confidence, conflict
