"""
LODESTONE — memory/entity_store.py
====================================
Stage 3: Entity memory — the coreference resolution layer.

This is what makes LODESTONE feel like a real research assistant rather
than a stateless chatbot. When the user says "What about their CEO?" in
turn 3, the system already knows "their" = "Apple Inc." from turn 1.

Responsibilities:
  1. Extract company entities from normalized user input (regex + ticker dict)
  2. Resolve pronouns/vague references to the last confirmed entity in state
  3. Normalize company names (strip "Inc.", "Corp.", etc. for cleaner queries)
  4. Maintain the EntityMemory dict in state across conversation turns
  5. Detect when a NEW company is being asked about (vs. follow-up on same one)

Design decision — no spaCy/NER model used here:
  For a business research assistant, the entity space is well-defined:
  company names. A curated ticker dictionary + capitalization heuristics
  outperforms a general-purpose NER model for this specific domain, with
  zero latency and no download dependency. If you need broader NER
  (person names, locations), spaCy can be dropped in as Layer 2.
"""

import re
import logging
from typing import Optional

from state import EntityMemory

logger = logging.getLogger(__name__)


# ── Ticker / company name dictionary ─────────────────────────────────────────
#
# Maps lowercase common name → stock ticker.
# Covers the companies that appear in >95% of business research queries.
# Add entries freely — this is just a dict.

KNOWN_COMPANIES: dict[str, str] = {
    # Big Tech
    "apple":          "AAPL",
    "microsoft":      "MSFT",
    "google":         "GOOGL",
    "alphabet":       "GOOGL",
    "amazon":         "AMZN",
    "meta":           "META",
    "facebook":       "META",
    "tesla":          "TSLA",
    "nvidia":         "NVDA",
    "netflix":        "NFLX",
    "intel":          "INTC",
    "amd":            "AMD",
    "qualcomm":       "QCOM",
    "oracle":         "ORCL",
    "ibm":            "IBM",
    "salesforce":     "CRM",
    "adobe":          "ADBE",
    "paypal":         "PYPL",
    "uber":           "UBER",
    "lyft":           "LYFT",
    "airbnb":         "ABNB",
    "spotify":        "SPOT",
    "snap":           "SNAP",
    "pinterest":      "PINS",
    "shopify":        "SHOP",
    "zoom":           "ZM",
    "palantir":       "PLTR",
    "snowflake":      "SNOW",
    "crowdstrike":    "CRWD",
    "datadog":        "DDOG",
    "mongodb":        "MDB",
    # Hardware / Semiconductors
    "tsmc":           "TSM",
    "samsung":        "SSNLF",
    "sony":           "SONY",
    "asml":           "ASML",
    "broadcom":       "AVGO",
    "arm":            "ARM",
    # Finance
    "jpmorgan":       "JPM",
    "jp morgan":      "JPM",
    "goldman sachs":  "GS",
    "morgan stanley": "MS",
    "bank of america":"BAC",
    "citigroup":      "C",
    "wells fargo":    "WFC",
    "berkshire":      "BRK",
    "blackrock":      "BLK",
    "visa":           "V",
    "mastercard":     "MA",
    # Consumer / Retail
    "walmart":        "WMT",
    "target":         "TGT",
    "costco":         "COST",
    "nike":           "NKE",
    "disney":         "DIS",
    "mcdonald":       "MCD",
    "mcdonalds":      "MCD",
    "starbucks":      "SBUX",
    "coca cola":      "KO",
    "pepsi":          "PEP",
    "pepsico":        "PEP",
    # Private / notable
    "openai":         "PRIVATE",
    "anthropic":      "PRIVATE",
    "spacex":         "PRIVATE",
    "stripe":         "PRIVATE",
    "bytedance":      "PRIVATE",
    "tiktok":         "PRIVATE",
    "x":              "PRIVATE",    # formerly twitter
    "twitter":        "PRIVATE",
    # Telecom / Other
    "at&t":           "T",
    "verizon":        "VZ",
    "tmobile":        "TMUS",
    "t-mobile":       "TMUS",
    "comcast":        "CMCSA",
}

# Reverse map: ticker → canonical name
_TICKER_TO_NAME: dict[str, str] = {
    v: k.title() for k, v in KNOWN_COMPANIES.items()
    if v not in ("PRIVATE", "N/A")
}

# Corporate suffix pattern — stripped during normalization
_SUFFIX_RE = re.compile(
    r'\s*\b(Inc\.?|Corp\.?|Corporation|Ltd\.?|LLC|Limited|Co\.?|'
    r'Group|Holdings?|PLC|S\.A\.?|AG|GmbH|N\.V\.)\b\.?\s*$',
    re.IGNORECASE,
)

# Manual misspelling corrections for company names.
# These catch the cases HuggingFace spell correction would handle
# if the model is available, but provides a deterministic fallback.
# Add entries as you encounter them in real usage.
COMPANY_MISSPELLINGS: dict[str, str] = {
    # Google variants
    "gooogle": "google", "gogle": "google", "googl": "google",
    "goggle": "google", "gooogel": "google",
    # Apple variants
    "aplle": "apple", "aple": "apple", "appel": "apple",
    # Amazon variants
    "amazn": "amazon", "amazom": "amazon", "amzon": "amazon",
    # Microsoft variants
    "microsft": "microsoft", "microsfot": "microsoft", "micorsoft": "microsoft",
    # Tesla variants
    "tesle": "tesla", "teslla": "tesla",
    # Nvidia variants
    "nvida": "nvidia", "nvidea": "nvidia",
    # Meta / Facebook
    "facbook": "facebook", "facebok": "facebook",
    # Netflix
    "netflex": "netflix", "netlix": "netflix",
    # General
    "palantir": "palantir",  # often misspelled as "palantier"
    "palantier": "palantir",
}


def _apply_misspelling_corrections(text: str) -> str:
    """
    Apply manual misspelling corrections to the text before entity extraction.
    Operates on individual tokens to avoid corrupting surrounding context.
    Only corrects tokens that exactly match a known misspelling.
    """
    tokens = text.split()
    corrected = []
    for token in tokens:
        clean_token = token.lower().strip("'\".,!?")
        if clean_token in COMPANY_MISSPELLINGS:
            # Preserve original capitalisation style if present
            correction = COMPANY_MISSPELLINGS[clean_token]
            if token[0].isupper():
                correction = correction.title()
            corrected.append(correction)
        else:
            corrected.append(token)
    return " ".join(corrected)

# Pronoun / vague reference patterns — signal "same company as before"
_PRONOUN_RE = re.compile(
    r'\b(they|them|their|theirs|it|its|the company|that company|'
    r'the firm|the organization|the brand|the startup|he|him|his|she|her|hers)\b',
    re.IGNORECASE,
)

# Capitalized multi-word company name pattern
# Matches: "Apple Inc", "Goldman Sachs", "Microsoft Corporation"
_PROPER_NOUN_RE = re.compile(
    r'\b([A-Z][a-zA-Z&]+(?:[\s\-][A-Z][a-zA-Z&]+)*'
    r'(?:\s+(?:Inc|Corp|Ltd|LLC|Group|Holdings|PLC)\.?)?)\b'
)

# Direct ticker reference (e.g., "TSLA earnings", "info on NVDA")
_TICKER_RE = re.compile(r'\b([A-Z]{2,5})\b')


# ── Internal helpers ──────────────────────────────────────────────────────────

def _strip_suffix(name: str) -> str:
    """Remove corporate suffixes from a company name."""
    return _SUFFIX_RE.sub('', name).strip()


def _lookup_ticker(name: str) -> Optional[str]:
    """
    Look up ticker for a company name.
    Tries lowercase match first, then checks if input IS a ticker.
    Returns None if unknown.
    """
    lower = name.lower().strip()

    # Direct lowercase name match
    if lower in KNOWN_COMPANIES:
        return KNOWN_COMPANIES[lower]

    # Partial match (e.g., "Apple Inc" → "apple" in dict)
    for known_name, ticker in KNOWN_COMPANIES.items():
        if known_name in lower or lower in known_name:
            return ticker

    # Check if the raw name itself is a known ticker symbol
    upper = name.upper().strip()
    if upper in _TICKER_TO_NAME:
        return upper

    return None


def _extract_company_candidates(text: str) -> list[str]:
    """
    Extract all plausible company name candidates from text.

    Returns a list ordered by confidence:
      1. Proper nouns (capitalized sequences) — highest confidence
      2. Known lowercase names found verbatim in text
      3. Ticker symbols found in text
    """
    candidates = []

    # 1. Proper noun extraction
    for match in _PROPER_NOUN_RE.finditer(text):
        cand = match.group(1).strip()
        # Filter out common false positives — sentence starters, question words,
        # business jargon, and any word that appears in KNOWN_COMPANIES stoplist.
        # "Now tell me about Ford" → "Now" must never win over "Ford".
        _STOPWORDS = {
            # Sentence starters / transitional words
            'now', 'tell', 'show', 'give', 'find', 'get', 'look',
            'also', 'just', 'still', 'then', 'next', 'last', 'first',
            'please', 'can', 'could', 'would', 'should', 'maybe',
            'instead', 'rather', 'actually', 'basically', 'switch', 'back',
            # Question words
            'what', 'who', 'how', 'when', 'where', 'which', 'why',
            # Common nouns mistaken as proper
            'ceo', 'cfo', 'coo', 'cto', 'vp', 'president',
            'the', 'a', 'an', 'is', 'are', 'was', 'were',
            'recent', 'latest', 'new', 'old', 'big', 'small',
            'top', 'best', 'worst', 'main', 'key', 'major',
            'news', 'stock', 'market', 'company', 'firm',
            'info', 'data', 'report', 'revenue', 'earnings',
            'about', 'more', 'some', 'any', 'all', 'no',
        }
        if cand.lower() not in _STOPWORDS and len(cand) > 2 and not (cand.isupper() and len(cand) > 6):
            candidates.append(cand)

    # 2. Lowercase known company names
    lower_text = text.lower()
    for known_name in sorted(KNOWN_COMPANIES.keys(), key=len, reverse=True):
        if known_name in lower_text and known_name not in [c.lower() for c in candidates]:
            candidates.append(known_name.title())

    # 3. Ticker symbols (if recognized)
    for match in _TICKER_RE.finditer(text):
        ticker = match.group(1)
        if ticker in _TICKER_TO_NAME:
            canonical = _TICKER_TO_NAME[ticker].title()
            if canonical not in candidates:
                candidates.append(canonical)

    return candidates


def _has_pronoun_reference(text: str) -> bool:
    """Return True if the text contains a pronoun/vague reference."""
    return bool(_PRONOUN_RE.search(text))


# ── Public API ────────────────────────────────────────────────────────────────

def extract_entity(
    normalized_input: str,
    current_memory: EntityMemory,
    current_turn: int,
) -> EntityMemory:
    """
    Main entry point. Called inside the Clarity Agent node.

    Given the current normalized input and existing entity memory,
    returns an updated EntityMemory dict.

    Resolution logic:
      Case A — New entity found in input:
        → Update memory with new company name + ticker + turn number.

      Case B — Pronoun/vague reference, existing memory present:
        → Keep current memory as-is (we already know the company).
        → Log that coreference was resolved.

      Case C — Pronoun/vague reference, NO existing memory:
        → Return empty memory. Clarity Agent will catch this and
          trigger the interrupt to ask the user.

      Case D — No entity found, no pronouns:
        → Return empty memory. Clarity Agent handles ambiguity.

    Args:
        normalized_input:  Output of normalizer.normalize()
        current_memory:    state["entity_memory"] from current state
        current_turn:      state["current_turn"]

    Returns:
        Updated EntityMemory dict (safe to merge into state directly)
    """
    has_pronoun = _has_pronoun_reference(normalized_input)
    # Apply manual misspelling corrections before candidate extraction
    # so "gooogle" → "google" before the regex runs
    corrected_input = _apply_misspelling_corrections(normalized_input)
    candidates  = _extract_company_candidates(corrected_input)

    logger.debug(
        f"entity_store: input={repr(normalized_input[:60])} | "
        f"candidates={candidates} | pronoun={has_pronoun}"
    )

    # ── Case A: New entity explicitly named ──────────────────────────────────
    if candidates:
        raw_name = candidates[0]          # highest-confidence candidate
        clean_name = _strip_suffix(raw_name)
        ticker = _lookup_ticker(raw_name)

        updated: EntityMemory = {
            "company_name": clean_name,
            "ticker":       ticker,
            "last_turn":    current_turn,
        }

        # If same company as before, just refresh the turn counter quietly
        existing = current_memory.get("company_name", "")
        if existing and existing.lower() == clean_name.lower():
            logger.debug(f"entity_store: same company confirmed — {clean_name}")
        else:
            logger.info(
                f"entity_store: new entity extracted — "
                f"'{clean_name}' (ticker={ticker})"
            )

        return updated

    # ── Case B: Pronoun reference, memory exists ─────────────────────────────
    if has_pronoun and current_memory.get("company_name"):
        logger.info(
            f"entity_store: pronoun resolved → "
            f"'{current_memory['company_name']}' (from turn {current_memory.get('last_turn')})"
        )
        # Refresh last_turn so memory stays "warm"
        return {**current_memory, "last_turn": current_turn}

    # ── Case C / D: Cannot resolve ───────────────────────────────────────────
    if has_pronoun:
        logger.warning(
            "entity_store: pronoun reference found but no entity in memory — "
            "Clarity Agent must interrupt"
        )
    else:
        logger.info(
            "entity_store: no entity detected — "
            "Clarity Agent will assess if query is still actionable"
        )

    return {
        "company_name": None,
        "ticker":       None,
        "last_turn":    None,
    }


def get_company_for_search(entity_memory: EntityMemory) -> Optional[str]:
    """
    Returns the best search string for the confirmed company.
    Prefers ticker-resolved canonical name for cleaner Tavily queries.

    Example:
        entity_memory = {"company_name": "Apple", "ticker": "AAPL", ...}
        → returns "Apple Inc. (AAPL)"

    Returns None if no company is confirmed (should never reach Research
    Agent in this case, but defensive check is good practice).
    """
    name   = entity_memory.get("company_name")
    ticker = entity_memory.get("ticker")

    if not name:
        return None

    if ticker and ticker not in ("PRIVATE", "N/A", None):
        return f"{name} ({ticker})"

    return name


def is_memory_fresh(entity_memory: EntityMemory, current_turn: int, max_gap: int = 5) -> bool:
    """
    Returns True if the entity memory is recent enough to trust for
    coreference resolution.

    If the user talked about Apple in turn 1, then had a long unrelated
    conversation for 5+ turns, and then says "what about their CEO?" —
    it's ambiguous enough that we should ask again rather than assume Apple.

    Args:
        max_gap: number of turns after which memory is considered stale
    """
    last_turn = entity_memory.get("last_turn")
    if last_turn is None:
        return False
    return (current_turn - last_turn) <= max_gap
