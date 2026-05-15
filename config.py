"""
LODESTONE — config.py
=====================
Single source of truth for all constants, credentials, and model choices.
Nothing is hardcoded anywhere else in the project — every magic number
and model name lives here.
"""

import os
from dotenv import load_dotenv

load_dotenv()


# ── API KEYS ──────────────────────────────────────────────────────────────────

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")   # Tavily — free tier (1 000 searches/month)
GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")     # Groq   — free tier, fast inference


# ── MODEL SELECTION ───────────────────────────────────────────────────────────
#
# Each agent can use a different model — heavier agents get smarter models,
# lightweight agents can use smaller/free models to save cost.
#
# HUGGINGFACE NOTE:
#   HuggingFace free models ARE useful here — specifically for the
#   Normalization step (Stage 2 / tools/normalizer.py).
#   Recommended: "oliverguhr/spelling-correction-english-base"
#   This is a fine-tuned BERT model for spell correction — faster and cheaper
#   than burning an LLM call just to fix "Aplle" → "Apple".
#   For all reasoning agents (Clarity, Research, Validator, Synthesis),
#   stick with a proper LLM — HuggingFace free inference is too slow and
#   unreliable for multi-step agentic reasoning under real conditions.

MODELS = {
    # Primary reasoning model — used by Clarity, Validator, Synthesis agents
    # Using Groq's free-tier llama3-8b-8192 — no credit card required
    "primary": "llama3-8b-8192",

    # Research agent model — same free Groq model
    "research": "llama3-8b-8192",

    # Normalization — HuggingFace model, runs locally, no API cost
    # Set to None to skip HF and use rule-based cleaning only
    "normalizer_hf": "oliverguhr/spelling-correction-english-base",
}


# ── ROUTING THRESHOLDS ────────────────────────────────────────────────────────

CONFIDENCE_THRESHOLD = 6      # Research → Validator if score < this
MAX_RESEARCH_RETRIES = 3      # Validator loop hard cap
MIN_SOURCES_REQUIRED = 2      # Validator: at least this many sources needed


# ── RESEARCH SETTINGS ─────────────────────────────────────────────────────────

TAVILY_MAX_RESULTS  = 5       # results per Tavily search call
TAVILY_SEARCH_DEPTH = "advanced"   # "basic" or "advanced"

# Sub-intent categories the Research Agent will search for
RESEARCH_INTENTS = ["news", "financials", "leadership", "competitors"]


# ── CONVERSATION SETTINGS ─────────────────────────────────────────────────────

MAX_HISTORY_MESSAGES = 10     # how many past messages agents can see
                              # keeps context window from bloating


# ── LOGGING ───────────────────────────────────────────────────────────────────

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")   # DEBUG for dev, INFO for demo
