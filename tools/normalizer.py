"""
LODESTONE — tools/normalizer.py
================================
Stage 2: Input normalization pipeline.

Runs BEFORE the Clarity Agent so the LLM receives the cleanest possible
version of the user's input — fixing noise without touching meaning.

Two-layer architecture:
  Layer 1 — Rule-based (always runs, zero cost, instant)
             Handles: whitespace collapse, repeated words, non-ASCII strip,
                      excessive punctuation, basic contraction fixes.

  Layer 2 — HuggingFace spell correction (runs if model loads successfully)
             Model: oliverguhr/spelling-correction-english-base
             A fine-tuned T5 model for English spell correction.
             Falls back to Layer 1 only if HF is unavailable (no internet,
             token issue, etc.) — the system never crashes because of this.

IMPORTANT: We deliberately do NOT fix ambiguous pronouns ("they", "that company").
Ambiguity is the Clarity Agent's job. The normalizer only removes noise.
"""

import re
import logging
from typing import Optional

from config import MODELS

logger = logging.getLogger(__name__)


# ── HuggingFace model (loaded once at module import, reused across calls) ─────

_hf_corrector = None   # will hold the pipeline if it loads successfully
_hf_attempted = False  # tracks whether we already tried loading (avoid retry spam)


def _load_hf_model() -> bool:
    """
    Attempt to load the HuggingFace spell correction pipeline.
    Returns True if successful, False if unavailable.
    Called lazily on first normalize() invocation.
    """
    global _hf_corrector, _hf_attempted

    if _hf_attempted:
        return _hf_corrector is not None

    _hf_attempted = True

    model_name = MODELS.get("normalizer_hf")
    if not model_name:
        logger.info("normalizer: HF model disabled in config, using rule-based only")
        return False

    try:
        from transformers import pipeline
        logger.info(f"normalizer: loading HF model '{model_name}' (first run only)...")
        _hf_corrector = pipeline(
            "text2text-generation",
            model=model_name,
            max_length=256,
        )
        logger.info("normalizer: HF spell correction model loaded successfully")
        return True

    except Exception as e:
        logger.warning(
            f"normalizer: could not load HF model '{model_name}' — "
            f"falling back to rule-based only. Reason: {e}"
        )
        return False


# ── Layer 1: Rule-based cleaning ──────────────────────────────────────────────

def _rule_based_clean(text: str) -> str:
    """
    Pure rule-based normalization. Fast, deterministic, no external deps.

    Rules applied in order:
      1. Strip leading/trailing whitespace
      2. Collapse internal whitespace (tabs, multiple spaces → single space)
      3. Limit consecutive newlines to 2 max
      4. Remove consecutive duplicate words  ("the the" → "the")
      5. Strip non-ASCII characters (emojis, special unicode noise)
      6. Collapse excessive punctuation ("!!!!!!" → "!")
      7. Fix common contraction spacing ("i m" → "i'm" is NOT done here —
         that changes meaning. We only fix unambiguous noise.)
    """
    # 1. Strip
    text = text.strip()

    # 2. Collapse internal whitespace
    text = re.sub(r'[ \t]+', ' ', text)

    # 3. Normalize newlines
    text = re.sub(r'\n{3,}', '\n\n', text)

    # 4. Remove consecutive duplicate words (case-insensitive)
    #    "apple apple" → "apple", but "New York" stays (different words)
    text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text, flags=re.IGNORECASE)

    # 5. Strip non-ASCII (keeps basic punctuation, removes emoji/unicode noise)
    text = re.sub(r'[^\x00-\x7F]+', '', text)

    # 6. Collapse excessive punctuation (3+ of any mix → keep first char only)
    text = re.sub(r'[!?.]{3,}', lambda m: m.group(0)[0], text)

    # 7. Final strip after all subs
    text = text.strip()

    return text


# ── Layer 2: HuggingFace spell correction ─────────────────────────────────────

def _hf_spell_correct(text: str) -> str:
    """
    Run the HuggingFace spell correction model on the text.
    Returns corrected text, or original text if model call fails.

    The model handles:
      - Misspelled company names: "aplle" → "apple", "amazn" → "amazon"
      - Common word typos: "abuot" → "about", "recnet" → "recent"
      - Does NOT hallucinate new content — it only corrects, doesn't rewrite.
    """
    if _hf_corrector is None:
        return text

    try:
        result = _hf_corrector(text, max_length=256)
        corrected = result[0]["generated_text"].strip()

        # Sanity check: if corrected output is drastically shorter than input,
        # the model may have truncated/hallucinated — fall back to original.
        if len(corrected) < len(text) * 0.5:
            logger.warning("normalizer: HF output suspiciously short, keeping original")
            return text

        return corrected

    except Exception as e:
        logger.warning(f"normalizer: HF correction failed mid-call — {e}")
        return text


# ── Public API ────────────────────────────────────────────────────────────────

def normalize(raw_input: str) -> str:
    """
    Main entry point. Called by the graph before the Clarity Agent node.

    Pipeline:
      raw input
        → rule-based clean   (always)
        → HF spell correct   (if model available)
        → rule-based clean   (second pass, catches artifacts from HF output)
        → normalized output

    Returns the cleaned string. Never raises — worst case returns the
    rule-cleaned version of the original input.

    Example:
      Input:  "  tell me abuot aplle inc financials financials  "
      Output: "tell me about apple inc financials"
    """
    if not raw_input or not raw_input.strip():
        return ""

    # Lazy-load HF model on first call
    _load_hf_model()

    # Pass 1: rule-based
    cleaned = _rule_based_clean(raw_input)
    logger.debug(f"normalizer: after rules: {repr(cleaned)}")

    # Pass 2: HF spell correction (skipped if model unavailable)
    if _hf_corrector is not None:
        cleaned = _hf_spell_correct(cleaned)
        logger.debug(f"normalizer: after HF: {repr(cleaned)}")

        # Pass 3: rule-based again (HF output can reintroduce whitespace artifacts)
        cleaned = _rule_based_clean(cleaned)
        logger.debug(f"normalizer: final: {repr(cleaned)}")

    return cleaned


def normalize_for_state(raw_input: str) -> dict:
    """
    Convenience wrapper that returns a partial state dict ready to merge.
    Used directly in graph.py node functions.

    Returns:
        {"raw_input": original, "normalized_input": cleaned}
    """
    return {
        "raw_input": raw_input,
        "normalized_input": normalize(raw_input),
    }
