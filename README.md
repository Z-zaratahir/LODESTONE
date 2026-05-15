# LODESTONE

> **A multi-agent business research assistant built on LangGraph.**  
> Type anything — messy, abbreviated, full of typos — and get a clean, sourced, structured answer about any public company.

---

## What It Does

LODESTONE takes a free-form user query like *"tell me abuot aplle inc financials and their ceo??"* and:

1. **Cleans** the input (fixes typos, collapses noise) without changing meaning
2. **Understands** what company and what topics are being asked about
3. **Asks** one targeted question if the query is genuinely ambiguous
4. **Searches** the live web via Tavily across multiple dimensions in parallel
5. **Validates** whether the results are actually good enough to answer the question
6. **Retries** up to three times if quality is too low, targeting specific gaps
7. **Synthesizes** a clean, structured, markdown-formatted response with sources
8. **Remembers** context across follow-up questions — *"what about their CEO?"* works without re-specifying the company

All decisions are traceable through a shared state object. No silent guessing.

---

## Technology Stack

| Component | Technology | Cost |
|---|---|---|
| Agent orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) | Free |
| LLM inference | [Groq](https://console.groq.com) — `llama3-8b-8192` | **Free tier** |
| Web search | [Tavily](https://tavily.com) | **Free tier** (1,000 searches/month) |
| LLM client | [LangChain Groq](https://python.langchain.com/docs/integrations/chat/groq/) | Free |
| Spell correction | HuggingFace `oliverguhr/spelling-correction-english-base` | Free (local) |
| Environment config | python-dotenv | Free |

**Total API cost to run: $0.** Both Groq and Tavily have free tiers that require no credit card.

---

## Project Structure

```
LODESTONE/
│
├── main.py              Entry point. Conversation loop, interrupt handling.
├── graph.py             LangGraph graph. All nodes, edges, routing logic.
├── state.py             AgentState TypedDict. The shared memory object.
├── config.py            All constants, API keys, model names, thresholds.
├── prompts.py           All LLM prompt functions in one file.
│
├── agents/
│   ├── clarity.py       Clarity Agent — intent decomposition, ambiguity detection.
│   ├── research.py      Research Agent — Tavily search + LLM assembly.
│   ├── validator.py     Validator Agent — quality check, source triangulation.
│   └── synthesis.py     Synthesis Agent — final user-facing response.
│
├── tools/
│   ├── normalizer.py    Two-layer input cleaner (rules + HF spell correction).
│   └── search.py        Tavily wrapper — parallel search, confidence scoring.
│
└── memory/
    └── entity_store.py  Company extractor — coreference resolution, freshness check.
```

---

## Setup

### Prerequisites

- Python 3.10 or higher (3.12 recommended)
- A [Groq API key](https://console.groq.com) (free, no credit card)
- A [Tavily API key](https://tavily.com) (free, 1,000 searches/month)

### 1. Clone and enter the project

```bash
git clone https://github.com/Z-zaratahir/LODESTONE.git
cd LODESTONE
```

### 2. Create and activate a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure your API keys

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

Edit `.env`:

```env
TAVILY_API_KEY=tvly-your-real-key-here
GROQ_API_KEY=gsk_your-real-key-here
LOG_LEVEL=INFO
```

Get your keys:
- **Groq:** [console.groq.com](https://console.groq.com) → Create API Key (free, instant)
- **Tavily:** [tavily.com](https://tavily.com) → Sign Up → API Keys (free)

### 5. Run

```bash
python main.py
```

**Options:**

```bash
python main.py --debug          # verbose logging (shows every agent decision)
python main.py --no-hf          # skip HuggingFace spell correction (faster startup)
python main.py --thread-id ID   # resume a specific past conversation
```

---

## Example Session

```
════════════════════════════════════════════════════════════
  LODESTONE — Business Research Assistant
  Multi-Agent System | LangGraph + Tavily
════════════════════════════════════════════════════════════
  Type your query. Type 'exit' or 'quit' to stop.
  Type 'new' to start a fresh conversation.
════════════════════════════════════════════════════════════

You: tell me abuot aplle inc financials and ceo

  Processing...

────────────────────────────────────────────────────────────
  LODESTONE
────────────────────────────────────────────────────────────

Here's a current overview of Apple Inc. across the topics you asked about.

## Financials
Apple reported revenue of $94.9B in Q1 FY2025, up 4% year-over-year...

## Leadership
Tim Cook has served as Apple's CEO since 2011. In early 2025...

────────────────────────────────────────────────────────────

You: what about their competitors?

  Processing...
```

The follow-up works without re-specifying Apple — entity memory resolves *"their"* automatically.

---

## Architecture Deep-Dive

### The Shared State

Every agent reads from and writes back to a single `AgentState` TypedDict defined in `state.py`. LangGraph passes this through every node. Key design choices:

- `conversation_history` uses `operator.add` as its merge function — agents **append** to it, never overwrite
- `_research_summary` is an internal pipeline field — it flows from Research → Validator → Synthesis but is never shown to the user directly
- All other fields are last-writer-wins

### Pipeline Flow

```
User Input
    │
    ▼
[Stage 0] Raw Input Capture (main.py)
    │
    ▼
[Stage 1] Normalizer Node
    ├── Layer 1: Rule-based cleaning (whitespace, duplicates, punctuation)
    └── Layer 2: HuggingFace spell correction (optional, local, free)
    │
    ▼
[Stage 2] Entity Extraction (inside Clarity Agent)
    ├── Regex proper-noun matching
    ├── 80+ company name dictionary
    ├── Ticker symbol recognition (TSLA, AAPL, etc.)
    └── Coreference resolution ("their" → last confirmed company)
    │
    ▼
[Stage 3] Clarity Agent (LLM)
    ├── [CLEAR] ──────────────────────────────────┐
    └── [NEEDS CLARIFICATION]                     │
            │                                     │
            ▼                                     │
    [Stage 3b] Human-in-the-Loop Interrupt        │
        (graph halts, user types answer,          │
         resumes back to normalizer)              │
                                                  ▼
                                    [Stage 4] Research Agent (LLM + Tavily)
                                        ├── Parallel Tavily searches per intent
                                        ├── LLM assembles raw results into summaries
                                        └── Mathematical confidence scoring (0–10)
                                                  │
                                    [Confidence ≥ 6] ──── [Confidence < 6]
                                                  │               │
                                                  │               ▼
                                                  │    [Stage 5] Validator Agent (LLM)
                                                  │        ├── [SUFFICIENT] ────┐
                                                  │        └── [INSUFFICIENT]   │
                                                  │               │             │
                                                  │        (retry, max 3x)      │
                                                  │               └─────────────┘
                                                  │                             │
                                                  └─────────────────────────────┘
                                                                  │
                                                                  ▼
                                                    [Stage 6] Synthesis Agent (LLM)
                                                        └── Markdown response + follow-ups
                                                                  │
                                                                  ▼
                                                    [Stage 7] Output + Loop
```

### Why Mathematical Confidence Scoring?

Asking an LLM *"how confident are you?"* is unreliable. LODESTONE computes confidence from three objective factors instead:

| Factor | Weight | What it measures |
|---|---|---|
| Volume | up to 4 pts | How many Tavily results were returned |
| Diversity | up to 3 pts | How many unique source domains |
| Relevance | up to 2 pts | Tavily's own per-result relevance scores |
| Recency | up to 1 pt | How recent results are (news/financials only) |

Score ≥ 6 → route directly to Synthesis. Score < 6 → send to Validator first.

### Why Entity Extraction Before the LLM?

Extracting a company name from text is a **pattern-matching problem**, not a reasoning problem. A curated dictionary of 80+ companies plus a capitalization heuristic outperforms a general-purpose LLM for this specific task — and costs nothing. The LLM only confirms or adjusts what the extractor found.

### Two-Step Research Design

The Research Agent separates the work into two distinct steps to prevent hallucination:

- **Step A (Tool call):** Tavily fetches raw results. No LLM involved. Just facts.
- **Step B (LLM assembly):** LLM is given *only* the Tavily snippets and asked to summarize them. It cannot invent facts because it has no opportunity to — the search results are the only input.

### All Prompts in One File

All six LLM prompt functions live in `prompts.py`. Every prompt:
- Returns strict JSON — agents parse the output, they don't free-text match
- Contains explicit **negative constraints** (what *not* to do) — LLMs hallucinate less with negative guidance
- Is a plain function — easy to test, easy to tune, no hidden state

---

## Configuration

All tunable values live in `config.py`. Nothing is hardcoded inside agents.

| Constant | Default | Effect |
|---|---|---|
| `CONFIDENCE_THRESHOLD` | `6` | Research → Validator if score below this |
| `MAX_RESEARCH_RETRIES` | `3` | Validator loop hard cap |
| `MIN_SOURCES_REQUIRED` | `2` | Minimum sources for Validator to pass |
| `TAVILY_MAX_RESULTS` | `5` | Results per Tavily search call |
| `TAVILY_SEARCH_DEPTH` | `"advanced"` | Tavily depth (`"basic"` is faster/cheaper) |
| `MAX_HISTORY_MESSAGES` | `10` | How many past messages agents can see |
| `MODELS["primary"]` | `"llama3-8b-8192"` | LLM for Clarity, Validator, Synthesis |
| `MODELS["research"]` | `"llama3-8b-8192"` | LLM for Research Agent |
| `MODELS["normalizer_hf"]` | `"oliverguhr/..."` | HF model (set `None` to disable) |

---

## Supported Research Intents

The system decomposes any query into up to four research dimensions:

| Intent | What Tavily searches for |
|---|---|
| `news` | Latest news, recent developments, press releases |
| `financials` | Revenue, earnings, quarterly/annual results |
| `leadership` | CEO, executive team, management changes |
| `competitors` | Market landscape, industry rivals, positioning |

A broad query like *"tell me about Apple"* triggers all four in parallel.  
A specific query like *"Apple's revenue"* triggers only `financials`.

---

## Entity Memory & Coreference Resolution

The system tracks the confirmed company across conversation turns in `entity_memory`:

```python
{
    "company_name": "Apple",
    "ticker":       "AAPL",
    "last_turn":    2
}
```

**Freshness check:** If more than 5 turns have passed since a company was last confirmed, the memory is treated as stale. *"What about their CEO?"* after 6 unrelated messages will trigger a clarification question instead of silently assuming the old company.

**Coreference:** Pronouns like *"they"*, *"their"*, *"it"*, *"the company"* are resolved deterministically — before the LLM is called — using the entity memory.

---

## Known Limitations

- **Private companies** (SpaceX, OpenAI, Stripe, etc.) have limited financial data on the web. The system will find news and leadership data but financials will be sparse.
- **Very recent events** (< 24 hours) may not be indexed by Tavily yet.
- **The HuggingFace spell correction model** downloads ~500 MB on first run. Use `--no-hf` to skip it — the rule-based Layer 1 still handles basic noise.
- **Rate limits:** Groq free tier allows ~30 requests/minute. Each full query uses 2–4 LLM calls. You'll rarely hit this in normal use.

---

## Requirements

```
langgraph>=0.2.0
langchain-core>=0.2.0
langchain-groq>=0.1.0
tavily-python>=0.3.0
python-dotenv>=1.0.0

# Optional — HuggingFace spell correction
# transformers>=4.40.0
# torch>=2.0.0
```

---

## License

MIT — use freely, attribution appreciated.
