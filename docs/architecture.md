# PB Copilot — Architecture

A privacy-first personal finance and research assistant built with Google's Agent
Development Kit (ADK) and Gemini. PB Copilot reads a structured financial planner and
provides prioritized, number-backed guidance, and researches topics on demand using live
web search — coordinating specialist agents through a single orchestrator.

> Educational guidance only — not a substitute for a licensed financial advisor.

---

## Overview

PB Copilot is a multi-agent system. A user asks a question in natural language; an
orchestrator routes it to the right specialist agent (or both), and the response is
assembled and returned. The design goal is a system that is modular, reliable, and
privacy-respecting by default.

Two specialist agents:

- **Finance Advisor** — reads the user's financial profile (income, expenses, assets,
  debts, insurance, goals) and advises in a sensible priority order: emergency fund →
  insurance → high-interest debt → goal and retirement investing.
- **Research Analyst** — answers news, stock, mutual-fund, and product-comparison
  questions using live Google Search, with cited sources.

The two collaborate: a query like "research this fund and tell me if I should invest given
my finances" invokes both — the Research Analyst gathers facts, the Finance Advisor assesses
fit against the user's actual situation, and the answers merge into one verdict.

---

## Architecture at a glance

```
User query
    │
    ▼
Orchestrator ── routes by matching intent to agent skill declarations
    ├── Finance Advisor ──► finance tools (MCP server; falls back to direct
    │                        in-process calls if MCP misbehaves)
    └── Research Analyst ──► Google Search (built-in tool, cache-on-success)
    │
    ├── every model call runs through the Agent Harness
    │     (context + redaction → throttle → timeout/retry → validation
    │      → self-check → observability logging)
    │
    ▼
Assembled, validated response
```

---

## Key design elements

### Skill-based routing
Each agent capability is declared as a self-contained skill (`skills/<name>/SKILL.md`) stating
its purpose, trigger intents, inputs, tools, outputs, and guardrails. The orchestrator routes
by matching user intent against these declarations rather than hardcoded keywords, so new
capabilities can be added by dropping in a new skill with no orchestrator changes.

### Agent harness
Every model call runs through a single reusable harness (`harness/agent_runner.py`) that
provides:
- **Context management** — the profile is never dumped into the prompt; it reaches the model
  only as tool results (and the aggregate `get_full_financial_analysis` returns all areas in
  one call). Loading a new profile resets the session, so context never bleeds between people.
- **Privacy redaction** — strips personally identifiable information before the model call.
- **Rate limiting** — a client-side throttle (`harness/throttle.py`) paces calls *under* the
  free-tier per-minute cap to avoid 429s in the first place; per-model (Flash-Lite gets 2x).
- **Control flow** — per-call timeout; retry-with-backoff that waits *longer* for a 429
  (honoring the server's retry delay, with jitter) than for a timeout; capped tool-call
  iterations per turn.
- **Output validation** — checks that reported figures match tool computations, sources are
  cited, and required disclaimers are present; repairs once or returns a safe fallback.
- **Self-check** — an optional light consistency pass over financial advice before it is
  finalized (off by default to conserve API quota; the deterministic figure-grounding check
  above already catches invented numbers for free).
- **Observability** — every turn is recorded as a structured JSONL line (agents/tools run,
  latency, retries, validation outcome, redacted profile tag), with per-agent spans and
  per-model-call latency/token metadata; see `observability/tracker.py`.

### Privacy by design
Personally identifiable information (name, account number, etc.) is redacted locally and
replaced with stable tokens before any model call, then re-hydrated in the final response.
Financial figures are processed by the model to generate advice; identifiers are not — we
do not claim the model never sees any of the user's data, since that would be false.
This project uses the free Google AI Studio tier with synthetic data only. For real
personal use, a no-training tier (paid Gemini API or Vertex AI) is required, as the free
tier may use data to improve products.

### Finance tools as an MCP server
The financial calculations (savings rate, net worth, emergency-fund status, debt payoff
ordering, life-insurance/HLV gap, retirement corpus and SIP, goal SIP) are exposed as an
MCP server and consumed by the Finance Advisor as tools, keeping the calculation layer
modular and independently testable. A direct in-process fallback (`PB_FINANCE_TOOLS=direct`)
is kept alongside it, so a misbehaving MCP server never takes the agent down.

### Memory & context handling
Two distinct layers, deliberately kept separate:

**Persistent state — a JSON datastore.** The parsed financial profile lives as an in-memory
dict mirrored to a JSON file (`data/active_profile.json`) on every change — a lightweight
create/read/update store (no database, single active profile). It survives process restarts
and is what the separate MCP-server process reads to stay in sync with the agent.

**Conversation context — in memory.** The turn-by-turn conversation is held in RAM by ADK's
`InMemorySessionService` (the session's event history is what ADK re-sends to the model each
turn — i.e. short-term conversational memory). It is *not* persisted to disk. Crucially, it is
**scoped to one profile**: uploading a new planner starts a fresh, empty ADK session (and the
UI clears the chat to match), so one person's conversation never bleeds into another's
context and the window can't grow unbounded across profiles.

**Context-window discipline.** Within a single profile, the model's context stays small
because the profile is never pasted into the prompt — it reaches the model only as tool
results, and the aggregate `get_full_financial_analysis` returns every area in one call.
Long-term / vector memory (e.g. ADK Memory Bank) is deliberately deferred.

---

## Data model

The input is an Excel financial planner (see `data/` for two synthetic samples). A parser
converts it into a single internal profile structure used throughout the system:

```
profile = {
  "profile":   { age, dependents, risk_appetite, city, ... },
  "income_monthly":   { ... },
  "expenses_monthly": { ... },
  "assets":    { ... },
  "debts":     [ { type, balance, rate_pct, emi }, ... ],   # may be empty
  "insurance": { life_cover, health_cover },
  "goals":     [ { name, cost, years, inflation }, ... ],
  "derived":   { net_worth, savings_rate, emergency_fund_months,
                 retirement_corpus, retirement_sip, hlv_gap, ... }
}
```

Totals are recomputed in the finance tools rather than trusted from the spreadsheet.

---

## Components

| Component | Responsibility |
|---|---|
| `agents/orchestrator.py` | Routes queries via skill declarations; merges responses |
| `agents/finance_agent.py` | Financial advisor; uses the finance MCP tools (direct-call fallback) |
| `agents/research_agent.py` | Research analyst; uses Google Search |
| `skills/<name>/SKILL.md` | Declarative capability definitions driving routing |
| `skill_registry.py` | Loads `SKILL.md` files and builds the orchestrator's tools + routing instruction from them |
| `harness/agent_runner.py` | Reliability wrapper around every model call (the six layers) |
| `harness/throttle.py` | Client-side per-model rate limiter (keeps calls under the quota) |
| `observability/tracker.py` | Structured per-turn JSONL logging, agent spans, LLM-call metadata, unified console |
| `privacy/redactor.py` | PII tokenization/re-hydration logic (pure, no ADK dependency) |
| `privacy/adk_privacy.py` | Wires redaction into ADK's `before_model_callback` on every agent |
| `tools/finance_tools.py` | Financial calculations (incl. `full_analysis` aggregate) |
| `tools/finance_mcp_server.py` | Exposes finance tools over MCP |
| `tools/search_cache.py` | Cache-on-success wrapper for ResearchAgent |
| `tools/excel_parser.py` | Parses the Excel planner into the internal profile |
| `memory/profile_store.py` | Loads, saves, and updates the profile |
| `api/server.py` | FastAPI: `/chat`, `/profile/upload`, `/profile/active`, `/health` (UI-agnostic) |
| `ui/app.py` | Gradio interface — upload a planner, chat; clears on new upload |
| `tests/` | Unit tests for the finance calculations and guardrails |

---

## Known limitations (privacy redaction)

Redaction was audited by tracing every path user data can take to the model or
a tool. Two real gaps were found and fixed (a profile-name leak into logs/API
responses, and an MCP JSON-text redaction bug that could corrupt numeric
figures). Three narrower gaps remain, by choice, and are documented here
rather than silently left:

- **Tool-call arguments are not redacted before dispatch.** `before_model_callback`
  redacts the *record* of a tool call for future model context, but the tool
  itself is invoked with whatever arguments the model produced, before that
  redaction runs. In this app the only tool argument the model ever supplies is
  a free-text goal name (`plan_goal(goal_name)`), which carries no realistic PII
  risk — but the mechanism does not generally protect tool-call arguments, only
  tool-call results and the surrounding conversation.
- **Google Search query construction is opaque.** `google_search` is
  model-internal grounding: Gemini decides the search query and calls Google's
  backend itself, inside the same API call. We redact the prompt Gemini
  receives, but have no code-level hook to inspect or redact the search query
  it derives from that prompt — so the guarantee is "the input was clean," not
  "the search request was verified clean."
- **`tools/finance_mcp_server.py --selftest` prints the raw profile name to
  stdout.** It's a manual, opt-in debug entry point (never invoked during
  normal agent operation), but if you pipe or share its output, the real name
  is in it.

---

## Technology

Google ADK · Gemini · MCP · FastAPI · Gradio · openpyxl · pytest
