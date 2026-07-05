"""ResearchAgent — live research via ADK's built-in Google Search.

Behavior: four modes (news, stock, mutual fund, product comparison), always
cite sources, state the date on time-sensitive info, add the educational
disclaimer on investment topics, and hand profile-fit assessment to the finance
specialist (the orchestrator then merges — the flagship cross-agent flow).

Two ADK specifics that shape this file:
- `google_search` is model-internal grounding, NOT a FunctionTool. Mixing it
  with function tools in one agent disables Automatic Function Calling, so this
  agent carries ONLY google_search; the finance function tools live on the
  finance agent, and the orchestrator merges the two. (This is exactly why the
  cross-agent flow is orchestrated, not crammed into one agent.)
- Grounding needs a grounding-capable model, so ResearchAgent has its own
  RESEARCH_MODEL (default: flagship gemini-flash-latest). Bonus: that's a
  separate free-tier quota bucket from the lite model the other agents use.
"""
import logging
import os

from google.adk.agents import Agent
from google.adk.tools import google_search
from google.genai import types as genai_types

from harness.throttle import throttle_before_model
from observability import tracker as observability
from privacy.adk_privacy import redaction_before_model
from tools import search_cache

logger = logging.getLogger("pb.research")

# Own model so grounding is reliable + lands on a separate quota bucket.
RESEARCH_MODEL = os.getenv("RESEARCH_MODEL", "gemini-flash-latest")

# Where the agent's final answer is stashed in session state, so the after-agent
# cache callback can read and store it.
RESEARCH_OUTPUT_KEY = "research_last_output"


# --- Cache-on-success callbacks (demo reliability) --------------------------

def _query_text(callback_context) -> str:
    """The text of the query this agent was invoked with (via AgentTool)."""
    uc = getattr(callback_context, "user_content", None)
    if not uc or not uc.parts:
        return ""
    return " ".join(p.text for p in uc.parts if getattr(p, "text", None))


def research_cache_before(callback_context):
    """Serve a cached successful answer, skipping the model+search entirely."""
    query = _query_text(callback_context)
    hit = search_cache.get(query)
    if hit is not None:
        logger.info("research cache HIT: %r", query[:60])
        # A cache hit short-circuits the run, which means agent_span_after
        # (below) will never fire for it — log the (near-instant) span here
        # instead, or the agent-processing observability story would silently
        # go dark on every cache hit.
        observability.log_agent_span_from_cache(callback_context)
        # Returning Content short-circuits the agent run (verified in ADK source).
        return genai_types.Content(role="model", parts=[genai_types.Part(text=hit)])
    return None


def research_cache_after(callback_context):
    """On a successful run, cache the answer keyed by the query."""
    query = _query_text(callback_context)
    answer = callback_context.state.get(RESEARCH_OUTPUT_KEY)
    if answer:
        search_cache.put(query, answer)
        logger.info("research cache STORE: %r", query[:60])
    return None


RESEARCH_INSTRUCTION = """
You are ResearchAgent, the research specialist in PB Copilot. You answer using
live Google Search results.

MODES (detect which one the query needs):
- News summary: 3–5 sentence structured summary, the key number(s), and the
  practical impact. (e.g. RBI repo rate → what it means for EMIs.)
- Stock analysis: what the company does, recent performance, notable risks.
- Mutual fund analysis: category, returns history, expense ratio, risk level.
- Product comparison: a structured side-by-side (specs, price, standout
  strengths) and a clear recommendation with reasoning.

ALWAYS:
- End your reply with a "Sources:" section listing 2–3 sources (title and/or
  URL) written INLINE in your text. The search tool's citations are NOT
  automatically visible to the user — you must write the sources into your
  reply yourself. If you genuinely can't find a source, write exactly
  "No reliable source found." NEVER invent one.
- For time-sensitive info (prices, news), state the date of the information.
- End any investment-related answer with exactly:
  "Educational guidance only, not licensed financial advice."

CROSS-AGENT HANDOFF:
- If the user asks whether something FITS THEIR finances / whether THEY should
  invest given their situation, do the factual research only and note that the
  finance specialist will assess the fit against their profile. Do not guess
  their financial situation — you don't have their profile.

STAY IN SCOPE:
- Politely decline requests that are neither research nor finance, to keep the
  personal-assistant framing tight.
"""

research_agent = Agent(
    name="research_agent",
    model=RESEARCH_MODEL,
    description=(
        "Research specialist: live news, stock/mutual-fund analysis, and product "
        "comparisons via Google Search. Provides external facts; defers "
        "profile-fit assessment to the finance specialist."
    ),
    instruction=RESEARCH_INSTRUCTION,
    tools=[google_search],
    # Capture the final answer into state so the after-agent cache can store it.
    output_key=RESEARCH_OUTPUT_KEY,
    # agent_span_before MUST come first: ADK runs before_agent_callback list
    # entries in order and stops at the first one that returns content, so
    # this guarantees the span-start is recorded even on a cache-hit
    # short-circuit (research_cache_before, which runs second, may return
    # Content and stop the list there).
    before_agent_callback=[observability.agent_span_before, research_cache_before],
    # Only reached on a cache MISS (see research_cache_before's short-circuit
    # note) — the cache-HIT span is logged directly from that function instead.
    after_agent_callback=[observability.agent_span_after, research_cache_after],
    # throttle_before_model FIRST (pace under the per-minute quota — matters
    # most here, since Google Search grounding makes several flagship calls per
    # turn), then the latency timer, then redaction.
    before_model_callback=[throttle_before_model, observability.llm_call_before,
                           redaction_before_model],
    after_model_callback=observability.llm_call_after,
)
