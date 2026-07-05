"""Orchestrator — routes each query to the specialist whose SKILL matches intent.

Two design choices worth calling out:

1. AgentTool, not sub_agents/transfer (spine-before-muscles). With AgentTool the
   orchestrator calls a specialist and STAYS in control afterward, so it can call
   two specialists in one turn and merge them (the flagship "research a fund AND
   assess fit for my profile" flow). LLM-delegation transfer hands off control
   permanently and would need a rewrite for that.

2. Skill-based routing, not hardcoded keywords. The orchestrator's tools AND its
   routing instruction are BUILT FROM the skill registry (skills/<name>/SKILL.md
   via skill_registry.py, at the project root — skills/ itself is pure data so
   ADK's skill discovery never trips over a stray __pycache__). Nothing here
   names finance vs research explicitly — so adding skills/<x>/SKILL.md that
   declares its agent adds a route with NO edit to this file. Docs and routing
   come from one source of truth.
"""
import os

from google.adk.agents import Agent
from google.adk.tools import AgentTool

from harness.throttle import throttle_before_model
from observability import tracker as observability
from privacy.adk_privacy import redaction_before_model
from skill_registry import build_routing_instruction, load_routable_skills

ORCHESTRATOR_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest")

# Discover the specialists from their skill declarations. This list — the tools
# and the routing instruction below — is entirely data-driven from skills/.
_SKILLS = load_routable_skills()
if not _SKILLS:
    raise RuntimeError(
        "No routable skills found in skills/. Each specialist needs a "
        "skills/<name>/SKILL.md with metadata.agent_ref set."
    )

root_agent = Agent(
    name="orchestrator",
    model=ORCHESTRATOR_MODEL,
    description="Routes queries to the specialist whose declared skill matches the user's intent, and merges answers when several apply.",
    instruction=build_routing_instruction(_SKILLS),
    tools=[AgentTool(skill.agent) for skill in _SKILLS],
    # Observability: log start/end + duration of the orchestrator's own span
    # (nested specialist spans are logged separately by their own callbacks).
    before_agent_callback=observability.agent_span_before,
    after_agent_callback=observability.agent_span_after,
    # throttle_before_model FIRST — pace calls under the per-minute quota before
    # anything else. Then llm_call_before (so its timer covers only the real
    # round trip, not the throttle wait), then redaction of the outgoing PII.
    before_model_callback=[throttle_before_model, observability.llm_call_before,
                           redaction_before_model],
    after_model_callback=observability.llm_call_after,
)
