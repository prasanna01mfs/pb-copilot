"""Skill registry — the single source of truth that turns SKILL.md files into
routing artifacts, so docs and routing can never drift apart.

Each specialist declares itself in skills/<name>/SKILL.md (ADK-native skill
format: YAML frontmatter + markdown body). This module discovers every skill
with ADK's `list_skills_in_dir`, loads it with `load_skill_from_dir`, resolves
the `metadata.agent_ref` to the actual agent object, and exposes:

  * the loaded skills (for building the orchestrator's routing instruction), and
  * skill -> agent bindings (for building the orchestrator's AgentTools).

Consequence: dropping a new skills/<x>/SKILL.md that names its agent is enough
to add a new route — the orchestrator reads from here and never needs editing.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path

from google.adk.skills import list_skills_in_dir, load_skill_from_dir

# skills/ is a PURE DATA directory (only <name>/SKILL.md) — this loader lives at
# the project root, not inside it, so no Python __pycache__ ever lands among the
# skills for ADK's discovery to trip over.
SKILLS_DIR = Path(__file__).resolve().parent / "skills"


@dataclass
class RoutableSkill:
    """A skill paired with the agent that fulfils it — everything the
    orchestrator needs to route by intent, sourced entirely from SKILL.md."""
    name: str
    description: str
    when_to_use: list[str]
    examples: list[str]
    agent: object  # the ADK Agent instance this skill routes to

    @property
    def agent_name(self) -> str:
        return self.agent.name


def _resolve_agent(agent_ref: str):
    """Import an agent instance from a 'module.path:variable' reference."""
    module_path, _, var = agent_ref.partition(":")
    if not module_path or not var:
        raise ValueError(f"agent_ref must be 'module:variable', got {agent_ref!r}")
    return getattr(importlib.import_module(module_path), var)


def load_routable_skills() -> list[RoutableSkill]:
    """Discover + load every skill, bound to its agent. Sorted for stable order.

    Skills whose frontmatter omits `metadata.agent_ref` are skipped (they're
    not routable to an agent), so the loader is tolerant of doc-only skills.
    """
    skills: list[RoutableSkill] = []
    for name in sorted(list_skills_in_dir(SKILLS_DIR)):
        skill = load_skill_from_dir(SKILLS_DIR / name)
        md = skill.frontmatter.metadata or {}
        agent_ref = md.get("agent_ref")
        if not agent_ref:
            continue
        skills.append(
            RoutableSkill(
                name=skill.frontmatter.name,
                description=" ".join(skill.frontmatter.description.split()),
                when_to_use=md.get("when_to_use", []),
                examples=md.get("example_queries", []),
                agent=_resolve_agent(agent_ref),
            )
        )
    return skills


def build_routing_instruction(skills: list[RoutableSkill]) -> str:
    """Render the orchestrator's routing instruction FROM the skill declarations.

    No hardcoded per-agent rules live here — every routing hint is generated
    from the SKILL.md files, so the docs the humans read and the instructions
    the model routes on are literally the same source.
    """
    lines = [
        "You are the PB Copilot orchestrator. You never answer finance or "
        "research questions yourself — you route each query to the specialist "
        "tool(s) whose skill best matches the user's intent, then relay or "
        "merge their answers into one reply.",
        "",
        "Available specialist skills (match the query's intent to these):",
    ]
    for s in skills:
        lines.append(f"\n• Tool `{s.agent_name}` — skill \"{s.name}\": {s.description}")
        if s.when_to_use:
            lines.append("  Use when: " + "; ".join(s.when_to_use))
        if s.examples:
            lines.append("  Example queries: " + " | ".join(s.examples[:3]))
    lines += [
        "",
        "CROSS-SKILL QUERIES — when a query needs MORE THAN ONE skill, call each "
        "relevant tool, then MERGE their outputs into ONE answer. In particular, "
        "a query like \"should I invest in / buy <X> given my profile / "
        "situation / finances\" or \"is that fine?\" needs BOTH: first the "
        "research skill for the facts about <X>, then the finance skill to "
        "assess fit against the loaded profile.",
        "",
        "FORMAT THE MERGED ANSWER IN EXACTLY THREE SECTIONS, IN THIS ORDER:",
        "  1. '### 📊 Your Profile Analysis' — the finance specialist's "
        "assessment of the user's OWN situation (emergency fund, insurance, "
        "debt, goals) relevant to this question. Write this section's body as "
        "a Markdown blockquote (every line starts with '> ') so it renders "
        "visually distinct from the rest of the answer.",
        "  2. '### 🔍 Research Analysis: <fund/stock/product/topic name>' — a "
        "concise summary of the research specialist's facts (category, "
        "performance, risk, price, etc., as relevant). Plain text, not a "
        "blockquote.",
        "  3. '### ✅ Conclusion — What To Do Next' — the actual verdict and a "
        "concrete next step. This must be unambiguous, not hedged: if any "
        "higher-priority gap (emergency fund, insurance, high-interest debt) is "
        "still open, say plainly that fixing it comes BEFORE this "
        "product/investment, even if the product itself is a good one — a good "
        "fund is still the wrong move right now. Do not soften this into "
        "vague 'could fit once covered' language; state the concrete next "
        "action (e.g. 'top up your emergency fund by ₹X before investing').",
        "If the query only needed ONE skill (no cross-agent merge), use "
        "section 1 and 3 only (skip the Research Analysis section) — still "
        "highlight the profile analysis as a blockquote and close with a clear "
        "conclusion/next-step section.",
        "",
        "WHEN YOU MERGE, preserve two things from the specialists' outputs in "
        "your final answer: (a) the research specialist's 'Sources:' section, "
        "copied through verbatim, and (b) for any investment/finance topic, end "
        "with exactly: 'Educational guidance only, not licensed financial "
        "advice.' Do not drop them. Also normalize currency: the user's profile "
        "is in Indian Rupees, so restate every monetary figure in the merged "
        "answer as ₹ — if either specialist reported a figure in $ or another "
        "currency, convert it before merging rather than mixing symbols.",
        "",
        "Always call at least one specialist before answering. If nothing "
        "matches, politely decline and state what PB Copilot can help with.",
    ]
    return "\n".join(lines)
