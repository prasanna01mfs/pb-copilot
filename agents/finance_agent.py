"""FinanceAgent — advisory logic backed by finance tools.

The agent no longer answers from general knowledge alone. It calls typed tools
that read the ACTIVE profile (loaded from an Excel planner into the profile
store) and return computed numbers. The agent's job is to turn those numbers
into prioritized, plain-English advice — never to invent figures.

Tool design (harness layer 2 — tool integration): each tool below is a thin
ADK adapter over a pure function in tools/finance_tools.py. The adapters:
- take only small, model-friendly arguments (or none) — the profile is read
  from the store, so the model can't pass a wrong/huge dict;
- return structured dicts, not prose;
- catch errors and return a clean {"error": ...} dict instead of raising a raw
  stack trace at the model (the reliability harness builds on this).
"""
import logging
import os
import sys

from google.adk.agents import Agent

from harness.throttle import throttle_before_model
from memory import profile_store
from observability import tracker as observability
from privacy.adk_privacy import redaction_before_model
from tools import finance_tools

logger = logging.getLogger("pb.finance")

FINANCE_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest")

# Which tool transport to use:
#   "mcp"    (default) — talk to the local MCP server (tools/finance_mcp_server.py)
#   "direct"           — call the in-process Python adapters below
# The direct path is the proven fallback: if MCP misbehaves in a demo, one env
# var (PB_FINANCE_TOOLS=direct) restores the working agent without code changes.
FINANCE_TOOLS_MODE = os.getenv("PB_FINANCE_TOOLS", "mcp").lower()


def _safe(fn, *args):
    """Run a calculator against the active profile; return a clean error dict on failure."""
    try:
        return fn(profile_store.get_active(), *args)
    except Exception as exc:  # noqa: BLE001 — deliberately convert to a clean tool error
        return {"error": str(exc)}


# --- ADK tool adapters (docstrings are read by the model) -------------------

def get_full_financial_analysis() -> dict:
    """Get the user's COMPLETE financial analysis in one call: profile, cash flow,
    net worth, emergency fund, debts (with payoff order), insurance gaps,
    retirement, and goals. Prefer THIS for any general review or "what should I
    fix first" question — it returns everything at once. Use the individual
    tools only to drill into a single area the user specifically asks about."""
    if not profile_store.has_active():
        return {"error": "No profile loaded. Ask the user to upload a planner (.xlsx) first."}
    return _safe(finance_tools.full_analysis)


def get_profile_summary() -> dict:
    """Get the loaded user's core financial snapshot: income, expenses, net
    monthly savings, savings rate, total assets, total debt and net worth."""
    if not profile_store.has_active():
        return {"error": "No profile loaded. Ask the user to upload a planner (.xlsx) first."}
    p = profile_store.get_active()
    sr = finance_tools.savings_rate(p)
    nw = finance_tools.net_worth(p)
    return {
        "name": p["profile"]["name"],
        "age": p["profile"]["age"],
        "dependents": p["profile"]["dependents"],
        "risk_appetite": p["profile"]["risk_appetite"],
        **sr,
        **nw,
    }


def check_emergency_fund() -> dict:
    """Get emergency-fund status: months of expenses covered by liquid assets,
    the 6-month target amount, and the shortfall (if any)."""
    return _safe(finance_tools.emergency_fund_status)


def analyze_debts() -> dict:
    """Analyze the user's debts: total owed, which debts are high-interest, and
    the avalanche (highest-interest-first) payoff order. Returns has_debt=False
    cleanly when the user has no debt."""
    return _safe(finance_tools.debt_analysis)


def check_insurance_gap() -> dict:
    """Assess life insurance (Human Life Value: how much cover is needed vs held)
    and health insurance adequacy, including whether the user has any cover."""
    return _safe(finance_tools.insurance_gap)


def plan_retirement() -> dict:
    """Compute the retirement corpus required and the monthly SIP needed to
    reach it, given the user's age, target retirement age and assumptions."""
    return _safe(finance_tools.retirement_plan)


def plan_goal(goal_name: str) -> dict:
    """Compute the future cost and monthly SIP for one financial goal, matched by
    name (partial match allowed). If not found, returns the list of the user's
    available goals so you can ask the user which one they meant.

    Args:
        goal_name: The goal to plan for, e.g. "house down payment" or "car".
    """
    return _safe(finance_tools.goal_plan, goal_name)


FINANCE_INSTRUCTION = """
You are FinanceAgent, the personal financial advisor inside PB Copilot.

HARD RULES
- Base every number you state on a tool result. NEVER invent or estimate
  financial figures. If you need a number, call the relevant tool first.
- If no profile is loaded (a tool returns an error saying so), tell the user to
  upload a planner (.xlsx) and stop.
- End EVERY response with exactly this line:
  "Educational guidance only, not licensed financial advice."

TOOL USE — be efficient:
- For a general review or a "what should I fix first / how are my finances"
  question, call get_full_financial_analysis ONCE — it returns every area
  (cash flow, emergency fund, debts + payoff order, insurance, retirement,
  goals) in a single call. Do NOT then call the individual tools for the same
  data.
- Use the granular tools (check_emergency_fund, analyze_debts,
  check_insurance_gap, plan_retirement, plan_goal, get_profile_summary) ONLY
  when the user asks specifically about that one area.

ADVISORY PRIORITY ORDER — always reason and recommend in this sequence:
  1. Emergency fund  — build to ~6 months of expenses first.
  2. Insurance       — adequate term life (HLV) + health cover.
  3. High-interest debt — clear anything above ~12% before investing; a 42%
     credit card is a guaranteed loss that beats any expected market return.
  4. Goals & retirement investing — only once 1–3 are on track.

STYLE
- Lead with what to fix FIRST and why, backed by the actual numbers.
- Be concrete: cite the computed figures (months covered, shortfall, cover gap,
  payoff order, SIP amounts).
- If the user has none of something (no debt, no insurance), don't error — say
  "you don't have X yet" and explain how to start.
- Never promise guaranteed or specific market returns.

Stay in your lane: news, live prices, and product/fund research belong to the
research specialist — defer those rather than answering them yourself.
"""

# The direct in-process tools (also the fallback path). The aggregate is listed
# first — it's the default/preferred path; the granular tools remain for
# drill-down into a single area.
_DIRECT_TOOLS = [
    get_full_financial_analysis,
    get_profile_summary,
    check_emergency_fund,
    analyze_debts,
    check_insurance_gap,
    plan_retirement,
    plan_goal,
]


def _build_finance_tools():
    """Return the finance tools: an MCP toolset if enabled+buildable, else direct.

    Keeping direct as the fallback honours the plan's "protect the working thing"
    rule — MCP is the showcase, but a broken server must never take the agent
    down. If MCP is selected but can't be wired, we log and use the direct path.
    """
    if FINANCE_TOOLS_MODE == "direct":
        logger.info("finance tools: DIRECT (in-process functions)")
        return _DIRECT_TOOLS
    try:
        from google.adk.tools.mcp_tool import McpToolset, StdioConnectionParams
        from mcp import StdioServerParameters
        from tools.finance_mcp_server import TOOL_NAMES

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        server_path = os.path.join(project_root, "tools", "finance_mcp_server.py")
        toolset = McpToolset(
            connection_params=StdioConnectionParams(
                server_params=StdioServerParameters(
                    command=sys.executable,          # the venv's python
                    args=[server_path],              # absolute path (MCP requires absolute)
                    cwd=project_root,
                ),
                timeout=30,
            ),
            tool_filter=TOOL_NAMES,  # expose exactly our six finance tools
        )
        logger.info("finance tools: MCP server (%s)", server_path)
        return [toolset]
    except Exception as exc:  # noqa: BLE001 — never let MCP wiring break the agent
        logger.warning("MCP toolset unavailable (%s); falling back to DIRECT tools", exc)
        return _DIRECT_TOOLS


finance_agent = Agent(
    name="finance_agent",
    model=FINANCE_MODEL,
    description=(
        "Personal financial advisor for the user's own situation: emergency "
        "fund, insurance gaps, debt payoff priority, retirement and goal "
        "planning — backed by calculators over the loaded profile."
    ),
    instruction=FINANCE_INSTRUCTION,
    tools=_build_finance_tools(),
    # Observability: log start/end + duration of this agent's invocation.
    before_agent_callback=observability.agent_span_before,
    after_agent_callback=observability.agent_span_after,
    # throttle_before_model FIRST (pace under the per-minute quota), then the
    # latency timer, then redaction.
    before_model_callback=[throttle_before_model, observability.llm_call_before,
                           redaction_before_model],
    after_model_callback=observability.llm_call_after,
)
