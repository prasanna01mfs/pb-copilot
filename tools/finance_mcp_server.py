"""Local MCP server exposing the finance calculators as MCP tools.

Why this exists: the same pure functions in finance_tools.py are served two ways
— directly (in-process) and over MCP (this file). Wrapping them as an MCP server
demonstrates the Model Context Protocol integration and lets the finance tools
run as their OWN process, decoupled from the agent.

Cross-process state: this server runs in a SEPARATE process from the agent, so
it cannot see the agent's in-memory active profile. Instead it reads the profile
the agent already persisted to data/active_profile.json (memory/profile_store.py
writes it on every load/update). Because both sides run the identical
finance_tools functions on the identical profile file, the advice is byte-for-
byte the same whether sourced directly or through MCP — and uploading a new
profile is picked up here on the next tool call.

Run standalone:
    python tools/finance_mcp_server.py            # start the stdio MCP server
    python tools/finance_mcp_server.py --selftest # list tools + one sample calc, exit
"""
from __future__ import annotations

import json
import os
import sys

from mcp.server.fastmcp import FastMCP

# Import the pure calculators by absolute package path so this file works both
# as a module (python -m) and as a script (python tools/finance_mcp_server.py).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import finance_tools  # noqa: E402

# Same location memory/profile_store.py persists the active profile to.
_PROFILE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "active_profile.json"
)

mcp = FastMCP("pb-finance")


def _load_profile() -> dict | None:
    """Read the active profile the agent persisted, or None if nothing loaded."""
    if not os.path.exists(_PROFILE_PATH):
        return None
    with open(_PROFILE_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _with_profile(fn, *args):
    """Run a calculator against the persisted profile; clean error dict if absent.

    Mirrors the direct adapters' contract: never raise across the MCP boundary —
    return a structured error the agent/harness can handle.
    """
    profile = _load_profile()
    if profile is None:
        return {"error": "No profile loaded. Ask the user to upload a planner (.xlsx) first."}
    try:
        return fn(profile, *args)
    except Exception as exc:  # noqa: BLE001 — convert to a clean tool error
        return {"error": str(exc)}


# --- MCP tools (names + docstrings match the direct adapters exactly, so the
#     agent instruction and the harness validation are identical either way) ---

@mcp.tool()
def get_full_financial_analysis() -> dict:
    """Get the user's COMPLETE financial analysis in one call: profile, cash flow,
    net worth, emergency fund, debts (with payoff order), insurance gaps,
    retirement, and goals. Prefer THIS for any general review or "what should I
    fix first" question — it returns everything at once. Use the individual
    tools only to drill into a single area the user specifically asks about."""
    return _with_profile(finance_tools.full_analysis)


@mcp.tool()
def get_profile_summary() -> dict:
    """Get the loaded user's core financial snapshot: income, expenses, net
    monthly savings, savings rate, total assets, total debt and net worth."""
    profile = _load_profile()
    if profile is None:
        return {"error": "No profile loaded. Ask the user to upload a planner (.xlsx) first."}
    sr = finance_tools.savings_rate(profile)
    nw = finance_tools.net_worth(profile)
    return {
        "name": profile["profile"]["name"],
        "age": profile["profile"]["age"],
        "dependents": profile["profile"]["dependents"],
        "risk_appetite": profile["profile"]["risk_appetite"],
        **sr,
        **nw,
    }


@mcp.tool()
def check_emergency_fund() -> dict:
    """Get emergency-fund status: months of expenses covered by liquid assets,
    the 6-month target amount, and the shortfall (if any)."""
    return _with_profile(finance_tools.emergency_fund_status)


@mcp.tool()
def analyze_debts() -> dict:
    """Analyze the user's debts: total owed, which debts are high-interest, and
    the avalanche (highest-interest-first) payoff order. Returns has_debt=False
    cleanly when the user has no debt."""
    return _with_profile(finance_tools.debt_analysis)


@mcp.tool()
def check_insurance_gap() -> dict:
    """Assess life insurance (Human Life Value: how much cover is needed vs held)
    and health insurance adequacy, including whether the user has any cover."""
    return _with_profile(finance_tools.insurance_gap)


@mcp.tool()
def plan_retirement() -> dict:
    """Compute the retirement corpus required and the monthly SIP needed to
    reach it, given the user's age, target retirement age and assumptions."""
    return _with_profile(finance_tools.retirement_plan)


@mcp.tool()
def plan_goal(goal_name: str) -> dict:
    """Compute the future cost and monthly SIP for one financial goal, matched by
    name (partial match allowed). If not found, returns the list of the user's
    available goals so you can ask the user which one they meant.

    Args:
        goal_name: The goal to plan for, e.g. "house down payment" or "car".
    """
    return _with_profile(finance_tools.goal_plan, goal_name)


# Tool names the agent connects to (single source of truth for the ADK filter).
TOOL_NAMES = [
    "get_full_financial_analysis",
    "get_profile_summary",
    "check_emergency_fund",
    "analyze_debts",
    "check_insurance_gap",
    "plan_retirement",
    "plan_goal",
]


def _selftest() -> None:
    """Prove the server process works without needing an MCP client handshake."""
    print(f"MCP server 'pb-finance' — {len(TOOL_NAMES)} tools: {', '.join(TOOL_NAMES)}")
    profile = _load_profile()
    if profile is None:
        print("No active_profile.json yet — upload a profile in the app first.")
        return
    print(f"Active profile: {profile['profile']['name']}")
    print("Sample check_emergency_fund():", json.dumps(check_emergency_fund(), default=str))


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        # Block on stdio: ADK (or any MCP client) spawns this and speaks MCP over
        # stdin/stdout. This is the "server running as its own process".
        mcp.run("stdio")
