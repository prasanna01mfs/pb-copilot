"""Finance calculators — plain Python functions (no ADK, no I/O, fully testable).

These are ordinary functions that take a parsed `profile` dict and return
structured dicts, mirroring the same formulas the source planner spreadsheets
use. They were validated to reproduce the planners' own computed cells
(retirement corpus, goal SIP, HLV cover) within rounding —
see tests/test_finance_tools.py.

Design rules:
- RECOMPUTE every total from line items; never trust the sheet's Total cells.
- Return numbers (structured data), not prose. The agent turns these into advice.
- No crashes on Priya's zeros: empty debts and zero insurance are valid inputs.
"""
from __future__ import annotations

# Assumed annual equity return used for GOAL SIPs. The planners use a fixed 11%
# for goals (verified by solving their own SIP cells) while RETIREMENT SIPs use
# the profile's own pre-retirement ROI. Kept explicit so the assumption is
# visible and tunable rather than a magic number buried in a formula.
GOAL_RETURN = 0.11

# A debt is "high interest" when its rate exceeds what diversified equity might
# reasonably return (~12%). Above this, paying the debt is a guaranteed return
# that beats investing — the core of the "clear the card first" advice.
HIGH_INTEREST_THRESHOLD_PCT = 12.0

# Emergency fund target, in months of expenses (standard planning guidance).
EMERGENCY_FUND_TARGET_MONTHS = 6

# Which asset line items count as "liquid" for emergency-fund purposes. Property
# and long-lock retirement vehicles (EPF/PPF/NPS) are intentionally excluded.
# Matched by substring against normalized asset keys.
LIQUID_ASSET_HINTS = ("savings", "fixed_deposit", "fd", "debt_mutual", "liquid")


def _sum(values) -> float:
    return float(sum(values))


def monthly_income(profile: dict) -> float:
    return _sum(profile["income_monthly"].values())


def monthly_expenses(profile: dict) -> float:
    return _sum(profile["expenses_monthly"].values())


def total_assets(profile: dict) -> float:
    return _sum(profile["assets"].values())


def total_debt(profile: dict) -> float:
    return _sum(d["balance"] for d in profile["debts"])


def _liquid_assets(profile: dict) -> float:
    """Assets reachable quickly in an emergency (savings, FDs, debt/liquid MF)."""
    total = 0.0
    for key, val in profile["assets"].items():
        if any(hint in key for hint in LIQUID_ASSET_HINTS):
            total += val
    return total


def _monthly_sip(future_value: float, annual_return: float, months: int) -> float:
    """Monthly SIP to reach `future_value` in `months` at `annual_return`.

    SIP = FV * (r/12) / ((1+r/12)^months - 1). If months <= 0, the goal is due
    now, so the whole future value is required immediately.
    """
    if months <= 0:
        return float(future_value)
    r = annual_return / 12.0
    if r == 0:
        return future_value / months
    return future_value * r / ((1 + r) ** months - 1)


# --- Core snapshot ----------------------------------------------------------

def savings_rate(profile: dict) -> dict:
    income = monthly_income(profile)
    expenses = monthly_expenses(profile)
    net = income - expenses
    rate = (net / income) if income > 0 else 0.0
    return {
        "monthly_income": income,
        "monthly_expenses": expenses,
        "net_monthly_savings": net,
        "savings_rate": rate,
    }


def net_worth(profile: dict) -> dict:
    assets = total_assets(profile)
    debt = total_debt(profile)
    return {
        "total_assets": assets,
        "total_liabilities": debt,
        "net_worth": assets - debt,
    }


def emergency_fund_status(profile: dict) -> dict:
    """Months of expenses currently covered by liquid assets, and the gap to 6."""
    expenses = monthly_expenses(profile)
    liquid = _liquid_assets(profile)
    months_covered = (liquid / expenses) if expenses > 0 else 0.0
    target_amount = EMERGENCY_FUND_TARGET_MONTHS * expenses
    gap = max(0.0, target_amount - liquid)
    return {
        "monthly_expenses": expenses,
        "liquid_assets": liquid,
        "months_covered": months_covered,
        "target_months": EMERGENCY_FUND_TARGET_MONTHS,
        "target_amount": target_amount,
        "shortfall": gap,
        "is_funded": months_covered >= EMERGENCY_FUND_TARGET_MONTHS,
    }


def debt_analysis(profile: dict) -> dict:
    """Flag high-interest debt and give avalanche (highest-rate-first) payoff order.

    Gracefully returns an empty, non-error result for Priya (no debts).
    """
    debts = profile["debts"]
    if not debts:
        return {
            "has_debt": False,
            "total_debt": 0.0,
            "high_interest_debts": [],
            "avalanche_order": [],
            "note": "No outstanding debts on record.",
        }

    # Avalanche = pay highest interest rate first (mathematically optimal).
    ranked = sorted(
        debts, key=lambda d: (d["rate_pct"] or 0.0), reverse=True
    )
    high = [d for d in ranked if (d["rate_pct"] or 0.0) >= HIGH_INTEREST_THRESHOLD_PCT]
    return {
        "has_debt": True,
        "total_debt": total_debt(profile),
        "high_interest_threshold_pct": HIGH_INTEREST_THRESHOLD_PCT,
        "high_interest_debts": [
            {"type": d["type"], "balance": d["balance"], "rate_pct": d["rate_pct"]}
            for d in high
        ],
        "avalanche_order": [
            {"type": d["type"], "balance": d["balance"], "rate_pct": d["rate_pct"],
             "rate_is_assumed": d.get("rate_is_assumed", False)}
            for d in ranked
        ],
    }


# --- Retirement (uses profile's own pre-retirement ROI) ---------------------

def retirement_plan(profile: dict) -> dict:
    """Corpus required at retirement and the monthly SIP to close the gap.

    Mirrors the Retirement sheet:
      inflated monthly expense = current * (1+infl)^years_to_retire
      corpus = annual_expense * (1 - (1+real)^-ret_years) / real
        where real = (1+roi_post)/(1+infl) - 1
      future value of what's already invested grows at roi_pre
      monthly SIP fills the remaining gap, accumulating at roi_pre
    """
    ri = profile["retirement_inputs"]
    years_to_retire = max(0, ri["retirement_age"] - ri["current_age"])
    ret_years = max(0, ri["life_expectancy"] - ri["retirement_age"])
    infl = ri["inflation"]

    inflated_monthly = ri["monthly_expense"] * (1 + infl) ** years_to_retire
    annual_expense = inflated_monthly * 12
    real_return = (1 + ri["roi_post"]) / (1 + infl) - 1

    if real_return == 0:
        corpus_required = annual_expense * ret_years
    else:
        corpus_required = annual_expense * (1 - (1 + real_return) ** -ret_years) / real_return

    fv_current = ri["already_invested"] * (1 + ri["roi_pre"]) ** years_to_retire
    net_needed = max(0.0, corpus_required - fv_current)
    monthly_sip = _monthly_sip(net_needed, ri["roi_pre"], years_to_retire * 12)

    return {
        "years_to_retirement": years_to_retire,
        "inflated_monthly_expense": inflated_monthly,
        "corpus_required": corpus_required,
        "future_value_of_current_investments": fv_current,
        "net_corpus_needed": net_needed,
        "monthly_sip_needed": monthly_sip,
    }


# --- Goals (fixed 11% assumed return) ---------------------------------------

def _goal_future_value(goal: dict) -> float:
    return goal["cost"] * (1 + goal["inflation"]) ** goal["years"]


def goal_plan(profile: dict, goal_name: str) -> dict:
    """Future value + monthly SIP for one goal, matched by (partial) name.

    Returns an error dict (not an exception) if the goal isn't found, listing the
    available goals so the agent can recover the conversation cleanly.
    """
    goals = profile["goals"]
    needle = (goal_name or "").strip().lower()
    match = None
    for g in goals:
        if needle and needle in g["name"].lower():
            match = g
            break
    if match is None:
        return {
            "status": "not_found",
            "requested": goal_name,
            "available_goals": [g["name"] for g in goals],
        }
    fv = _goal_future_value(match)
    months = match["years"] * 12
    return {
        "status": "ok",
        "goal": match["name"],
        "current_cost": match["cost"],
        "years": match["years"],
        "inflation": match["inflation"],
        "future_value": fv,
        "assumed_return": GOAL_RETURN,
        "monthly_sip_needed": _monthly_sip(fv, GOAL_RETURN, months),
    }


def all_goals_plan(profile: dict) -> dict:
    """Every goal's FV + SIP, plus the combined monthly SIP for all goals."""
    items = []
    total_sip = 0.0
    total_fv = 0.0
    for g in profile["goals"]:
        fv = _goal_future_value(g)
        sip = _monthly_sip(fv, GOAL_RETURN, g["years"] * 12)
        total_sip += sip
        total_fv += fv
        items.append(
            {"goal": g["name"], "future_value": fv, "years": g["years"],
             "monthly_sip_needed": sip}
        )
    return {"goals": items, "total_future_value": total_fv, "total_monthly_sip": total_sip}


# --- Insurance (HLV for life; rule-of-thumb for health) ---------------------

def insurance_gap(profile: dict) -> dict:
    """Life cover gap via Human Life Value; plus a health-cover adequacy check.

    HLV additional cover = income_replacement_corpus + outstanding_loans
                           + total_goal_corpus - current_investments
                           - existing_life_cover   (floored at 0)
    where income_replacement_corpus = annual_income * (1-(1+real)^-years)/real,
    real = (1+roi)/(1+infl) - 1. "Current investments" = liquid+financial assets
    (all assets except self-occupied property, which you can't spend to live).
    """
    hi = profile["hlv_inputs"]
    infl = hi["inflation"]
    real = (1 + hi["roi"]) / (1 + infl) - 1
    years = hi["years_to_provide"]

    if real == 0:
        income_replacement = hi["annual_income_to_replace"] * years
    else:
        income_replacement = (
            hi["annual_income_to_replace"] * (1 - (1 + real) ** -years) / real
        )

    loans = total_debt(profile)
    goal_corpus = all_goals_plan(profile)["total_future_value"]

    # Investable assets = everything except self-occupied property.
    property_value = sum(
        v for k, v in profile["assets"].items() if "propert" in k
    )
    current_investments = total_assets(profile) - property_value

    existing_life = profile["insurance"]["life_cover"]
    total_required = income_replacement + loans + goal_corpus - current_investments
    additional_life_cover = max(0.0, total_required - existing_life)

    # Health: simple adequacy rule since the template has no health calculator.
    # Base 5L for the individual + 5L per dependent.
    dependents = profile["profile"].get("dependents", 0)
    recommended_health = 500000 * (1 + dependents)
    health_cover = profile["insurance"]["health_cover"]

    return {
        "life": {
            "income_replacement_corpus": income_replacement,
            "outstanding_loans": loans,
            "goal_corpus": goal_corpus,
            "current_investments": current_investments,
            "existing_cover": existing_life,
            "total_cover_required": total_required,
            "additional_cover_needed": additional_life_cover,
            "has_cover": existing_life > 0,
        },
        "health": {
            "existing_cover": health_cover,
            "recommended_min": recommended_health,
            "shortfall": max(0.0, recommended_health - health_cover),
            "has_cover": health_cover > 0,
        },
    }


# --- Derived block ----------------------------------------------------------

def compute_derived(profile: dict) -> dict:
    """Populate profile['derived'] with a compact snapshot for the agent/UI.

    Mutates and returns the profile so callers can chain. This is what the
    Phase-4b debug panel and the agent's context builder read.
    """
    sr = savings_rate(profile)
    nw = net_worth(profile)
    ef = emergency_fund_status(profile)
    ret = retirement_plan(profile)
    goals = all_goals_plan(profile)
    ins = insurance_gap(profile)
    profile["derived"] = {
        "monthly_income": sr["monthly_income"],
        "monthly_expenses": sr["monthly_expenses"],
        "net_monthly_savings": sr["net_monthly_savings"],
        "savings_rate": sr["savings_rate"],
        "net_worth": nw["net_worth"],
        "emergency_fund_months": ef["months_covered"],
        "emergency_fund_shortfall": ef["shortfall"],
        "retirement_corpus": ret["corpus_required"],
        "retirement_sip": ret["monthly_sip_needed"],
        "goal_sips_total": goals["total_monthly_sip"],
        "life_cover_gap": ins["life"]["additional_cover_needed"],
        "health_cover_gap": ins["health"]["shortfall"],
    }
    return profile


# --- Aggregate: everything in one call --------------------------------------

def full_analysis(profile: dict) -> dict:
    """One coherent analysis bundling every sub-calculator's DETAILED output.

    Why this exists: the finance agent's default question ("what should I fix
    first / review my finances") needs all six areas. Calling the granular
    tools one at a time costs one model round-trip EACH (~4-5 per turn); this
    lets the agent get everything in a SINGLE tool call — ~4x fewer model calls
    on the heaviest path, faster responses, and a coherent picture rather than
    fragments to stitch. The granular tools remain for drill-down on one area.

    Returns the full sub-tool dicts (not the compact 'derived' summary), so the
    agent has the payoff order, insurance breakdown, per-goal SIPs, etc.
    """
    prof = profile["profile"]
    return {
        "profile": {
            "name": prof.get("name"),
            "age": prof.get("age"),
            "dependents": prof.get("dependents"),
            "risk_appetite": prof.get("risk_appetite"),
        },
        "cash_flow": savings_rate(profile),
        "net_worth": net_worth(profile),
        "emergency_fund": emergency_fund_status(profile),
        "debts": debt_analysis(profile),
        "insurance": insurance_gap(profile),
        "retirement": retirement_plan(profile),
        "goals": all_goals_plan(profile),
    }
