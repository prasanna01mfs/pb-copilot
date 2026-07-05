"""Correctness tests for the finance calculators + the harness guardrails.

Two things are proven here:

1. The calculators reproduce EACH PLANNER'S OWN computed cells (retirement corpus
   & SIP, HLV additional cover, goal FV & SIP, net worth) — i.e. the Python math
   matches the spreadsheet's independently-computed ground truth, for BOTH Arjun
   and Priya. The expected numbers below were read directly out of the .xlsx
   formula cells, so this is a real cross-check, not a restatement of our code.

2. Priya's edge cases (empty debts, zero insurance) produce clean, correct
   results with no crashes.

3. The harness guardrails actually fire: guaranteed-return language is caught,
   a missing disclaimer is caught (including reasonable paraphrases), and
   ungrounded figures are caught.
"""
import os

import pytest

from tools.excel_parser import parse_planner
from tools import finance_tools
from harness import agent_runner as H

_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def _load(filename: str) -> dict:
    profile = parse_planner(os.path.join(_DATA, filename))
    finance_tools.compute_derived(profile)
    return profile


@pytest.fixture(scope="module")
def arjun() -> dict:
    return _load("PB_Planner_Profile1_Arjun.xlsx")


@pytest.fixture(scope="module")
def priya() -> dict:
    return _load("PB_Planner_Profile2_Priya.xlsx")


@pytest.fixture(scope="module")
def legit(arjun) -> set:
    """The set of figures the finance tools can legitimately produce for Arjun,
    used by the guardrail grounding check."""
    return H.collect_legitimate_figures(arjun)


# Relative tolerance for money: our formulas match the sheet to the rupee, but
# 0.1% absorbs any last-digit rounding without hiding a real error.
REL = 1e-3


# --- Arjun: the "fragile family" profile ------------------------------------

class TestArjun:
    def test_net_worth_and_savings(self, arjun):
        assert finance_tools.net_worth(arjun)["net_worth"] == pytest.approx(7_800_000, rel=REL)
        assert finance_tools.savings_rate(arjun)["net_monthly_savings"] == pytest.approx(98_000, rel=REL)

    def test_emergency_fund_months_and_shortfall(self, arjun):
        ef = finance_tools.emergency_fund_status(arjun)
        # liquid = savings 180k + FD 300k + debt MF 200k = 680k; /117k expenses
        assert ef["months_covered"] == pytest.approx(680_000 / 117_000, rel=REL)
        assert ef["months_covered"] == pytest.approx(5.8120, rel=1e-3)
        assert ef["shortfall"] == pytest.approx(22_000, rel=REL)
        assert ef["is_funded"] is False

    def test_debt_avalanche_order_and_high_interest(self, arjun):
        da = finance_tools.debt_analysis(arjun)
        assert da["has_debt"] is True
        # Avalanche = highest rate first: 42% card before the 8.5% home loan.
        assert [d["type"] for d in da["avalanche_order"]] == ["credit_card", "home_loan"]
        # Only the card clears the high-interest threshold.
        assert [d["type"] for d in da["high_interest_debts"]] == ["credit_card"]
        assert da["high_interest_debts"][0]["rate_pct"] == pytest.approx(42.0)

    def test_hlv_additional_cover(self, arjun):
        ins = finance_tools.insurance_gap(arjun)
        assert ins["life"]["additional_cover_needed"] == pytest.approx(73_311_809.15, rel=REL)
        assert ins["life"]["has_cover"] is True
        assert ins["health"]["has_cover"] is True

    def test_retirement_corpus_and_sip(self, arjun):
        ret = finance_tools.retirement_plan(arjun)
        assert ret["corpus_required"] == pytest.approx(77_183_715.41, rel=REL)
        assert ret["monthly_sip_needed"] == pytest.approx(76_705.40, rel=REL)

    def test_goal_sip(self, arjun):
        g = finance_tools.goal_plan(arjun, "House Down Payment")
        assert g["status"] == "ok"
        assert g["future_value"] == pytest.approx(4_908_931.06, rel=REL)
        assert g["monthly_sip_needed"] == pytest.approx(61_733.52, rel=REL)

    def test_all_goals_total_sip(self, arjun):
        assert finance_tools.all_goals_plan(arjun)["total_monthly_sip"] == pytest.approx(188_540.87, rel=REL)


# --- Priya: the "blank slate" profile (edge cases: no debt, no insurance) ----

class TestPriya:
    def test_net_worth_and_savings(self, priya):
        assert finance_tools.net_worth(priya)["net_worth"] == pytest.approx(310_000, rel=REL)
        assert finance_tools.savings_rate(priya)["net_monthly_savings"] == pytest.approx(37_000, rel=REL)

    def test_emergency_fund_months_and_shortfall(self, priya):
        ef = finance_tools.emergency_fund_status(priya)
        # liquid = savings 60k + FD 50k = 110k; /51k expenses
        assert ef["months_covered"] == pytest.approx(110_000 / 51_000, rel=REL)
        assert ef["shortfall"] == pytest.approx(196_000, rel=REL)
        assert ef["is_funded"] is False

    def test_empty_debts_no_crash(self, priya):
        da = finance_tools.debt_analysis(priya)
        assert da["has_debt"] is False
        assert da["avalanche_order"] == []
        assert da["high_interest_debts"] == []

    def test_zero_insurance_no_crash(self, priya):
        ins = finance_tools.insurance_gap(priya)
        assert ins["life"]["has_cover"] is False
        assert ins["health"]["has_cover"] is False
        # HLV still computes a positive additional-cover need even at zero cover.
        assert ins["life"]["additional_cover_needed"] == pytest.approx(13_317_409.18, rel=REL)

    def test_retirement_corpus_and_sip(self, priya):
        ret = finance_tools.retirement_plan(priya)
        assert ret["corpus_required"] == pytest.approx(63_750_294.71, rel=REL)
        assert ret["monthly_sip_needed"] == pytest.approx(13_128.60, rel=REL)

    def test_goal_sip(self, priya):
        g = finance_tools.goal_plan(priya, "First Car")
        assert g["status"] == "ok"
        assert g["future_value"] == pytest.approx(972_405.0, rel=REL)
        assert g["monthly_sip_needed"] == pytest.approx(16_218.60, rel=REL)

    def test_all_goals_total_sip(self, priya):
        assert finance_tools.all_goals_plan(priya)["total_monthly_sip"] == pytest.approx(45_432.99, rel=REL)


# --- Guardrails (harness layer 4) -------------------------------------------

class TestGuardrails:
    def test_compliant_answer_passes(self, legit):
        text = ("Clear your ₹80,000 card first. Your retirement SIP is ₹76,705. "
                "Educational guidance only, not licensed financial advice.")
        assert H.validate_finance_text(text, legit).ok is True

    def test_guaranteed_return_language_is_flagged(self, legit):
        text = ("This fund gives guaranteed returns of 15%. "
                "Educational guidance only, not licensed financial advice.")
        result = H.validate_finance_text(text, legit)
        assert result.ok is False
        assert any("guaranteed" in i for i in result.issues)

    def test_missing_disclaimer_is_flagged(self, legit):
        result = H.validate_finance_text("Clear your ₹80,000 card first.", legit)
        assert result.ok is False
        assert any("disclaimer" in i for i in result.issues)

    def test_paraphrased_disclaimer_is_accepted(self, legit):
        # Live models paraphrase the disclaimer even when told to be verbatim;
        # the guardrail must accept the substance, not one exact string.
        text = ("Clear your ₹80,000 card first. This information is for educational "
                "purposes and does not constitute licensed financial advice.")
        assert H.validate_finance_text(text, legit).ok is True

    def test_ungrounded_figure_is_flagged(self, legit):
        text = ("Your retirement SIP is ₹99,999. "
                "Educational guidance only, not licensed financial advice.")
        result = H.validate_finance_text(text, legit)
        assert result.ok is False
        assert any("not grounded" in i for i in result.issues)
