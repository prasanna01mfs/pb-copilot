"""Agent harness — the reliability wrapper EVERY turn runs through (Section 2c).

Why a harness at all: an LLM agent on its own is "prompt + tools and hope." The
six layers below turn that into an engineered system that fails safe, never
shows a stack trace to the user, and never lets a hallucinated number or a
missing disclaimer reach the screen. All of it lives HERE, once, so both agents
(and the orchestrator that routes to them) inherit the same guarantees instead
of each re-implementing ad-hoc handling.

The layers, in execution order (Section 2c):
  1. Context + PII redaction — start a redaction session so the before-model
     hook tokenizes PII out of every model call this turn.
  2. Call with timeout — never hang the UI/demo on a slow model call.
  3. Control flow — a client-side rate limiter paces calls UNDER the free-tier
     per-minute cap so we avoid 429s in the first place (harness/throttle.py);
     when one still slips through, retry-with-backoff waits LONGER for a 429
     (honoring the server's retryDelay) than for a timeout, with jitter; plus a
     hard cap on tool-loop iterations so the agent can't loop forever.
  4. Output validation — numbers must match what the tools can actually compute,
     a disclaimer must be present, no "guaranteed returns" language, research
     must cite a source. On failure: repair once, else a safe fallback.
  5. Self-check — one optional light pass asking the model whether its advice is
     consistent with the numbers + priority order (quota-gated; off by default).
  6. Log/state — record which agent(s) and tools ran, the profile version, retry
     count and validation outcome, for the debug panel AND a structured
     per-turn observability record (see observability/tracker.py).

Design note: the harness is agent-agnostic — it wraps a `Runner` execution, so
it works for the orchestrator (both specialists then run inside it) or either
specialist alone. The only ADK surface it touches is `runner.run_async(...)`,
which makes the reliability layers testable with a fake runner and ZERO API
quota (see tests / the STOP instructions).
"""
from __future__ import annotations

import asyncio
import logging
import os
import random
import re
import time
from dataclasses import dataclass, field

from google.adk.agents.invocation_context import LlmCallsLimitExceededError
from google.adk.agents.run_config import RunConfig
from google.genai import types
from google.genai.errors import ClientError, ServerError

from memory import profile_store
from observability import tracker as observability
from privacy import redactor
from tools import finance_tools

logger = logging.getLogger("pb.harness")

# --- Tunables (env-overridable so tests can force timeouts/retries cheaply) ---
# gemini-2.5-pro reasons more (and this turn's tool-calling loop can run several
# rounds), so it routinely needs well over a flash model's ~10-20s — 60s was
# tuned for flash and cut Pro off mid-response every time in practice. 150s
# gives a full-analysis turn room to actually finish on the first attempt.
TIMEOUT_S = float(os.getenv("PB_HARNESS_TIMEOUT", "150"))        # layer 2
MAX_RETRIES = int(os.getenv("PB_HARNESS_RETRIES", "2"))          # layer 3 (retries AFTER the first try)
BACKOFF_BASE_S = float(os.getenv("PB_HARNESS_BACKOFF", "2"))     # layer 3 (timeouts/5xx: 2s, 4s, ...)
# A 429 means a per-MINUTE window is full — short 2s/4s waits won't clear it, so
# rate-limit retries back off much longer (and honor the server's own suggested
# retry delay when present). Capped so a retry storm can't hang a turn forever.
RATE_LIMIT_BACKOFF_S = float(os.getenv("PB_HARNESS_RL_BACKOFF", "8"))       # 8s, 16s, ...
RATE_LIMIT_BACKOFF_MAX_S = float(os.getenv("PB_HARNESS_RL_BACKOFF_MAX", "45"))
MAX_LLM_CALLS = int(os.getenv("PB_HARNESS_MAX_LLM_CALLS", "12")) # layer 3 (tool-loop cap per turn)
SELF_CHECK_DEFAULT = os.getenv("PB_SELF_CHECK") == "1"           # layer 5 (costs one extra model call)

# layer 3 — MODEL fallback. A 503 "model overloaded / high demand" error means
# THAT MODEL's serving capacity is the problem, not our request pattern (unlike
# a 429, which is OUR quota) — retrying the same model with backoff just wastes
# the turn's time budget. After this many CONSECUTIVE overload errors, swap the
# affected agent(s) to a different model for the rest of this turn (and grant
# one extra attempt so the swap actually gets tried, not just used up giving
# up). Set PB_MODEL_FALLBACK_AFTER=0 to disable. Fallbacks are pinned (non
# "-latest") versions on purpose: an alias can point at the same overloaded
# release the primary model resolved to. This is real financial analysis, so
# the fallback default is still flagship Flash — NOT a lite model — a 503
# should cost some depth/speed, never accuracy.
MODEL_FALLBACK_AFTER = int(os.getenv("PB_MODEL_FALLBACK_AFTER", "2"))
GEMINI_MODEL_FALLBACK = os.getenv("GEMINI_MODEL_FALLBACK", "gemini-2.5-flash")
RESEARCH_MODEL_FALLBACK = os.getenv("RESEARCH_MODEL_FALLBACK", "gemini-2.5-flash")

# Numbers below this are months/percentages/ages/years — not monetary figures,
# so we don't try to validate them against tool outputs.
_FIGURE_MIN = 1000
# Allowed relative error between an answer figure and a computed figure. Covers
# the model rounding 76,705.40 -> 76,705 without flagging it as invented.
_FIGURE_TOL = 0.015

SAFE_FALLBACK = (
    "I hit a problem completing that safely, so I'm not going to guess. "
    "Please try again in a moment. "
    "Educational guidance only, not licensed financial advice."
)

# Guardrail patterns (layer 4).
_GUARANTEE_RE = re.compile(
    r"guarantee(?:d|s)?\s+(?:returns?|profits?|gains?)|assured\s+returns?|"
    r"risk[-\s]?free\s+returns?|promis\w*\s+returns?",
    re.IGNORECASE,
)
# Both agents are instructed to append the disclaimer VERBATIM, but live models
# reliably paraphrase it slightly (e.g. "does not constitute licensed financial
# advice", "for educational purposes") even when told "exactly". Matching only
# the literal string caused real, compliant answers to fail validation and hit
# the safe fallback (caught via live testing, not the unit tests, which only
# ever fed it the literal string). This matches the SUBSTANCE — an educational/
# non-advice framing plus a reference to licensed financial advice — rather
# than one exact wording.
_DISCLAIMER_RE = re.compile(
    r"educational\s+(?:guidance|purposes?|only)|"
    r"not\s+(?:a\s+)?(?:substitute\s+for\s+)?licensed\s+financial\s+advice|"
    r"does\s+not\s+constitute\s+(?:licensed\s+)?financial\s+advice|"
    r"not\s+(?:intended\s+as|financial)\s+advice",
    re.IGNORECASE,
)
_SOURCE_RE = re.compile(r"https?://|\bsources?\b\s*:|\[\d+\]|according to", re.IGNORECASE)
_NO_SOURCE_RE = re.compile(r"no\s+(?:sources?|results?)\s+(?:found|available)", re.IGNORECASE)
_INVESTMENT_RE = re.compile(
    r"\b(invest|sip|mutual fund|stock|equity|fund|portfolio|returns?)\b", re.IGNORECASE
)
# A MONETARY figure in the answer. Two ways to qualify (so we don't mistake a
# bare year like 2013/2026 — or any non-money integer a research answer cites —
# for a financial figure to validate):
#   (1) a ₹ / Rs. currency prefix, or
#   (2) thousands-separator commas (Indian "2,15,000" or western "215,000").
# Bare integers with neither are ignored. This is safe because every large
# figure the finance agent states is written with ₹ and/or commas.
_NUM_RE = re.compile(
    r"(?:₹|Rs\.?)\s*(\d[\d,]*(?:\.\d+)?)"      # group 1: currency-prefixed
    r"|\b(\d{1,3}(?:,\d{2,3})+(?:\.\d+)?)\b"    # group 2: comma-grouped
)
# The profile is always in Indian Rupees, so a $-prefixed figure in a finance
# answer is wrong regardless of its numeric value (it's either a mislabeled ₹
# figure or a stray USD one) — catch it even though _NUM_RE's comma-grouped
# alternative would otherwise let it slip through as "grounded".
_USD_RE = re.compile(r"\$\s*\d[\d,]*(?:\.\d+)?")


# --- Structured results -----------------------------------------------------

@dataclass
class ValidationResult:
    ok: bool
    issues: list[str] = field(default_factory=list)


@dataclass
class HarnessState:
    agents_run: list[str] = field(default_factory=list)
    tools_called: list[str] = field(default_factory=list)
    profile_version: str = "none#0"
    attempts: int = 0
    timed_out: bool = False
    capped: bool = False
    validation: ValidationResult | None = None
    repaired: bool = False
    self_check: dict | None = None
    fell_back: bool = False
    model_fallback: bool = False
    elapsed_s: float = 0.0

    def summary(self) -> str:
        v = self.validation
        return (
            f"agents={self.agents_run or '-'} tools={self.tools_called or '-'} "
            f"profile={self.profile_version} attempts={self.attempts} "
            f"timed_out={self.timed_out} capped={self.capped} "
            f"valid={None if v is None else v.ok} repaired={self.repaired} "
            f"model_fallback={self.model_fallback} "
            f"fell_back={self.fell_back} {self.elapsed_s:.1f}s"
        )


@dataclass
class HarnessResult:
    text: str
    state: HarnessState


@dataclass
class _RunOutput:
    """Raw material collected from one agent run."""
    final_text: str | None
    agents_run: list[str]
    tools_called: list[str]


# --- Layer 4 helpers (pure, unit-testable without any model) ----------------

def collect_legitimate_figures(profile: dict) -> set[float]:
    """Every monetary figure the finance tools can legitimately produce/see.

    The harness recomputes these from the active profile rather than scraping
    the event stream, because with the AgentTool architecture the specialist's
    inner tool outputs don't bubble up to the top-level run. Recomputing is also
    stricter: any large rupee figure in the answer that ISN'T here is, by
    definition, not grounded in the tools — i.e. a hallucination.
    """
    figs: set[float] = set()

    def walk(o):
        if isinstance(o, bool):
            return
        if isinstance(o, (int, float)):
            figs.add(round(float(o), 2))
        elif isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v)

    for fn in (
        finance_tools.savings_rate,
        finance_tools.net_worth,
        finance_tools.emergency_fund_status,
        finance_tools.debt_analysis,
        finance_tools.insurance_gap,
        finance_tools.retirement_plan,
        finance_tools.all_goals_plan,
    ):
        try:
            walk(fn(profile))
        except Exception:  # noqa: BLE001 — a broken calc must not break validation
            continue
    for section in ("derived", "income_monthly", "expenses_monthly", "assets",
                    "debts", "insurance"):
        walk(profile.get(section))
    return figs


def _answer_figures(text: str) -> list[float]:
    """Large monetary numbers mentioned in an answer (>= _FIGURE_MIN).

    _NUM_RE has two alternatives (currency-prefixed | comma-grouped), so each
    match yields a 2-tuple with one group populated — take whichever matched.
    """
    out = []
    for g1, g2 in _NUM_RE.findall(text or ""):
        raw = g1 or g2
        try:
            val = float(raw.replace(",", ""))
        except ValueError:
            continue
        if val >= _FIGURE_MIN:
            out.append(val)
    return out


def _figure_is_grounded(n: float, legit: set[float]) -> bool:
    return any(abs(n - f) <= max(1.0, _FIGURE_TOL * f) for f in legit)


def validate_finance_text(text: str, legit_figures: set[float],
                          strict_figures: bool = True) -> ValidationResult:
    """Validate a finance answer against the tool-computable figures + guardrails.

    `strict_figures` enforces that every large monetary figure matches a finance
    tool output — the anti-hallucination guard for a FINANCE-ONLY turn. It's
    turned OFF for a cross-agent (research + finance) MERGE, because that answer
    legitimately synthesizes external research with the profile and derives new
    amounts (a suggested SIP in the fund, a combined monthly total) that aren't
    verbatim tool outputs. The safety guards (disclaimer, no guaranteed returns,
    ₹-not-$) still apply in both cases.
    """
    issues: list[str] = []
    if not _DISCLAIMER_RE.search(text or ""):
        issues.append("missing educational disclaimer")
    if _GUARANTEE_RE.search(text or ""):
        issues.append("contains guaranteed-return language")
    if _USD_RE.search(text or ""):
        issues.append("used $ instead of ₹ for a rupee figure")
    if strict_figures:
        ungrounded = [n for n in _answer_figures(text) if not _figure_is_grounded(n, legit_figures)]
        if ungrounded:
            issues.append(f"figures not grounded in tool outputs: {sorted(set(ungrounded))}")
    return ValidationResult(ok=not issues, issues=issues)


def validate_research_text(text: str) -> ValidationResult:
    """Validate a research answer: cite a source (or say none), disclaimer if investment."""
    issues: list[str] = []
    t = text or ""
    if not (_SOURCE_RE.search(t) or _NO_SOURCE_RE.search(t)):
        issues.append("no cited source and no 'no sources found' statement")
    if _INVESTMENT_RE.search(t) and not _DISCLAIMER_RE.search(t):
        issues.append("investment topic without educational disclaimer")
    return ValidationResult(ok=not issues, issues=issues)


def _validate(text: str, agents_run: list[str], legit_figures: set[float]) -> ValidationResult:
    """Pick the validation profile from which specialist(s) actually ran."""
    issues: list[str] = []
    ran_finance = "finance_agent" in agents_run
    ran_research = "research_agent" in agents_run
    if ran_finance:
        # Strict figure-grounding only for a finance-ONLY turn; a cross-agent
        # merge (research + finance) synthesizes derived amounts that aren't
        # verbatim tool outputs, so relax it there (guardrails still enforced).
        issues += validate_finance_text(
            text, legit_figures, strict_figures=not ran_research).issues
    if ran_research:
        issues += validate_research_text(text).issues
    # If neither specialist ran (e.g. an off-topic decline), there's nothing to
    # ground — don't manufacture a failure.
    return ValidationResult(ok=not issues, issues=issues)


# --- Layer 3 helper ---------------------------------------------------------

def _is_transient(exc: BaseException) -> bool:
    """Is this worth retrying? Timeouts and 429/5xx are; a bad request isn't."""
    if isinstance(exc, asyncio.TimeoutError):
        return True
    if isinstance(exc, ClientError) and getattr(exc, "code", None) in (429, 500, 503):
        return True
    s = str(exc).upper()
    return any(tok in s for tok in
               ("RESOURCE_EXHAUSTED", "429", "DEADLINE", "TIMEOUT", "UNAVAILABLE", "503"))


def _is_rate_limit(exc: BaseException) -> bool:
    """Specifically a 429 / quota-exhausted error (needs a much longer backoff)."""
    if isinstance(exc, ClientError) and getattr(exc, "code", None) == 429:
        return True
    s = str(exc).upper()
    return "429" in s or "RESOURCE_EXHAUSTED" in s


def _is_server_overload(exc: BaseException) -> bool:
    """Specifically a 503 'model overloaded / high demand' error — as opposed to
    a 429 (OUR quota) or a bare timeout. Distinguished from _is_transient because
    the response here is different: retrying the SAME model just re-queues
    behind the same overload, so this triggers a MODEL swap (see
    MODEL_FALLBACK_AFTER) rather than another backoff-and-retry of the model
    that's already struggling."""
    if isinstance(exc, ServerError) and getattr(exc, "code", None) == 503:
        return True
    s = str(exc).upper()
    return "UNAVAILABLE" in s or "HIGH DEMAND" in s


def _iter_llm_agents(runner):
    """Yield every LlmAgent this turn might call: the root agent plus each
    specialist wrapped in an AgentTool. Walking `runner.agent.tools` rather than
    importing finance_agent/research_agent directly keeps the harness
    agent-agnostic (same reason it takes a bare `runner`, per the module
    docstring) — it works for the orchestrator or a lone specialist alike."""
    root = getattr(runner, "agent", None)
    if root is None:
        return
    yield root
    for tool in getattr(root, "tools", None) or []:
        inner = getattr(tool, "agent", None)
        if inner is not None:
            yield inner


def _fallback_model_for(agent) -> str:
    """Which fallback model an agent should switch to on repeated overload.

    research_agent needs a grounding-capable model (see agents/research_agent.py);
    everything else (orchestrator, finance_agent) shares GEMINI_MODEL, so they
    share its fallback too.
    """
    return RESEARCH_MODEL_FALLBACK if getattr(agent, "name", "") == "research_agent" \
        else GEMINI_MODEL_FALLBACK


def _retry_delay(exc: BaseException, attempt: int) -> float:
    """How long to wait before the next retry — with jitter, and 429-aware.

    - 429: honor the server's suggested retryDelay if present (it knows when the
      per-minute window clears), otherwise exponential from a longer base;
      capped so a retry can't hang a turn indefinitely.
    - everything else (timeout/5xx): the ordinary short exponential backoff.
    Jitter (±20%) avoids synchronized ret/storms hitting the same reset instant.
    """
    if _is_rate_limit(exc):
        m = re.search(r"retry in ([0-9.]+)s", str(exc)) or \
            re.search(r"retryDelay['\":\s]+([0-9.]+)s", str(exc))
        server_delay = float(m.group(1)) if m else 0.0
        base = max(server_delay, RATE_LIMIT_BACKOFF_S * (2 ** (attempt - 1)))
        base = min(base, RATE_LIMIT_BACKOFF_MAX_S)
    else:
        base = BACKOFF_BASE_S * (2 ** (attempt - 1))
    return max(0.1, base * (1 + random.uniform(-0.2, 0.2)))


# --- The run itself ---------------------------------------------------------

async def _drive_once(runner, *, user_id, session_id, message) -> _RunOutput:
    """One agent run: stream events, collect final text + which agents/tools fired."""
    content = types.Content(role="user", parts=[types.Part.from_text(text=message)])
    agents_run: list[str] = []
    tools_called: list[str] = []
    final_text: str | None = None
    run_config = RunConfig(max_llm_calls=MAX_LLM_CALLS)  # layer 3: hard cap on the tool loop

    async for event in runner.run_async(
        user_id=user_id, session_id=session_id, new_message=content, run_config=run_config
    ):
        for part in (event.content.parts if event.content else []) or []:
            fc = getattr(part, "function_call", None)
            if fc is not None:
                # AgentTool calls are named after the specialist; everything else
                # is a plain tool. This is our visibility into routing (layer 6).
                if fc.name in ("finance_agent", "research_agent"):
                    agents_run.append(fc.name)
                else:
                    tools_called.append(fc.name)
        if event.is_final_response() and event.content and event.content.parts:
            final_text = event.content.parts[0].text
    return _RunOutput(final_text, agents_run, tools_called)


async def run_through_harness(
    *,
    runner,
    user_id: str,
    session_id: str,
    message: str,
    known_names: list[str] | None = None,
    self_check: bool | None = None,
) -> HarnessResult:
    """Run one turn through all six harness layers. Returns rehydrated text + state."""
    started = time.monotonic()

    # ---- Layer 1: context + PII redaction ----
    # Starting the redaction session here (once, in the harness) is what makes
    # "redact before the model, rehydrate only at the end" a single guarantee
    # rather than something each caller must remember to do. This must run
    # BEFORE state.profile_version is computed: redacted_version() reuses this
    # session so the name-token in the debug trace matches whatever token was
    # used elsewhere in this turn, instead of leaking the raw name (see
    # memory/profile_store.redacted_version — the log/API leak this closes).
    session = redactor.new_session(known_names=known_names or [])
    state = HarnessState(profile_version=profile_store.redacted_version())
    legit_figures = (
        collect_legitimate_figures(profile_store.get_active())
        if profile_store.has_active()
        else set()
    )

    # ---- Layers 2 + 3: call with timeout, retry-with-backoff, capped loop ----
    run_out: _RunOutput | None = None
    max_attempts = MAX_RETRIES + 2  # 1 initial try + MAX_RETRIES retries
    overload_streak = 0
    fallback_applied = False
    original_models: dict[int, str] = {}
    attempt = 0
    try:
        while attempt < max_attempts:
            attempt += 1
            state.attempts = attempt
            try:
                run_out = await asyncio.wait_for(
                    _drive_once(runner, user_id=user_id, session_id=session_id, message=message),
                    timeout=TIMEOUT_S,
                )
                break
            except LlmCallsLimitExceededError:
                # Runaway tool loop hit the cap — do not retry a runaway; fail safe.
                state.capped = True
                logger.warning("tool-loop cap (%s) exceeded; returning safe fallback", MAX_LLM_CALLS)
                break
            except Exception as exc:  # noqa: BLE001 — classify, then retry or give up
                if isinstance(exc, asyncio.TimeoutError):
                    state.timed_out = True
                transient = _is_transient(exc)
                overload_streak = overload_streak + 1 if _is_server_overload(exc) else 0
                # The model itself (not our request pattern) is overloaded —
                # backing off and re-asking the SAME model just re-queues behind
                # the same congestion. Switch to a different model instead, and
                # grant one extra attempt so the swap actually gets exercised
                # rather than immediately being consumed by "last attempt, give
                # up" below.
                if (MODEL_FALLBACK_AFTER > 0 and not fallback_applied
                        and overload_streak >= MODEL_FALLBACK_AFTER):
                    fallback_applied = True
                    state.model_fallback = True
                    max_attempts += 1
                    for agent in _iter_llm_agents(runner):
                        original_models.setdefault(id(agent), agent.model)
                        target = _fallback_model_for(agent)
                        if agent.model != target:
                            logger.warning(
                                "model overload (%d consecutive 503s) — switching %s: %s -> %s",
                                overload_streak, getattr(agent, "name", "?"), agent.model, target,
                            )
                            agent.model = target
                last_attempt = attempt >= max_attempts
                if not transient or last_attempt:
                    logger.warning("attempt %s failed (%s, transient=%s) — giving up",
                                   attempt, type(exc).__name__, transient)
                    break
                # 429 -> long backoff (per-minute window must clear); else short.
                # Jittered to avoid synchronized retries. See _retry_delay.
                delay = _retry_delay(exc, attempt)
                logger.warning("attempt %s failed (%s, rate_limit=%s) — backing off %.1fs",
                               attempt, type(exc).__name__, _is_rate_limit(exc), delay)
                await asyncio.sleep(delay)
    finally:
        # Always restore the primary model(s), win or lose — a fallback used for
        # ONE degraded turn must not silently stick for every turn after it; the
        # next turn should try the primary model fresh in case Google's overload
        # has cleared by then.
        for agent in _iter_llm_agents(runner):
            orig = original_models.get(id(agent))
            if orig is not None:
                agent.model = orig

    # No usable output from any attempt -> safe fallback (never a stack trace).
    if run_out is None or not run_out.final_text:
        state.fell_back = True
        state.elapsed_s = time.monotonic() - started
        observability.record_turn(state)
        return HarnessResult(text=SAFE_FALLBACK, state=state)

    state.agents_run = run_out.agents_run
    state.tools_called = run_out.tools_called
    text = run_out.final_text

    # ---- Layer 4: validate; repair once; else safe fallback ----
    state.validation = _validate(text, state.agents_run, legit_figures)
    if not state.validation.ok:
        logger.warning("validation failed: %s", state.validation.issues)
        repaired = await _repair(runner, user_id, session_id, message, state.validation.issues)
        state.repaired = True
        if repaired and repaired.final_text:
            reval = _validate(repaired.final_text, repaired.agents_run or state.agents_run,
                              legit_figures)
            if reval.ok:
                text, state.validation = repaired.final_text, reval
                state.agents_run = repaired.agents_run or state.agents_run
            else:
                # Still bad after one repair -> don't show unvalidated advice.
                state.fell_back = True
                text = SAFE_FALLBACK
        else:
            state.fell_back = True
            text = SAFE_FALLBACK

    # ---- Layer 5: light self-check (optional; costs one model call) ----
    do_self_check = SELF_CHECK_DEFAULT if self_check is None else self_check
    if do_self_check and not state.fell_back and "finance_agent" in state.agents_run:
        state.self_check = await _self_check(text, legit_figures)

    # ---- Layer 6: log/state, then rehydrate PII for the user only ----
    state.elapsed_s = time.monotonic() - started
    observability.record_turn(state)
    return HarnessResult(text=redactor.rehydrate(text, session.mapping), state=state)


async def _repair(runner, user_id, session_id, message, issues) -> _RunOutput | None:
    """One corrective re-ask: tell the agent exactly what to fix, run again.

    A single repair (not a loop) keeps cost bounded; if it still fails we fall
    back rather than risk showing the user unvalidated financial advice.
    """
    # Issue-driven, agent-agnostic correction: the same repair path runs for a
    # research turn (needs an inline "Sources:" citation) and a finance turn
    # (needs tool-grounded figures), so don't hardcode finance-only guidance.
    correction = (
        f"{message}\n\n[system correction: your previous answer had these problems: "
        f"{'; '.join(issues)}. Fix ALL of them and answer again, keeping every "
        f"standing rule: for research, list your sources inline in a 'Sources:' "
        f"section; for finance advice, use ONLY figures your tools returned; "
        f"state every monetary figure in ₹ (Indian Rupees), never $; "
        f"include the educational disclaimer; make no guaranteed-return claims.]"
    )
    try:
        return await asyncio.wait_for(
            _drive_once(runner, user_id=user_id, session_id=session_id, message=correction),
            timeout=TIMEOUT_S,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("repair attempt failed: %s", type(exc).__name__)
        return None


async def _self_check(text: str, legit_figures: set[float]) -> dict:
    """Ask the model once whether the advice is internally consistent.

    Deliberately cheap and side-effect-free: a single call that returns a short
    verdict. Gated off by default because on the free tier every call counts;
    the number-grounding in layer 4 already catches the highest-risk error
    (invented figures) deterministically and for free.
    """
    from google import genai  # local import so the harness imports without a key

    model = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
    prompt = (
        "You are a strict reviewer. Does the following personal-finance answer stay "
        "consistent with the priority order (emergency fund -> insurance -> "
        "high-interest debt -> investing) and avoid contradicting itself? "
        "Reply with 'OK' or 'ISSUE: <one line>'.\n\n" + text
    )
    try:
        client = genai.Client()
        resp = await asyncio.wait_for(
            asyncio.to_thread(client.models.generate_content, model=model, contents=prompt),
            timeout=TIMEOUT_S,
        )
        verdict = (resp.text or "").strip()
        return {"consistent": verdict.upper().startswith("OK"), "verdict": verdict[:200]}
    except Exception as exc:  # noqa: BLE001 — self-check is best-effort, never fatal
        return {"consistent": None, "verdict": f"self-check skipped: {type(exc).__name__}"}
