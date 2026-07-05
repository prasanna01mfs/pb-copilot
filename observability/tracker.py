"""Observability — one consistent console setup + a structured per-turn log.

Why this exists: the project already had four independent loggers
(pb.finance, pb.harness, pb.privacy, pb.research), but none had a configured
handler by default. INFO-level operational events (routing, retries,
validation outcomes, cache hits) were silently dropped unless a caller
happened to set PB_DEBUG_REDACTION=1 — an accidental, inconsistent story, and
there was no machine-readable record of what happened turn-to-turn at all
(only free-text log lines, awkward to aggregate or query after a demo run).

This module adds, all intentionally simple for a local POC:
  1. `configure_logging()` — one call, made once at each entrypoint's startup
     (API, UI), that gives every `pb.*` logger a single, consistently
     formatted console handler, independent of ADK's/mcp's own internal
     logging setup (attached to the "pb" namespace, not the root logger, so it
     can't clash with or duplicate third-party log output).
  2. `record_turn(state)` — appends one JSON line per completed turn to
     logs/observability.jsonl and emits a concise console summary.
  3. `agent_span_before/after` — logs when each agent (orchestrator,
     finance_agent, research_agent) starts and finishes, with duration.
  4. `llm_call_before/after` — logs each individual model call's latency,
     token usage, and finish_reason.
  5. `record_http_request(...)` — logs every API request (method, path,
     status, latency) to logs/api_requests.jsonl.

One rule threads through all five: METADATA ONLY, never content. No raw user
message, no model answer text, no HTTP request/response bodies. `record_turn`'s
profile tag was already redacted upstream by the harness; everything else here
never touches PII-bearing text at all, so none of this can become a new leak
path on its own.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

_OBS_LOG_PATH = Path(__file__).resolve().parent.parent / "logs" / "observability.jsonl"
_API_LOG_PATH = Path(__file__).resolve().parent.parent / "logs" / "api_requests.jsonl"

obs_logger = logging.getLogger("pb.observability")
api_logger = logging.getLogger("pb.api")

_lock = threading.Lock()
_turn_counter = 0


def configure_logging(verbose: bool = False) -> None:
    """Attach one formatted console handler to the "pb" logger namespace.

    All of pb.finance / pb.harness / pb.privacy / pb.research / pb.observability
    / pb.api are children of "pb", so one handler here covers them all via
    propagation — no need to touch the root logger (which ADK/mcp may
    configure themselves; touching root risks clashing with or duplicating
    their output).

    `verbose=False` (default): pb.observability's per-turn/per-agent/per-call
    summaries and pb.api's per-request lines always show (this IS the
    observability feature, not debug noise), plus WARNING+ from the others
    (so a retry or validation failure still surfaces, just not the full
    detail). `verbose=True` (what PB_DEBUG_REDACTION=1 enables) additionally
    shows the detailed INFO-level traces: the redacted outgoing payload, the
    harness's own diagnostic lines, cache hits/misses.
    """
    pb_logger = logging.getLogger("pb")
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s | %(message)s", "%H:%M:%S"))
    pb_logger.handlers = [handler]
    pb_logger.propagate = False
    pb_logger.setLevel(logging.INFO)

    obs_logger.setLevel(logging.INFO)
    api_logger.setLevel(logging.INFO)
    logging.getLogger("pb.ui").setLevel(logging.INFO)  # coarse, high-value UI actions — always on
    detail_level = logging.INFO if verbose else logging.WARNING
    for name in ("pb.finance", "pb.harness", "pb.privacy", "pb.research"):
        logging.getLogger(name).setLevel(detail_level)


def record_turn(state) -> None:
    """Append a structured record for one completed harness turn.

    Takes a `harness.agent_runner.HarnessState`. Never raises — a broken
    observability write must not break the turn it's describing.
    """
    global _turn_counter
    with _lock:
        _turn_counter += 1
        turn_id = _turn_counter

    validation_ok = None if state.validation is None else state.validation.ok
    validation_issues = [] if state.validation is None else state.validation.issues

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "turn_id": turn_id,
        # Already redacted by the harness before this point (see
        # memory/profile_store.redacted_version) — safe to write/share as-is.
        "profile": state.profile_version,
        "agents_run": state.agents_run,
        "tools_called": state.tools_called,
        "attempts": state.attempts,
        "timed_out": state.timed_out,
        "capped": state.capped,
        "validation_ok": validation_ok,
        "validation_issues": validation_issues,
        "repaired": state.repaired,
        "fell_back": state.fell_back,
        "self_check": state.self_check,
        "elapsed_s": round(state.elapsed_s, 3),
    }

    try:
        _OBS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_OBS_LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    except OSError:
        pass  # never let observability logging break a turn

    obs_logger.info(
        "turn=%s profile=%s agents=%s attempts=%s valid=%s fell_back=%s %.1fs",
        turn_id, state.profile_version, state.agents_run or "-", state.attempts,
        validation_ok, state.fell_back, state.elapsed_s,
    )


# --- Agent-level spans (start/end/duration per agent invocation) -----------
# Keyed by (invocation_id, agent_name) so nested/sequential agent invocations
# within one turn don't clobber each other's start time. Process-local by
# design — same precedent as privacy.redactor's module-global session state;
# fine for this local, single-user app with no concurrent overlapping turns
# (api/server.py's _turn_lock guarantees that process-wide).
_agent_span_starts: dict[tuple[str, str], float] = {}


def _span_key(callback_context) -> tuple[str, str]:
    return (
        getattr(callback_context, "invocation_id", "?"),
        getattr(callback_context, "agent_name", "?"),
    )


async def agent_span_before(callback_context) -> None:
    """Record when an agent starts.

    Must be listed FIRST in before_agent_callback (ahead of anything that
    might short-circuit, e.g. ResearchAgent's cache check) — ADK runs
    before_agent_callback list entries in order and stops at the first one
    that returns content, so putting this first guarantees the start time is
    captured even when the agent body itself never runs.
    """
    _agent_span_starts[_span_key(callback_context)] = time.monotonic()
    return None


async def agent_span_after(callback_context) -> None:
    """Record when an agent finishes normally and log its span duration.

    NOTE: ADK skips after_agent_callback entirely when a before_agent_callback
    already short-circuited the run (verified in ADK source: `run_async`
    returns early on `ctx.end_invocation` before ever reaching the after-agent
    handler). ResearchAgent's cache-hit path logs its own span directly via
    `log_agent_span_from_cache` instead, since this callback won't fire there.
    """
    key = _span_key(callback_context)
    start = _agent_span_starts.pop(key, None)
    duration = (time.monotonic() - start) if start is not None else -1.0
    obs_logger.info("agent=%s span=%.2fs source=live", key[1], duration)
    return None


def log_agent_span_from_cache(callback_context) -> None:
    """Log a (near-instant) agent span for a cache-hit short-circuit.

    Call this directly from a before_agent_callback that returns Content to
    short-circuit the run — agent_span_after will never fire on that path.
    """
    key = _span_key(callback_context)
    start = _agent_span_starts.pop(key, None)
    duration = (time.monotonic() - start) if start is not None else 0.0
    obs_logger.info("agent=%s span=%.2fs source=cache", key[1], duration)


# --- Per-LLM-call metadata (latency, tokens, finish_reason — NEVER text) ----
# Deliberately excludes response text: logging model-generated content that
# hasn't yet passed the harness's validation layer would risk surfacing
# hallucinated or PII-adjacent text in a log outside the redact/rehydrate
# contract. Metadata only, same rule as record_turn.
#
# Single shared variable, not keyed per-call: model calls within this app are
# always sequential (one asyncio task per turn, AgentTool calls are awaited,
# never fired concurrently), so before/after always pair up correctly — same
# reasoning as the agent-span keying above, just without needing the key.
_pending_llm_call_start: float | None = None


async def llm_call_before(callback_context, llm_request) -> None:
    """Mark the start of one model call. List this ahead of redaction_before_model
    in before_model_callback so the timer includes the full round trip."""
    global _pending_llm_call_start
    _pending_llm_call_start = time.monotonic()
    return None


async def llm_call_after(callback_context, llm_response) -> None:
    """Log one model call's latency, token usage, and finish_reason."""
    global _pending_llm_call_start
    start, _pending_llm_call_start = _pending_llm_call_start, None
    latency = (time.monotonic() - start) if start is not None else -1.0

    agent = getattr(callback_context, "agent_name", "?")
    usage = getattr(llm_response, "usage_metadata", None)
    prompt_tokens = getattr(usage, "prompt_token_count", None) if usage else None
    response_tokens = getattr(usage, "candidates_token_count", None) if usage else None
    total_tokens = getattr(usage, "total_token_count", None) if usage else None
    finish_reason = getattr(llm_response, "finish_reason", None)
    finish_reason = getattr(finish_reason, "name", finish_reason)  # enum -> its name

    obs_logger.info(
        "llm_call agent=%s latency=%.2fs prompt_tokens=%s response_tokens=%s "
        "total_tokens=%s finish_reason=%s",
        agent, latency, prompt_tokens, response_tokens, total_tokens, finish_reason,
    )
    return None


# --- API request/response logging (transport metadata only, NEVER bodies) --
# Request/response BODIES are deliberately excluded: a /chat request body can
# carry a user-typed message (possibly PII, before it's redacted downstream),
# and a /chat response body carries the REHYDRATED real answer — logging
# either would open a new PII path outside the redact/rehydrate contract.
def record_http_request(*, method: str, path: str, status_code: int, elapsed_s: float,
                         client_host: str | None = None) -> None:
    """Append one structured record per HTTP request/response pair."""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "method": method,
        "path": path,
        "status_code": status_code,
        "elapsed_s": round(elapsed_s, 3),
        "client_host": client_host,
    }
    try:
        _API_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_API_LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    except OSError:
        pass  # never let observability logging break a request

    api_logger.info("%s %s -> %s (%.3fs)", method, path, status_code, elapsed_s)
