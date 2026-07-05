"""Cache-on-success for the ResearchAgent (demo reliability).

Live Google Search is impressive but risky on camera — a flaky network or a
throttled call can ruin a take. We cache a SUCCESSFUL research answer for the
session, keyed by the (normalized) query, so re-running the same query during a
recording is instant and identical.

Why cache at the ResearchAgent boundary (query -> final answer) rather than
around google_search itself: google_search is model-internal grounding, not a
Python function we can wrap. Caching the agent's answer achieves the same goal
and, importantly, only caches the volatile EXTERNAL research — the profile-
dependent finance fit and the cross-agent merge still run fresh every time, so
Arjun and Priya never get a stale, mixed-up verdict.

Scope: in-process, session-lifetime, single-user — matches this local POC.
Toggle with PB_SEARCH_CACHE=0 (e.g. to force a genuinely live run on camera).
"""
from __future__ import annotations

import os
import re
import threading

_ENABLED = os.getenv("PB_SEARCH_CACHE", "1") != "0"
_cache: dict[str, str] = {}
_lock = threading.Lock()


def enabled() -> bool:
    return _ENABLED


def _key(query: str) -> str:
    """Normalize so trivial spacing/case differences still hit the same entry."""
    return re.sub(r"\s+", " ", (query or "").strip().lower())


def get(query: str) -> str | None:
    if not _ENABLED:
        return None
    with _lock:
        return _cache.get(_key(query))


def put(query: str, answer: str) -> None:
    """Store only non-empty answers — we cache SUCCESS, never a failed/empty run."""
    if not _ENABLED or not query or not answer:
        return
    with _lock:
        _cache[_key(query)] = answer


def clear() -> None:
    with _lock:
        _cache.clear()


def stats() -> dict:
    with _lock:
        return {"enabled": _ENABLED, "entries": len(_cache)}
