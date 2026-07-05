"""Client-side rate limiter — pace outgoing model calls UNDER the free-tier
per-minute cap, so we don't trigger a 429 by our own call volume in the first
place (prevention is cheaper and faster than reacting to a 429 after the fact).

How it's wired: `throttle_before_model` is added as the FIRST entry in each
agent's before_model_callback list. before_model_callback is awaited by ADK
immediately before the model call, so awaiting a spacing delay there paces the
actual API calls. Because the agents run sequentially (AgentTool awaits each
sub-agent, and the API serializes turns), simple per-model spacing is enough —
no distributed token bucket needed.

Per-model, because flash and flash-lite are SEPARATE free-tier quota buckets
with different limits: lite's per-minute allowance is ~2x flash's, so it gets
double the RPM here. Defaults sit comfortably under the caps (≈15 flash /
≈30 lite) to leave headroom for calls we don't control (e.g. Google Search
grounding makes its own internal model calls).

Tune with PB_MAX_RPM (the flash/flagship bucket; lite is auto-doubled).
"""
from __future__ import annotations

import asyncio
import logging
import os
import time

logger = logging.getLogger("pb.harness")

# Flash/flagship bucket RPM. Kept a few under the ~15 free-tier cap for margin;
# lite is doubled below. Turn it up if you move to a paid tier.
_BASE_RPM = int(os.getenv("PB_MAX_RPM", "12"))

_locks: dict[str, asyncio.Lock] = {}
_last_call: dict[str, float] = {}


def _rpm_for(model: str) -> int:
    # Flash-Lite gets roughly double the free-tier RPM of flagship Flash.
    return _BASE_RPM * 2 if "lite" in (model or "").lower() else _BASE_RPM


def _lock_for(model: str) -> asyncio.Lock:
    # Safe to create lazily without a guard: asyncio is single-threaded and
    # there's no await between the check and the assignment.
    lock = _locks.get(model)
    if lock is None:
        lock = _locks[model] = asyncio.Lock()
    return lock


async def throttle(model: str) -> None:
    """Block until >= (60 / RPM) seconds have elapsed since this model's last call."""
    interval = 60.0 / max(1, _rpm_for(model))
    async with _lock_for(model):
        wait = interval - (time.monotonic() - _last_call.get(model, 0.0))
        if wait > 0:
            if wait > 0.5:  # only log meaningful waits, not sub-second spacing
                logger.info("throttle: pacing %s — waiting %.1fs (cap %d rpm)",
                            model, wait, _rpm_for(model))
            await asyncio.sleep(wait)
        _last_call[model] = time.monotonic()


async def throttle_before_model(callback_context, llm_request) -> None:
    """before_model_callback entry — pace this call under the per-minute cap.

    Placed FIRST in the callback list so the llm_call latency timer that runs
    next measures true API round-trip time, not the throttle wait.
    """
    await throttle(getattr(llm_request, "model", None) or "unknown")
    return None
