"""Client-side rate limiter — pace outgoing model calls UNDER whatever
per-minute cap applies to your tier, so we don't trigger a 429 by our own call
volume in the first place (prevention is cheaper and faster than reacting to a
429 after the fact).

How it's wired: `throttle_before_model` is added as the FIRST entry in each
agent's before_model_callback list. before_model_callback is awaited by ADK
immediately before the model call, so awaiting a spacing delay there paces the
actual API calls. Because the agents run sequentially (AgentTool awaits each
sub-agent, and the API serializes turns), simple per-model spacing is enough —
no distributed token bucket needed.

Per-model, because flash and flash-lite were SEPARATE free-tier quota buckets
with different limits: lite's per-minute allowance was ~2x flash's, so it
still gets double the configured RPM below — harmless on a paid tier where
neither model is actually named "lite" (see PB_MAX_RPM's default), and it
keeps this logic correct if you ever repin back to a lite model.

Tune with PB_MAX_RPM. Defaults to a generous pay-as-you-go-tier value; if you
know your tier's actual per-minute limit, set this a bit under it instead.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time

logger = logging.getLogger("pb.harness")

# Base RPM (per model); a "lite"-named model gets 2x this, below. 60 is a
# generous pay-as-you-go-tier default — turn it down if you actually see 429s,
# or up once you know your tier's real per-minute limit.
_BASE_RPM = int(os.getenv("PB_MAX_RPM", "60"))

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
