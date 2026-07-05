"""Profile store — the memory layer (load / save / update the active profile).

This is deliberately simple for a local single-user POC: one "active profile"
held in process and mirrored to a JSON file so it persists across sessions
(the plan's "memory across sessions" concept). Excel is just a front door —
once parsed, everything downstream uses the same internal dict, and we can
persist/restore it without re-reading the spreadsheet.

The finance ADK tools read the active profile from here, which keeps the
model-facing tool signatures tiny (the model can't pass a wrong/huge profile
dict — it just asks "what's the emergency fund?" and the tool reads state).
"""
from __future__ import annotations

import json
import os

from privacy import redactor
from tools.excel_parser import parse_planner
from tools import finance_tools

# Where the active profile is persisted between runs. Kept out of git via the
# data/ path being fine to commit, but this specific file is runtime state; add
# it to .gitignore if you don't want it committed.
_STORE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "active_profile.json")

# The bundled synthetic planners the demo can load by short name.
AVAILABLE_PROFILES = {
    "arjun": "data/PB_Planner_Profile1_Arjun.xlsx",
    "priya": "data/PB_Planner_Profile2_Priya.xlsx",
}

# In-process active profile. None until something is loaded.
_active: dict | None = None

# Monotonic version counter, bumped whenever the active profile changes (load or
# field update). The harness records this per turn (Section 2c layer 6) so the
# debug panel / logs show which profile version produced a given answer — the
# basis for the "last time your emergency fund was X, now it's Y" memory story.
_version: int = 0


def load_from_excel(path: str) -> dict:
    """Parse an Excel planner, compute derived figures, set + persist as active."""
    profile = parse_planner(path)
    finance_tools.compute_derived(profile)
    set_active(profile)
    return profile


def load_sample(name: str) -> dict:
    """Load one of the bundled sample profiles by short name ('arjun'/'priya')."""
    key = name.strip().lower()
    if key not in AVAILABLE_PROFILES:
        raise KeyError(f"Unknown sample profile '{name}'. Choose from {list(AVAILABLE_PROFILES)}.")
    return load_from_excel(AVAILABLE_PROFILES[key])


def set_active(profile: dict) -> None:
    global _active, _version
    _active = profile
    _version += 1
    _save_json()


def version() -> str:
    """A stable tag identifying the current profile state, e.g. 'Arjun#3'.

    Combines name + counter so a log line unambiguously ties an answer to the
    exact profile snapshot that produced it, even after in-session edits.

    Returns the RAW name — internal/state-tracking use only. Anything that
    logs, prints, or returns this over an API boundary must use
    `redacted_version()` instead (see there for why this one isn't safe there).
    """
    if _active is None:
        return "none#0"
    return f"{_active['profile'].get('name', 'unknown')}#{_version}"


def redacted_version() -> str:
    """Same tag as `version()`, but with the profile name tokenized.

    `version()`'s raw name is fine for internal bookkeeping, but it was also
    flowing unredacted into the harness log and every API response field
    (/health, /profile/active, /chat) — a real gap against the "no raw PII
    outside the final user-facing answer" rule, found by tracing every path
    PII can take. This is the fix: always route the tag through the redactor
    before it leaves this module.

    Reuses the active per-turn Redactor if one exists (`redactor.current()`),
    so the token matches whatever was already assigned to this name elsewhere
    in the same turn. Falls back to a one-off Redactor when no turn is active
    (e.g. a bare `GET /health` outside any harnessed call) so the raw name is
    NEVER emitted regardless of context — just with a token that won't be
    stable across separate calls in that case.
    """
    tag = version()
    if _active is None:
        return tag  # "none#0" — no name to protect
    name = _active["profile"].get("name")
    if not name:
        return tag
    session = redactor.current()
    if session is not None:
        return session.redact_text(tag)
    return redactor.Redactor(known_names=[name]).redact_text(tag)


def get_active() -> dict:
    """Return the active profile, or raise a clear error if none is loaded.

    The finance tools surface this message to the model as a clean tool error
    (never a stack trace), so the agent can ask the user to upload a profile.
    """
    if _active is None:
        raise RuntimeError("No profile is loaded. Upload a planner (.xlsx) first.")
    return _active


def has_active() -> bool:
    return _active is not None


def update_field(section: str, key: str, value) -> dict:
    """Update one field in the active profile, recompute derived, persist.

    Enables the "last time your emergency fund was X, now it's Y" memory demo.
    Only allows updating existing sections to avoid silent schema drift.
    """
    profile = get_active()
    if section not in profile:
        raise KeyError(f"Unknown profile section '{section}'.")
    if not isinstance(profile[section], dict):
        raise TypeError(f"Section '{section}' is not a simple field map.")
    profile[section][key] = value
    finance_tools.compute_derived(profile)
    set_active(profile)
    return profile


def _save_json() -> None:
    if _active is None:
        return
    os.makedirs(os.path.dirname(_STORE_PATH), exist_ok=True)
    with open(_STORE_PATH, "w", encoding="utf-8") as fh:
        json.dump(_active, fh, indent=2, default=str)


def load_json() -> dict | None:
    """Restore a previously persisted active profile from disk, if present."""
    global _active
    if not os.path.exists(_STORE_PATH):
        return None
    with open(_STORE_PATH, "r", encoding="utf-8") as fh:
        _active = json.load(fh)
    return _active
