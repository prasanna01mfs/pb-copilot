"""PII redactor — the privacy differentiator (pure logic, no ADK dependency).

What this does and, just as importantly, what it does NOT do (see Build Plan
Sec 1b — do not overclaim):

TRUE:
- Personally identifiable information (name, email, card number, phone,
  account number) is replaced with stable tokens like [NAME_1] BEFORE any
  model call, and swapped back only in the final user-facing response.
- Redaction runs locally, in-process; the token->value mapping never leaves
  this machine.

NOT CLAIMED (and deliberately so):
- We do NOT claim "the model never sees your data." Financial FIGURES (income,
  expenses, balances, SIPs) are intentionally sent to the model — that is what
  lets it reason and advise. Only PII is redacted.

Design rule that keeps that claim honest and avoids destroying the numbers:
- Numeric values are NEVER redacted. The regex patterns for email/card/phone/
  account run on STRINGS only. Financial figures arrive as floats/ints from the
  calculators, so they pass through untouched. The one PII field that shows up
  in tool results is the name, which is alphabetic and matched by a known-name
  literal — it can't collide with a figure.
- String leaves that are THEMSELVES JSON (e.g. MCP wraps tool results as
  JSON-serialized text) are parsed, redacted as a structure, and re-serialized
  — never pattern-matched as raw text — so a number's digits are never at risk
  of being mistaken for a card/account number. See `_redact_string_leaf`.
"""
from __future__ import annotations

import json
import re

# --- PII patterns (applied to strings only) --------------------------------
# Order matters: more specific / longer matches first so a card number isn't
# half-eaten by the phone or account pattern. Each match is replaced by a token
# (which contains no PII), so later patterns can't re-match an earlier hit.
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
# 13-16 digits, optionally grouped with spaces/dashes (typical card formats).
# Digit-boundary lookarounds so it isn't part of a longer number, and it must
# end on a digit (not a trailing separator).
_CARD_RE = re.compile(r"(?<!\d)(?:\d[ -]?){13,16}(?<![ -])(?!\d)")
# Indian mobile: optional +91 prefix, then 10 digits (5+5) starting 6-9, with an
# optional separator between the two halves. Digit-boundary lookarounds stop a
# longer account number being partly eaten as a phone.
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{4}[\s-]?\d{5}(?!\d)")
# Bank account: a 9-18 digit run (min 9 keeps it clear of this app's financial
# figures, which are all <= 8 digits). Runs last, so card/phone match first.
_ACCOUNT_RE = re.compile(r"(?<!\d)\d{9,18}(?!\d)")

# (category, compiled regex) in application order.
_PATTERNS = [
    ("EMAIL", _EMAIL_RE),
    ("CARD", _CARD_RE),
    ("PHONE", _PHONE_RE),
    ("ACCOUNT", _ACCOUNT_RE),
]


class Redactor:
    """Holds one redaction session's mapping so tokens are stable within a turn.

    A fresh Redactor per user turn keeps token numbering deterministic and stops
    mappings leaking across unrelated turns.
    """

    def __init__(self, known_names: list[str] | None = None):
        self.mapping: dict[str, str] = {}      # token -> original value
        self._reverse: dict[str, str] = {}     # original value -> token
        self._counters: dict[str, int] = {}    # category -> next index
        # Longest names first so multi-word names redact before their parts.
        self.known_names = sorted(
            [n for n in (known_names or []) if n and str(n).strip()],
            key=len,
            reverse=True,
        )
        # Redacted snapshots of what was sent to the model, for the debug view.
        self.model_calls: list[str] = []

    def _token_for(self, category: str, value: str) -> str:
        """Return a stable token for `value`, minting a new one if unseen."""
        if value in self._reverse:
            return self._reverse[value]
        idx = self._counters.get(category, 0) + 1
        self._counters[category] = idx
        token = f"[{category}_{idx}]"
        self.mapping[token] = value
        self._reverse[value] = token
        return token

    def redact_text(self, text: str) -> str:
        """Redact all PII in a free-text string (names + all patterns)."""
        if not text:
            return text
        result = text
        # Structured patterns FIRST — an email like "arjun.k@x.com" contains a
        # name substring, so tokenizing the whole email before name-matching
        # stops the name pattern from splitting it.
        for category, regex in _PATTERNS:
            def _sub(m, _cat=category):
                return self._token_for(_cat, m.group(0))

            result = regex.sub(_sub, result)
        # Names last (alphabetic, whole-word, case-insensitive); no digits, so
        # they can't collide with the numeric patterns above.
        for name in self.known_names:
            pattern = re.compile(rf"\b{re.escape(name)}\b", re.IGNORECASE)

            def _sub_name(_m, _name=name):
                return self._token_for("NAME", _name)

            result = pattern.sub(_sub_name, result)
        return result

    def redact_obj(self, obj):
        """Recursively redact STRING leaves of a dict/list; numbers pass through.

        This is what protects financial figures: ints/floats are returned as-is,
        so calculator outputs reach the model intact while any string PII (a
        name, an email) is tokenized.
        """
        if isinstance(obj, str):
            return self._redact_string_leaf(obj)
        if isinstance(obj, dict):
            return {k: self.redact_obj(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(self.redact_obj(v) for v in obj)
        return obj  # int, float, bool, None — never redacted

    def _redact_string_leaf(self, s: str) -> str:
        """Redact one string leaf — structurally if it's itself JSON, else as text.

        Why this exists: MCP wraps every tool result as JSON-serialized TEXT
        (`{"content": [{"type": "text", "text": "<json>"}]}`), so a naive
        `redact_text(s)` here would run the card/account digit-run patterns
        against numbers sitting inside that JSON string — e.g. a savings_rate
        of 0.4558139534883721 has a 16-digit decimal tail that matches the
        13-16-digit card pattern, corrupting the figure into "0.[CARD_1]" and
        breaking the JSON. Parsing back to real objects first means numbers are
        `float`/`int` again and go through `redact_obj`'s numeric passthrough
        instead of the string patterns, exactly like the non-MCP direct-call
        path already does. Falls back to plain text redaction for genuinely
        non-JSON strings (free text, error messages, etc).
        """
        stripped = s.strip()
        if stripped[:1] in "{[":
            try:
                parsed = json.loads(s)
            except (ValueError, TypeError):
                parsed = None
            if isinstance(parsed, (dict, list)):
                return json.dumps(self.redact_obj(parsed))
        return self.redact_text(s)

    def rehydrate(self, text: str) -> str:
        """Swap tokens back to real values — the FINAL user-facing step only."""
        if not text:
            return text
        result = text
        # Replace longer tokens first so [NAME_11] isn't clobbered by [NAME_1].
        for token in sorted(self.mapping, key=len, reverse=True):
            result = result.replace(token, self.mapping[token])
        return result


# --- Module-level "current turn" redactor ----------------------------------
# The ADK before-model hook is a plain callback with no natural place to stash
# per-turn state, so we keep the active Redactor here. The harness starts a fresh
# session each turn and reads it back to rehydrate. Process-local state is fine
# for this local, single-user POC.
_current: Redactor | None = None


def new_session(known_names: list[str] | None = None) -> Redactor:
    """Begin a fresh redaction session for one user turn; returns the Redactor."""
    global _current
    _current = Redactor(known_names=known_names)
    return _current


def current() -> Redactor | None:
    return _current


# --- Convenience one-shot API (matches the plan's redact()/rehydrate()) -----

def redact(data, known_names: list[str] | None = None):
    """Redact a string or nested dict/list. Returns (redacted_data, mapping).

    A standalone helper (independent of the ADK hook) so the redactor can be
    unit-tested and used outside an agent run.
    """
    r = Redactor(known_names=known_names)
    if isinstance(data, str):
        return r.redact_text(data), r.mapping
    return r.redact_obj(data), r.mapping


def rehydrate(model_output: str, mapping: dict[str, str]) -> str:
    """Swap tokens in `model_output` back to real values using `mapping`."""
    result = model_output or ""
    for token in sorted(mapping, key=len, reverse=True):
        result = result.replace(token, mapping[token])
    return result
