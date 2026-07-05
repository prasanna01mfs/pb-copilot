"""ADK glue for the privacy layer — the before-model hook + payload debug log.

Kept separate from redactor.py so the redaction logic stays ADK-free and unit
testable. This module is the ONE place where PII is scrubbed out of the actual
outgoing model request, so it's easy to audit that nothing bypasses it.

Wiring: attach `redaction_before_model` as `before_model_callback` on every
agent (orchestrator + both specialists). before_model_callback fires with the
fully-assembled LlmRequest right before it goes to Gemini — including tool
results already appended to the conversation — so redacting here covers user
text, tool outputs (e.g. the profile name from get_profile_summary), and the
system instruction in a single choke point.
"""
from __future__ import annotations

import logging
import os

from privacy import redactor

logger = logging.getLogger("pb.privacy")

# Redacted snapshots are also appended here so you can inspect them after a run
# without wiring up a log handler. Contains TOKENS only — never raw PII.
_PAYLOAD_LOG = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "logs", "outgoing_model_payload.log"
)


def _redact_system_instruction(si, r: redactor.Redactor):
    """system_instruction may be a str, a Content, or a list of either."""
    if si is None:
        return si
    if isinstance(si, str):
        return r.redact_text(si)
    if isinstance(si, (list, tuple)):
        return type(si)(_redact_system_instruction(x, r) for x in si)
    # Content-like object with .parts
    parts = getattr(si, "parts", None)
    if parts:
        for p in parts:
            if getattr(p, "text", None):
                p.text = r.redact_text(p.text)
    return si


async def redaction_before_model(callback_context, llm_request):
    """Redact PII in-place in the outgoing LlmRequest. Returns None to proceed.

    If no redaction session is active (redactor.new_session was not called for
    this turn), we do nothing rather than guess — the caller owns the mapping.
    """
    r = redactor.current()
    if r is None:
        return None

    # 1) System instruction.
    cfg = getattr(llm_request, "config", None)
    if cfg is not None and getattr(cfg, "system_instruction", None):
        cfg.system_instruction = _redact_system_instruction(cfg.system_instruction, r)

    # 2) Conversation contents: user/model text, tool results, tool-call args.
    for content in llm_request.contents or []:
        for part in content.parts or []:
            if getattr(part, "text", None):
                part.text = r.redact_text(part.text)
            fr = getattr(part, "function_response", None)
            if fr is not None and fr.response is not None:
                fr.response = r.redact_obj(fr.response)
            fc = getattr(part, "function_call", None)
            if fc is not None and getattr(fc, "args", None):
                fc.args = r.redact_obj(fc.args)

    _log_outgoing(callback_context, llm_request, r)
    return None


def _log_outgoing(callback_context, llm_request, r: redactor.Redactor) -> None:
    """Record the REDACTED outgoing payload so redaction is verifiable.

    Security rule (Build Plan Sec 8 — no logging of raw PII): this runs AFTER
    redaction, so only tokens are ever written. The token->value mapping is
    never logged.
    """
    agent = getattr(callback_context, "agent_name", "?")
    lines = [f"===== OUTGOING MODEL CALL (agent={agent}) — REDACTED ====="]
    for content in llm_request.contents or []:
        role = getattr(content, "role", "?")
        for part in content.parts or []:
            if getattr(part, "text", None):
                lines.append(f"[{role}] text: {part.text}")
            fr = getattr(part, "function_response", None)
            if fr is not None and fr.response is not None:
                lines.append(f"[{role}] tool_result({fr.name}): {fr.response}")
            fc = getattr(part, "function_call", None)
            if fc is not None and getattr(fc, "args", None):
                lines.append(f"[{role}] tool_call({fc.name}): {fc.args}")
    snapshot = "\n".join(lines)

    r.model_calls.append(snapshot)
    logger.info("\n%s", snapshot)  # visible on console when pb.privacy logging is on
    try:
        os.makedirs(os.path.dirname(_PAYLOAD_LOG), exist_ok=True)
        with open(_PAYLOAD_LOG, "a", encoding="utf-8") as fh:
            fh.write(snapshot + "\n\n")
    except OSError:
        pass  # never let debug logging break a run
