"""Gradio UI — a pure HTTP client of api/server.py, nothing more.

No agent logic lives here. All routing, redaction, retries and validation
happen server-side in the harness; this file only renders a chat box, a
profile picker, and a debug view of what the API already returns
(routed_to + harness summary), so that engineering is VISIBLE here, not
re-implemented.

Run (with the API server already running — see the printed instructions):
    python ui/app.py
"""
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

import gradio as gr
import httpx
from dotenv import load_dotenv

load_dotenv()

# Put the project root on sys.path. Running `python ui/app.py` only adds ui/'s
# own directory to the path (not the project root), so the observability import
# below would otherwise fail with ModuleNotFoundError.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from observability import tracker as observability  # noqa: E402

logger = logging.getLogger("pb.ui")

API_BASE_URL = os.getenv("PB_API_URL", "http://127.0.0.1:8000")
_client = httpx.AsyncClient(base_url=API_BASE_URL, timeout=90.0)

def _format_active(data: dict) -> str:
    """Render the active-profile panel from GET /profile/active or the upload
    response. Shows the REAL name + key attributes on purpose: it's the user's
    own uploaded data, echoed back so they can confirm the right planner
    parsed (see api/server._profile_display for why this is not a leak path)."""
    if not data.get("loaded"):
        return "**No profile loaded.** Upload a planner (.xlsx) to begin."
    name = data.get("name") or "(unnamed)"
    attrs = []
    if data.get("age") is not None:
        attrs.append(f"age {data['age']}")
    if data.get("dependents") is not None:
        attrs.append(f"{data['dependents']} dependent(s)")
    if data.get("risk_appetite"):
        attrs.append(f"{data['risk_appetite']} risk")
    money = []
    if data.get("monthly_income"):
        money.append(f"income ₹{data['monthly_income']:,.0f}/mo")
    if data.get("net_worth"):
        money.append(f"net worth ₹{data['net_worth']:,.0f}")
    line = f"**Active profile: {name}** ✓"
    if attrs:
        line += "  \n" + " · ".join(attrs)
    if money:
        line += "  \n" + " · ".join(money)
    return line


async def fetch_active_profile() -> tuple[str, str]:
    """(profile_status_text, error_text) — called on page load to show whatever
    is currently active (nothing, until the first upload)."""
    try:
        r = await _client.get("/profile/active")
        r.raise_for_status()
        return _format_active(r.json()), ""
    except httpx.HTTPError as exc:
        logger.warning("fetch_active_profile failed: %s", type(exc).__name__)
        return "**Active profile:** (unknown)", f"⚠️ Could not reach API at {API_BASE_URL}: {exc}"


async def upload_profile(filepath: str | None):
    """Upload-planner handler — POSTs the .xlsx to /profile/upload.

    Returns (active_profile_markdown, error_markdown, chatbot_history). On a
    SUCCESSFUL upload the chat is cleared to [] — the previous conversation was
    about a different person's finances, so it must not carry over to the newly
    loaded profile. On failure the chat is left untouched (gr.update()).

    The API derives the profile name from the filename, so the original name is
    sent as the multipart filename — but NOT logged here (a filename like
    '...Arjun.xlsx' contains PII; we log only that an upload happened).
    """
    if not filepath:
        return gr.update(), "Please choose a .xlsx planner file first.", gr.update()
    started = time.monotonic()
    filename = os.path.basename(filepath)
    try:
        with open(filepath, "rb") as fh:
            files = {"file": (filename, fh,
                              "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
            r = await _client.post("/profile/upload", files=files)
        r.raise_for_status()
        logger.info("upload_profile: planner uploaded and parsed (%.2fs)", time.monotonic() - started)
        # Success -> update profile panel, clear any error, CLEAR THE CHAT.
        return _format_active(r.json()), "", []
    except httpx.HTTPStatusError as exc:
        logger.warning("upload_profile failed: HTTP %s", exc.response.status_code)
        try:
            detail = exc.response.json().get("detail", str(exc))
        except Exception:  # noqa: BLE001 — non-JSON error body
            detail = str(exc)
        return gr.update(), f"⚠️ {detail}", gr.update()
    except httpx.HTTPError as exc:
        logger.warning("upload_profile failed: %s", type(exc).__name__)
        return gr.update(), f"⚠️ Could not reach API at {API_BASE_URL}: {exc}", gr.update()


async def send_message(message: str, history: list[dict]):
    """Chat submit handler — an async GENERATOR so the UI updates progressively.

    `history` is Gradio's "messages" format: a list of {"role","content"} dicts
    (the only format gr.Chatbot supports as of Gradio 6). A turn can take
    15–60s (research/cross-agent), so we don't wait for the whole thing before
    touching the screen:
      1st yield — post the user's message + clear the box + a "…thinking…"
                  placeholder IMMEDIATELY, so the user sees their question land
                  and isn't tempted to click Send again;
      2nd yield — replace the placeholder with the real answer.
    """
    message = (message or "").strip()
    if not message:
        yield history, ""
        return

    # Log METADATA only — message length, never the text itself, which may be
    # PII the user typed (redaction happens server-side, but this log line is
    # local to the UI process and must not become its own leak path).
    logger.info("send_message: sending message (%d chars)", len(message))
    started = time.monotonic()

    history = history + [{"role": "user", "content": message}]
    # Immediate feedback: user bubble shows, textbox clears, placeholder appears.
    yield history + [{"role": "assistant", "content": "_…thinking…_"}], ""

    try:
        r = await _client.post("/chat", json={"message": message})
        r.raise_for_status()
        data = r.json()
        answer = data["answer"]
        logger.info("send_message: routed=%s (%.2fs)", data.get("routed_to") or "-",
                    time.monotonic() - started)
    except httpx.HTTPStatusError as exc:
        logger.warning("send_message failed: HTTP %s", exc.response.status_code)
        try:
            detail = exc.response.json().get("detail", str(exc))
        except Exception:  # noqa: BLE001
            detail = str(exc)
        answer = f"⚠️ API error: {detail}"
    except httpx.HTTPError as exc:
        logger.warning("send_message failed: %s (%.2fs)", type(exc).__name__, time.monotonic() - started)
        answer = f"⚠️ Could not reach API at {API_BASE_URL}: {exc}"

    # Replace the placeholder with the real answer.
    yield history + [{"role": "assistant", "content": answer}], ""


with gr.Blocks(title="PB Copilot") as demo:
    gr.Markdown(
        "# PB Copilot\n"
        "Privacy-first personal finance & research assistant — talks to a FastAPI backend "
        "that runs an ADK orchestrator (FinanceAgent + ResearchAgent) through a reliability "
        "harness. *Educational guidance only, not licensed financial advice.*"
    )

    # Top strip: a COMPACT uploader on the left, the active-profile summary on
    # the right (so "what's loaded" sits right next to the upload control).
    with gr.Row():
        with gr.Column(scale=1, min_width=240):
            upload = gr.File(
                label="Upload planner (.xlsx)",
                file_types=[".xlsx"],
                file_count="single",
                height=125,  # compact, but tall enough to fit the drop-zone text
            )
        with gr.Column(scale=2):
            active_profile_md = gr.Markdown("**No profile loaded.** Upload a planner (.xlsx) to begin.")
            error_md = gr.Markdown("")

    # Chat stacked BELOW the uploader, full width.
    # Gradio 6 dropped the tuples format entirely — gr.Chatbot now ONLY accepts
    # the "messages" shape ({"role","content"} dicts), so there's no `type=`
    # kwarg to set (verified against the installed 6.19.0).
    chatbot = gr.Chatbot(height=620, label="PB Copilot")
    msg_box = gr.Textbox(
        label="Message",
        placeholder="e.g. 'what should I fix first?' or 'latest RBI repo rate news'",
        lines=2,
    )
    send_btn = gr.Button("Send", variant="primary")

    # Uploading a new planner also CLEARS the chat (outputs include chatbot),
    # so a previous person's conversation never bleeds into the new profile.
    upload.upload(fn=upload_profile, inputs=[upload],
                  outputs=[active_profile_md, error_md, chatbot])
    demo.load(fn=fetch_active_profile, inputs=None, outputs=[active_profile_md, error_md])

    send_btn.click(fn=send_message, inputs=[msg_box, chatbot], outputs=[chatbot, msg_box])
    msg_box.submit(fn=send_message, inputs=[msg_box, chatbot], outputs=[chatbot, msg_box])


if __name__ == "__main__":
    # A separate process from api/server.py — needs its own logging setup.
    observability.configure_logging(verbose=os.getenv("PB_DEBUG_REDACTION") == "1")
    print(f"Gradio UI starting. Talking to PB Copilot API at {API_BASE_URL}")
    print("(Make sure `uvicorn api.server:app --port 8000` is already running.)")
    demo.launch()
