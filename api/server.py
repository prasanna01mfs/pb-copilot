"""FastAPI layer — a thin, UI-agnostic POST /chat over the orchestrator.

Purpose (per the plan): decouple the UI from the agents so Gradio now and Angular
later both talk to the SAME endpoint without touching agent code. This file adds
NO logic of its own — it just turns an HTTP request into one harnessed turn and
returns the answer. All the real behavior (routing, redaction, retries,
validation) lives in the harness, so the UI stays a thin client and can't
drift from the real behavior.

Run:
    uvicorn api.server:app --reload --port 8000     (from the project root)
"""
from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

# Load .env by ABSOLUTE path (project root), so it works no matter what CWD
# uvicorn is launched from — find_dotenv() searches from the caller's dir and
# would miss it when the module lives under api/.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from fastapi import FastAPI, File, HTTPException, Request, UploadFile  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from google.adk.runners import Runner  # noqa: E402
from google.adk.sessions import InMemorySessionService  # noqa: E402

from agents.orchestrator import root_agent  # noqa: E402
from harness import agent_runner  # noqa: E402
from memory import profile_store  # noqa: E402
from observability import tracker as observability  # noqa: E402

APP_NAME = "pb_copilot"
USER_ID = "api_user"

# PB_DEBUG_REDACTION=1 surfaces the
# detailed traces (full redacted payload, harness diagnostics) in the server's
# console output, on top of the always-on per-turn observability summary.
DEBUG_REDACTION = os.getenv("PB_DEBUG_REDACTION") == "1"

# Holds the singleton Runner, the session service, and the CURRENT session id
# (which changes on every profile load — see _reset_session). A lock serializes
# turns because the redaction session (and the profile store) are process-global
# — fine for this local, single-user POC, and the lock keeps concurrent requests
# from clobbering each other's PII mapping.
_ctx: dict = {}
_turn_lock = asyncio.Lock()


async def _reset_session() -> str:
    """Start a FRESH ADK session, wiping the model's conversation context.

    Called at startup and on every profile change. The conversation history is
    what ADK re-sends to the model each turn, so a new session = empty context.
    This is the backend half of "clear on new profile": without it, uploading
    Priya after chatting as Arjun would still carry Arjun's Q&A into Priya's
    context (growing the window AND bleeding one person's conversation into
    another's). The old session is deleted so sessions don't pile up in memory.
    """
    session_service = _ctx["session_service"]
    old = _ctx.get("session_id")
    new_id = f"session-{uuid.uuid4().hex[:12]}"
    await session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=new_id)
    _ctx["session_id"] = new_id
    if old:
        try:
            await session_service.delete_session(app_name=APP_NAME, user_id=USER_ID, session_id=old)
        except Exception:  # noqa: BLE001 — best-effort cleanup, never fatal
            pass
    return new_id


@asynccontextmanager
async def lifespan(app: FastAPI):
    observability.configure_logging(verbose=DEBUG_REDACTION)
    _ctx["session_service"] = InMemorySessionService()
    _ctx["runner"] = Runner(agent=root_agent, app_name=APP_NAME,
                            session_service=_ctx["session_service"])
    await _reset_session()  # create the first session
    # No profile is auto-loaded: the UI is upload-driven, so the active profile
    # must reflect ONLY what the user uploaded (showing a pre-loaded default
    # would be exactly the confusion we're avoiding). Until an upload happens,
    # the finance tools return a clean "load a profile first" message.
    yield


app = FastAPI(title="PB Copilot API", version="1.0", lifespan=lifespan)


@app.middleware("http")
async def observability_middleware(request: Request, call_next):
    """Log every request's method/path/status/latency — transport metadata
    ONLY. Deliberately does not read request or response bodies: a /chat
    request body can carry a user-typed message (PII before it's redacted
    downstream), and a /chat response body carries the REHYDRATED real
    answer — logging either would open a new PII path outside the
    redact/rehydrate contract. See observability.tracker.record_http_request.
    """
    started = time.monotonic()
    response = await call_next(request)
    observability.record_http_request(
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        elapsed_s=time.monotonic() - started,
        client_host=request.client.host if request.client else None,
    )
    return response


class ChatRequest(BaseModel):
    message: str
    profile_id: str | None = None  # "arjun" | "priya"; omit to keep the current profile


class ChatResponse(BaseModel):
    answer: str
    profile: str          # e.g. "[NAME_1]#3" — which profile version produced this (name tokenized)
    routed_to: list[str]  # which specialist(s) ran — the multi-agent proof
    harness: str          # one-line harness trace (attempts, validation, fallback)


# Uploaded planners are parsed from a temp file, then discarded. Cap the size:
# the real planners are ~16 KB; this stops a pathological upload from filling
# disk while parsing. Local single-user POC, so the limit is generous.
_MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB


def _profile_display() -> dict:
    """The active profile as the UI should DISPLAY it — includes the REAL name.

    Deliberate, documented exception to the redact-in-API-responses rule: this
    is the user's OWN uploaded data echoed back to them in their OWN UI, so
    they can confirm the right planner parsed (a wrong/tokenized name would
    just confuse them). Two things keep this from reopening a leak path:
      - our API middleware logs request/response METADATA only, never bodies,
        so this name is not written to any log by us;
      - the name that flows into logs/observability and the /chat metadata
        field still goes through redacted_version() — only THIS display
        endpoint surfaces the real value.
    Financial figures are never PII (they're sent to the model regardless), so
    including a couple here to help the user verify the parse is fine.
    """
    p = profile_store.get_active()
    prof = p["profile"]
    derived = p.get("derived", {})
    return {
        "name": prof.get("name"),
        "age": prof.get("age"),
        "dependents": prof.get("dependents"),
        "risk_appetite": prof.get("risk_appetite"),
        "monthly_income": derived.get("monthly_income"),
        "net_worth": derived.get("net_worth"),
    }


@app.get("/health")
async def health() -> dict:
    # redacted_version(), not version(): every field this API returns is
    # potentially logged by something outside our control (a reverse proxy,
    # API gateway, monitoring middleware) — the raw name has no business
    # leaving this module. See memory/profile_store.redacted_version.
    return {"status": "ok", "profile": profile_store.redacted_version()}


@app.get("/profile/active")
async def active_profile() -> dict:
    """The currently-loaded profile as the UI should display it (real name +
    key attributes), or {"loaded": false} if nothing is loaded yet."""
    if not profile_store.has_active():
        return {"loaded": False}
    return {"loaded": True, **_profile_display()}


@app.post("/profile/upload")
async def upload_profile(file: UploadFile = File(...)) -> dict:
    """Parse an uploaded .xlsx planner and make it the active profile.

    Replaces the old sample-picker flow: the UI uploads Arjun/Priya (or any
    planner in the same template) instead of choosing from a fixed list. The
    profile NAME is derived from the uploaded filename (the template has no
    name cell), so uploading 'PB_Planner_Profile1_Arjun.xlsx' yields 'Arjun'.
    """
    if not (file.filename or "").lower().endswith(".xlsx"):
        raise HTTPException(status_code=400, detail="Please upload a .xlsx planner file.")

    async with _turn_lock:
        tmpdir = tempfile.mkdtemp(prefix="pb_upload_")
        # Preserve the original filename in the temp path — the parser derives
        # the profile name from it.
        tmp_path = os.path.join(tmpdir, os.path.basename(file.filename))
        try:
            size = 0
            with open(tmp_path, "wb") as out:
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > _MAX_UPLOAD_BYTES:
                        raise HTTPException(status_code=413, detail="File too large (max 5 MB).")
                    out.write(chunk)
            try:
                profile_store.load_from_excel(tmp_path)
            except HTTPException:
                raise
            except Exception as exc:  # noqa: BLE001 — parser errors -> clean 400
                raise HTTPException(
                    status_code=400,
                    detail=f"Could not parse that planner: {exc}. "
                           "Make sure it matches the expected template.",
                )
            # New profile -> fresh model context (see _reset_session). This is
            # the backend counterpart to the UI clearing the chat on upload.
            await _reset_session()
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    return {"loaded": True, **_profile_display()}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    if not os.getenv("GOOGLE_API_KEY"):
        raise HTTPException(status_code=500, detail="GOOGLE_API_KEY is not set on the server.")
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message must not be empty.")

    async with _turn_lock:
        # Optional profile switch (loads a bundled sample by name).
        # Switching the profile also resets the session, so the new profile
        # doesn't inherit the previous one's conversation context.
        if req.profile_id:
            try:
                profile_store.load_sample(req.profile_id)
            except KeyError as exc:
                raise HTTPException(status_code=400, detail=str(exc))
            await _reset_session()

        known_names = []
        if profile_store.has_active():
            name = profile_store.get_active()["profile"].get("name")
            if name:
                known_names.append(name)

        result = await agent_runner.run_through_harness(
            runner=_ctx["runner"],
            user_id=USER_ID,
            session_id=_ctx["session_id"],  # current session (reset on profile change)
            message=req.message,
            known_names=known_names,
        )

    return ChatResponse(
        answer=result.text,
        # redacted_version(), not version() — same reasoning as /health. Note
        # this is deliberately inconsistent with `answer`, which DOES show the
        # real name: `answer` is "the final response shown to the user" (the
        # one place the privacy contract requires rehydration); `profile` is
        # response metadata, held to the stricter no-raw-PII-in-output-fields
        # rule everywhere else.
        profile=profile_store.redacted_version(),
        routed_to=result.state.agents_run,
        harness=result.state.summary(),
    )
