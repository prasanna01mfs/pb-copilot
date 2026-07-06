"""Tests for the harness's retry + model-fallback logic (layer 3), using a fake
runner so this costs zero model calls / API quota — the same reasoning the
guardrail tests in test_finance_tools.py already apply to layer 4.

Covers the behavior added for repeated Gemini 503s ("model currently
experiencing high demand"): retrying the SAME overloaded model just re-queues
behind the same congestion, so after PB_MODEL_FALLBACK_AFTER consecutive 503s
the harness should switch the affected agent(s) to a different model, and
always revert to the primary model once the turn ends (success or failure) so
the next turn tries the primary model fresh.
"""
import asyncio

import pytest
from google.genai.errors import ServerError

from harness import agent_runner as H


def _overload() -> ServerError:
    return ServerError(503, {"error": {"message": "high demand", "status": "UNAVAILABLE"}})


class _FakePart:
    def __init__(self, text):
        self.text = text
        self.function_call = None


class _FakeContent:
    def __init__(self, text):
        self.parts = [_FakePart(text)]


class _FakeEvent:
    def __init__(self, text):
        self.content = _FakeContent(text)

    def is_final_response(self):
        return True


class _FakeAgent:
    def __init__(self, name, model):
        self.name = name
        self.model = model
        self.tools = []


class _FakeRunner:
    """Raises `fail_times` overload errors, then yields a final answer."""

    def __init__(self, agent, fail_times: int, final_text: str = "final answer"):
        self.agent = agent
        self._fail_times = fail_times
        self._final_text = final_text
        self.calls = 0
        self.seen_models = []

    async def run_async(self, **kwargs):
        self.calls += 1
        self.seen_models.append(self.agent.model)
        if self.calls <= self._fail_times:
            raise _overload()
        yield _FakeEvent(self._final_text)


@pytest.fixture(autouse=True)
def _fast_and_isolated(monkeypatch):
    """Keep the test instant and independent of the developer's real .env."""
    monkeypatch.setattr(H, "TIMEOUT_S", 5.0)
    monkeypatch.setattr(H, "BACKOFF_BASE_S", 0.001)
    monkeypatch.setattr(H, "RATE_LIMIT_BACKOFF_S", 0.001)
    monkeypatch.setattr(H, "RATE_LIMIT_BACKOFF_MAX_S", 0.01)
    monkeypatch.setattr(H, "MAX_RETRIES", 2)
    monkeypatch.setattr(H, "MODEL_FALLBACK_AFTER", 2)
    monkeypatch.setattr(H, "GEMINI_MODEL_FALLBACK", "fallback-model")
    monkeypatch.setattr(H, "RESEARCH_MODEL_FALLBACK", "fallback-research-model")


def _run(runner, **kwargs):
    return asyncio.run(
        H.run_through_harness(runner=runner, user_id="u", session_id="s", message="hi", **kwargs)
    )


class TestModelFallback:
    def test_switches_model_after_consecutive_overloads_then_succeeds(self):
        agent = _FakeAgent("orchestrator", "primary-model")
        runner = _FakeRunner(agent, fail_times=2)  # 2 overloads, then succeeds

        result = _run(runner)

        assert "final answer" in result.text
        # First 2 calls used the primary model; the 3rd (granted as the extra
        # fallback attempt) used the switched model.
        assert runner.seen_models == ["primary-model", "primary-model", "fallback-model"]
        assert result.state.model_fallback is True
        assert result.state.fell_back is False
        # Restored afterward so the NEXT turn starts fresh with the primary model.
        assert agent.model == "primary-model"

    def test_research_agent_gets_its_own_fallback_model(self):
        root = _FakeAgent("orchestrator", "primary-model")
        research = _FakeAgent("research_agent", "research-primary")

        class _ToolWrapper:
            def __init__(self, inner):
                self.agent = inner

        root.tools = [_ToolWrapper(research)]

        class _RootRunner(_FakeRunner):
            async def run_async(self, **kwargs):
                self.calls += 1
                self.seen_models.append((root.model, research.model))
                if self.calls <= 2:
                    raise _overload()
                yield _FakeEvent("final answer")

        runner = _RootRunner(root, fail_times=2)
        _run(runner)

        assert runner.seen_models[-1] == ("fallback-model", "fallback-research-model")
        # Both revert afterward.
        assert root.model == "primary-model"
        assert research.model == "research-primary"

    def test_below_threshold_never_switches(self):
        agent = _FakeAgent("orchestrator", "primary-model")
        # Only 1 overload (threshold is 2), then succeeds on the ORIGINAL retry
        # budget — no extra attempt should have been granted, and the model
        # should never change.
        runner = _FakeRunner(agent, fail_times=1)

        result = _run(runner)

        assert "final answer" in result.text
        assert all(m == "primary-model" for m in runner.seen_models)
        assert result.state.model_fallback is False
        assert agent.model == "primary-model"

    def test_persistent_overload_still_falls_back_safely(self):
        agent = _FakeAgent("orchestrator", "primary-model")
        # Overloaded on every attempt, including the extra fallback one -> the
        # turn must still end in the safe fallback text, never a stack trace,
        # and the model must still be restored.
        runner = _FakeRunner(agent, fail_times=999)

        result = _run(runner)

        assert result.text == H.SAFE_FALLBACK
        assert result.state.fell_back is True
        assert result.state.model_fallback is True
        assert agent.model == "primary-model"
