"""Shared test setup. Every test runs offline and can never spend API money."""

from __future__ import annotations

import importlib
from concurrent.futures import Future

import anthropic
import pytest

from fake_llm import FakeLLM

# Where the code under test lives.
STAGES_MODULE = "idea_finder.pipeline"   # has ThreadPoolExecutor / as_completed
CLIENT_MODULE = "idea_finder.llm"        # has the cached _client


@pytest.fixture(autouse=True)
def block_real_api(monkeypatch):
    """Fail loudly if any test tries to build a real Anthropic client."""
    def refuse(*args, **kwargs):
        raise RuntimeError("Tests must not create a real Anthropic client. Use the fake_llm fixture.")
    monkeypatch.setattr(anthropic, "Anthropic", refuse)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    monkeypatch.setenv("COLUMNS", "100")  # fixed width, so --help text is stable


@pytest.fixture(autouse=True)
def fresh_trace_state(monkeypatch):
    """Each test starts with no run folder set, so traces never leak between tests."""
    from idea_finder import llm
    monkeypatch.setattr(llm, "_trace_path", None)
    monkeypatch.setattr(llm, "_run_id", "")
    monkeypatch.setattr(llm, "_call_count", 0)


@pytest.fixture
def fake_llm(monkeypatch) -> FakeLLM:
    """Put a FakeLLM where the code keeps its client."""
    fake = FakeLLM()
    monkeypatch.setattr(importlib.import_module(CLIENT_MODULE), "_client", fake)
    return fake


class SyncExecutor:
    """Runs each task at once, in order. Makes threaded stages deterministic in tests."""

    def __init__(self, *args, **kwargs) -> None:
        pass

    def __enter__(self) -> "SyncExecutor":
        return self

    def __exit__(self, *exc) -> bool:
        return False

    def submit(self, fn, *args, **kwargs) -> Future:
        fut: Future = Future()
        try:
            fut.set_result(fn(*args, **kwargs))
        except Exception as e:
            fut.set_exception(e)
        return fut


@pytest.fixture
def sync_pool(monkeypatch) -> None:
    """Replace the thread pool so results come back in submission order."""
    module = importlib.import_module(STAGES_MODULE)
    monkeypatch.setattr(module, "ThreadPoolExecutor", SyncExecutor)
    monkeypatch.setattr(module, "as_completed", lambda futures: list(futures))
