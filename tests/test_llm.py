"""Tests for the LLM gateway: tracing, cost and error handling."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from fake_llm import FakeBadRequest, FakeLLM, make_message
from idea_finder import llm
from test_golden import run_pipeline

TRACE_FIELDS = {"call_id", "ts", "run_id", "purpose", "model", "request_hash", "input_tokens",
                "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens",
                "latency_s", "stop_reason", "cost_usd", "response"}


def reply(model: str = "claude-sonnet-5-5", **kw) -> object:
    """A plain text response with chosen token counts."""
    return make_message(model, [{"type": "text", "text": "ok"}], "end_turn",
                        kw.get("input_tokens", 1000), kw.get("output_tokens", 500),
                        cache_write=kw.get("cache_write", 0), cache_read=kw.get("cache_read", 0))


def usage(fresh: int, out: int, written: int = 0, read: int = 0) -> SimpleNamespace:
    return SimpleNamespace(input_tokens=fresh, output_tokens=out,
                           cache_creation_input_tokens=written, cache_read_input_tokens=read)


def ask(model: str = "claude-sonnet-5-5") -> object:
    return llm.call(purpose="test", model=model, messages=[{"role": "user", "content": "hi"}])


def test_call_writes_one_trace_line(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "_client", FakeLLM(script=[reply(cache_write=200, cache_read=3000)]))
    llm.start_run(tmp_path / "run1")
    resp = ask()
    assert resp.content[0].text == "ok"
    records = llm.read_trace(tmp_path / "run1" / "trace.jsonl")
    assert len(records) == 1
    rec = records[0]
    assert set(rec) == TRACE_FIELDS
    assert rec["call_id"] == 1 and rec["run_id"] == "run1" and rec["purpose"] == "test"
    assert rec["stop_reason"] == "end_turn"
    assert rec["response"]["content"][0]["text"] == "ok"
    # 1000 in x $2 + 200 cache write x $2.50 + 3000 cache read x $0.10 + 500 out x $10, per million
    assert rec["cost_usd"] == pytest.approx(0.0078)


@pytest.mark.parametrize("model, u, dollars", [
    ("claude-haiku-5-5", usage(1000, 100), (1000 * 0.10 + 100 * 0.50) / 1e6),
    ("claude-haiku-5-5", usage(150_000, 1000), (150_000 * 0.50 + 1000 * 2.50) / 1e6),  # long prompt
    ("claude-haiku-5-5", usage(10_000, 1000, read=95_000),                              # long via cache
     (10_000 * 0.50 + 95_000 * 0.05 + 1000 * 2.50) / 1e6),
    ("claude-sonnet-5-5", usage(150_000, 1000), (150_000 * 2.00 + 1000 * 10.00) / 1e6),  # no long tier
    ("claude-opus-5-5", usage(2000, 300, written=600), (2000 * 4 + 600 * 5 + 300 * 20) / 1e6),
])
def test_estimate_cost(model, u, dollars):
    assert llm.estimate_cost(model, u) == pytest.approx(dollars)


def test_unknown_model_has_no_cost_but_still_works(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "_client", FakeLLM(script=[reply("claude-made-up-1")]))
    llm.start_run(tmp_path)
    ask("claude-made-up-1")
    assert llm.read_trace(tmp_path / "trace.jsonl")[0]["cost_usd"] is None
    assert llm.trace_totals(tmp_path / "trace.jsonl")["unpriced_calls"] == 1


def test_failed_call_is_traced_then_raised(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "_client", FakeLLM(reject_forced_tool_choice_for={"claude-sonnet-5-5"}))
    llm.start_run(tmp_path)
    with pytest.raises(FakeBadRequest):
        llm.call(purpose="test", model="claude-sonnet-5-5", messages=[{"role": "user", "content": "hi"}],
                 tools=[{"name": "t", "input_schema": {"type": "object"}}],
                 tool_choice={"type": "tool", "name": "t"})
    rec = llm.read_trace(tmp_path / "trace.jsonl")[0]
    assert rec["error"].startswith("FakeBadRequest")
    assert "cost_usd" not in rec


def test_threads_write_whole_lines(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "_client", FakeLLM(script=[reply() for _ in range(100)]))
    llm.start_run(tmp_path)
    threads = [threading.Thread(target=lambda: [ask() for _ in range(25)]) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    records = llm.read_trace(tmp_path / "trace.jsonl")  # fails if any line got mixed up
    assert sorted(r["call_id"] for r in records) == list(range(1, 101))


def test_trace_totals_and_rerun_numbering(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "_client", FakeLLM(script=[reply(), reply(), reply()]))
    llm.start_run(tmp_path)
    ask()
    ask()
    llm.start_run(tmp_path)  # a second command in the same run folder
    ask()
    records = llm.read_trace(tmp_path / "trace.jsonl")
    assert [r["call_id"] for r in records] == [1, 2, 3]
    totals = llm.trace_totals(tmp_path / "trace.jsonl")
    assert totals["calls"] == 3 and totals["errors"] == 0
    assert totals["input_tokens"] == 3000 and totals["output_tokens"] == 1500
    assert totals["cost_usd"] == pytest.approx(3 * (1000 * 2 + 500 * 10) / 1e6)


def test_pipeline_run_writes_a_trace(tmp_path, monkeypatch, fake_llm, sync_pool):
    wd = tmp_path / "run"
    run_pipeline(wd, monkeypatch)
    records = llm.read_trace(wd / "trace.jsonl")
    assert len(records) == len(fake_llm.requests)
    assert {r["purpose"] for r in records} == {"extract", "label"}
    assert all(r["stop_reason"] == "tool_use" and r["cost_usd"] > 0 for r in records)
