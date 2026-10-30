"""The one place that talks to the model API.

Every model call goes through call(). For each call it appends one line to
<run folder>/trace.jsonl with the tokens, time, stop reason, estimated cost
and the full response. That gives us one place for logging, cost control and
(later) replaying a saved run without the network.

Tests replace _client with a fake, so they never reach the real API.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import LONG_PROMPT_TOKENS, PRICES
from .utils import log

_client = None
_trace_path: Path | None = None
_run_id = ""
_call_count = 0
_lock = threading.Lock()  # extraction calls the model from several threads
_warned_models: set[str] = set()


def get_client():
    """Create the real Anthropic client the first time it is needed."""
    global _client
    if _client is None:
        from anthropic import Anthropic
        _client = Anthropic(max_retries=6)  # the SDK retries rate limits and server errors
    return _client


def start_run(run_dir: Path) -> None:
    """Write the trace for all following calls to run_dir/trace.jsonl."""
    global _trace_path, _run_id, _call_count
    run_dir.mkdir(parents=True, exist_ok=True)
    _trace_path = run_dir / "trace.jsonl"
    _run_id = run_dir.name
    _call_count = len(read_trace(_trace_path))  # a rerun in the same folder keeps counting


def read_trace(path: Path) -> list[dict]:
    """All records in a trace file (an empty list if there is no file yet)."""
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def estimate_cost(model: str, usage: Any) -> float | None:
    """Estimated cost in US dollars for one call, or None if we have no price for the model."""
    price = PRICES.get(model)
    if price is None:
        if model not in _warned_models:
            _warned_models.add(model)
            log(f"Warning: no price for model {model!r}; its cost will show as unknown.")
        return None
    fresh = usage.input_tokens or 0
    written = usage.cache_creation_input_tokens or 0
    read = usage.cache_read_input_tokens or 0
    if "long_prompt" in price and fresh + written + read > LONG_PROMPT_TOKENS:
        price = price["long_prompt"]
    dollars = (fresh * price["input"] + written * price["cache_write"]
               + read * price["cache_read"] + (usage.output_tokens or 0) * price["output"])
    return dollars / 1_000_000


def _request_hash(request: dict) -> str:
    """Short fingerprint of a request. Replay mode uses it to check it sends the same calls."""
    text = json.dumps(request, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _write_trace(record: dict) -> None:
    """Give the record a call number and append it to the trace file."""
    global _call_count
    with _lock:
        _call_count += 1
        record = {"call_id": _call_count, **record}
        if _trace_path is not None:  # the CLI always sets a run folder
            with _trace_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


def call(*, purpose: str, model: str, messages: list[dict], system: str | list | None = None,
         tools: list[dict] | None = None, tool_choice: dict | None = None,
         max_tokens: int = 4096) -> Any:
    """Send one request to the model, log it to the trace, and return the response."""
    request: dict[str, Any] = {"model": model, "max_tokens": max_tokens, "messages": messages}
    if system is not None:
        request["system"] = system
    if tools:
        request["tools"] = tools
    if tool_choice:
        request["tool_choice"] = tool_choice

    base = {"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "run_id": _run_id, "purpose": purpose, "model": model,
            "request_hash": _request_hash(request)}
    start = time.perf_counter()
    try:
        resp = get_client().messages.create(**request)
    except Exception as e:
        _write_trace({**base, "latency_s": round(time.perf_counter() - start, 3),
                      "error": f"{type(e).__name__}: {e}"})
        raise
    usage = resp.usage
    cost = estimate_cost(model, usage)
    _write_trace({
        **base,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_creation_input_tokens": usage.cache_creation_input_tokens or 0,
        "cache_read_input_tokens": usage.cache_read_input_tokens or 0,
        "latency_s": round(time.perf_counter() - start, 3),
        "stop_reason": resp.stop_reason,
        "cost_usd": None if cost is None else round(cost, 6),
        "response": resp.to_dict(),
    })
    return resp


def trace_totals(path: Path) -> dict:
    """Add up calls, errors, tokens, time and cost in a trace file."""
    records = read_trace(path)
    keys = ["input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"]
    totals = {"calls": len(records), "errors": sum(1 for r in records if "error" in r)}
    for k in keys:
        totals[k] = sum(r.get(k) or 0 for r in records)
    totals["latency_s"] = round(sum(r.get("latency_s") or 0 for r in records), 3)
    totals["cost_usd"] = round(sum(r.get("cost_usd") or 0 for r in records), 6)
    totals["unpriced_calls"] = sum(1 for r in records if "error" not in r and r.get("cost_usd") is None)
    return totals


def call_tool(model: str, system: str, user: str, tool: dict, max_tokens: int = 4096,
              purpose: str = "tool") -> dict:
    """Ask the model to answer by calling `tool`; return the tool input as a dict."""
    resp = call(
        purpose=purpose,
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
        tools=[tool],
        tool_choice={"type": "tool", "name": tool["name"]},
    )
    for block in resp.content:
        if block.type == "tool_use":
            return block.input
    raise RuntimeError(f"No tool call in response (stop_reason={resp.stop_reason})")
