"""The one place that talks to the model API."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# LLM call (forced tool use = reliable structured JSON)
# ---------------------------------------------------------------------------

_client = None


def call_tool(model: str, system: str, user: str, tool: dict, max_tokens: int = 4096) -> dict:
    """Ask the model to answer by calling `tool`; return the tool input as a dict."""
    global _client
    if _client is None:
        from anthropic import Anthropic
        _client = Anthropic(max_retries=6)
    resp = _client.messages.create(
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
