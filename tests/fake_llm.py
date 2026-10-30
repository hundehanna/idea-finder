"""A fake Anthropic client for tests. It never touches the network.

It has the same shape as the real client: fake.messages.create(**kwargs)
returns a real anthropic.types.Message. It answers in one of two ways:

1. Scripted: give it a list of responses, and it returns them in order.
   (Used for agent tests: "first call search_reviews, then submit_brief".)
2. Rule-based: for the pipeline tools (record_problems, describe_cluster)
   it builds an answer from keywords in the prompt, so the output is
   the same every run.

Every request is saved in fake.requests so tests can check what was sent.
"""

from __future__ import annotations

import json
import re
import threading
from typing import Any, Callable, Iterable

from anthropic.types import Message

# Keyword -> problem statement, plus an optional workaround keyword.
# These match the phrasings in tests/fixtures/make_fixtures.py.
PROBLEM_RULES: list[tuple[str, str, str | None, str]] = [
    # (problem regex, problem statement, workaround regex, workaround text)
    (r"leak", "Lid leaks when the mug tips over in a bag", r"plastic bag", "Keeps the mug inside a plastic bag"),
    (r"\bcold\b|lukewarm", "Drink goes cold within an hour", None, ""),
    (r"cup holder", "Mug does not fit car cup holders", r"towel", "Wedges it in with a towel"),
    (r"smell", "Mug keeps a smell after washing", None, ""),
    (r"dishwasher", "Mug is not dishwasher safe", None, ""),
    (r"batter", "Batteries run out after a short play time", r"rechargeable", "Buys rechargeable batteries in bulk"),
    (r"\bapp\b", "Companion app will not connect to the robot", None, ""),
    (r"snapped|broke", "Arms and legs snap off easily", None, ""),
    (r"\bloud\b", "Robot is too loud for quiet times", r"tape", "Puts tape over the speaker"),
    (r"\bthe instructions\b", "Instructions are hard to follow", None, ""),
    (r"flicker", "Light flickers", None, ""),
    (r"clamp", "Clamp does not fit thick desks", r"block of wood", "Uses a block of wood as a spacer"),
    (r"too hot to touch|hot enough to worry", "Lamp head gets too hot", None, ""),
    (r"too dim", "Light is not bright enough for work", None, ""),
    (r"cable", "Power cable is too short", r"extension cord", "Bought an extension cord"),
]

WHO_RULES = [(r"\bson\b|daughter|year old|baby", "Parent of a young child")]
CONTEXT_RULES = [(r"commute|train|\bcar\b|bike", "Commuting"),
                 (r"home office|work from home", "Working from home")]

REVIEW_RE = re.compile(r'<review id="(\d+)"[^>]*>\n(.*?)\n</review>', re.S)


class FakeBadRequest(Exception):
    """Raised the way the real API would return a 400 error."""


def make_message(model: str, content: list[dict], stop_reason: str,
                 input_tokens: int, output_tokens: int, msg_id: str = "msg_fake",
                 cache_write: int = 0, cache_read: int = 0) -> Message:
    """Build a real SDK Message object from plain dicts."""
    return Message.model_validate({
        "id": msg_id, "type": "message", "role": "assistant", "model": model,
        "content": content, "stop_reason": stop_reason, "stop_sequence": None,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens,
                  "cache_creation_input_tokens": cache_write, "cache_read_input_tokens": cache_read},
    })


def _first_match(rules: list[tuple[str, str]], text: str) -> str:
    for pattern, value in rules:
        if re.search(pattern, text, re.I):
            return value
    return ""


def extract_answer(user: str) -> dict:
    """Rule-based answer for the record_problems tool."""
    results = []
    for rid, body in REVIEW_RE.findall(user):
        problems = []
        for pattern, problem, wa_pattern, wa_text in PROBLEM_RULES:
            if re.search(pattern, body, re.I):
                has_wa = bool(wa_pattern and re.search(wa_pattern, body, re.I))
                problems.append({
                    "problem": problem,
                    "context": _first_match(CONTEXT_RULES, body),
                    "who": _first_match(WHO_RULES, body),
                    "workaround": wa_text if has_wa else "",
                    "wish": "",
                })
        results.append({"review_id": int(rid), "noise": not problems, "problems": problems})
    return {"results": results}


def label_answer(user: str) -> dict:
    """Rule-based answer for the describe_cluster tool."""
    lines = [ln[2:] for ln in user.splitlines() if ln.startswith("- ")]
    first = lines[0].split(" (")[0] if lines else "Unknown problem"
    workarounds = sorted({m for ln in lines for m in re.findall(r"workaround: ([^;)]+)", ln)})
    defect = re.search(r"snap|break|broke|peel", first, re.I)
    return {
        "name": " ".join(first.split()[:8]),
        "underlying_need": f"People need a product where this does not happen: {first.lower()}.",
        "who_and_when": "People who wrote these reviews",
        "current_workarounds": workarounds,
        "why_products_fail": "Current products are not designed for this situation.",
        "product_directions": [f"Redesign so that this is fixed: {first.lower()}", "Test a prototype with real users"],
        "opportunity_type": "quality_defect" if defect else "design_gap",
    }


RULE_HANDLERS: dict[str, Callable[[str], dict]] = {
    "record_problems": extract_answer,
    "describe_cluster": label_answer,
}


class _Messages:
    def __init__(self, owner: "FakeLLM") -> None:
        self._owner = owner

    def create(self, **kwargs: Any) -> Message:
        return self._owner._create(kwargs)


# Like the real API (checked 2026-10-29): these models return a 400 for a forced tool_choice.
REJECTS_FORCED_TOOL_CHOICE = ("claude-sonnet-5-5", "claude-opus-5-5")


class FakeLLM:
    """Stand-in for anthropic.Anthropic()."""

    def __init__(self, script: Iterable[Message | Callable[[dict], Message]] | None = None,
                 reject_forced_tool_choice_for: Iterable[str] = REJECTS_FORCED_TOOL_CHOICE) -> None:
        self.messages = _Messages(self)
        self.requests: list[dict] = []
        self._script = list(script or [])
        self._reject_forced = set(reject_forced_tool_choice_for)
        self._lock = threading.Lock()
        self._count = 0

    def _create(self, kwargs: dict) -> Message:
        with self._lock:
            self._count += 1
            n = self._count
            self.requests.append(kwargs)
            step = self._script.pop(0) if self._script else None

        model = kwargs["model"]
        choice = kwargs.get("tool_choice") or {}
        if choice.get("type") in ("tool", "any") and model in self._reject_forced:
            raise FakeBadRequest(f"tool_choice: type \"{choice['type']}\" is not supported for this model.")

        if step is not None:
            return step(kwargs) if callable(step) else step

        tools = kwargs.get("tools") or []
        name = choice.get("name") or (tools[0]["name"] if tools else None)
        if name not in RULE_HANDLERS:
            raise AssertionError(f"FakeLLM has no script left and no rule for tool {name!r}")
        user = kwargs["messages"][-1]["content"]
        answer = RULE_HANDLERS[name](user)
        prompt_chars = len(str(kwargs.get("system", ""))) + len(user) + len(json.dumps(tools))
        return make_message(
            model,
            [{"type": "tool_use", "id": f"toolu_fake_{n:04d}", "name": name, "input": answer}],
            stop_reason="tool_use",
            input_tokens=prompt_chars // 4,
            output_tokens=len(json.dumps(answer)) // 4 + 10,
            msg_id=f"msg_fake_{n:04d}",
        )
