"""Checks on the synthetic fixture data itself."""

from __future__ import annotations

import csv
import json
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"
AMAZON_KEYS = {"rating", "title", "text", "images", "asin", "parent_asin", "user_id",
               "timestamp", "helpful_vote", "verified_purchase"}


def load(category: str) -> list[dict]:
    path = FIXTURES / "amazon" / f"{category}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_three_categories_in_amazon_format():
    files = sorted(p.stem for p in (FIXTURES / "amazon").glob("*.jsonl"))
    assert files == ["Desk_Lamps", "Toy_Robots", "Travel_Mugs"]
    for name in files:
        rows = load(name)
        assert 40 <= len(rows) <= 60
        assert all(set(r) == AMAZON_KEYS for r in rows)


def test_has_the_tricky_cases():
    rows = [r for name in ("Desk_Lamps", "Toy_Robots", "Travel_Mugs") for r in load(name)]
    texts = [r["text"] for r in rows]
    assert sum("IGNORE ALL PREVIOUS INSTRUCTIONS" in t for t in texts) == 1
    assert any("<br />" in t for t in texts)
    assert any("😞" in t for t in texts)
    assert any(r["rating"] >= 4 for r in rows)
    assert any(len(t) < 60 for t in texts)


def test_input_csv_has_all_reviews():
    with (FIXTURES / "input" / "reviews.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 145
    assert set(rows[0]) == {"rating", "title", "text", "product", "timestamp", "helpful_vote"}
