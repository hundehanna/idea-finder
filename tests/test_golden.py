"""Golden-file test: the pipeline must keep producing exactly the same output.

The golden files in tests/golden/ were made by the original single-file script.
If this test fails after a refactor, the refactor changed behavior.

To rebuild the golden files on purpose (only when a behavior change is intended):
    set UPDATE_GOLDEN=1 && pytest tests/test_golden.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import pytest

from idea_finder.cli import main

TESTS = Path(__file__).parent
INPUT_DIR = TESTS / "fixtures" / "input"
GOLDEN = TESTS / "golden"
UPDATE = os.environ.get("UPDATE_GOLDEN") == "1"

COMPARED = ["reviews.jsonl", "meta.json", "extractions.jsonl", "problems.jsonl",
            "problems_clustered.jsonl", "labels.json", "report.md", "clusters.csv",
            "eval_sample.csv"]


def normalize(name: str, text: str) -> list[str]:
    """Drop the parts that change every run (timestamps) and ignore line endings."""
    if name == "meta.json":
        data = json.loads(text)
        data.pop("collected_at", None)
        return json.dumps(data, indent=2).splitlines()
    lines = text.splitlines()
    if name == "report.md":
        lines = [ln for ln in lines if not ln.startswith("Generated:")]
    return lines


def run_pipeline(workdir: Path, monkeypatch) -> None:
    """Run all stages on the fixture CSV, the same way a user would."""
    monkeypatch.chdir(INPUT_DIR)  # so the recorded source is just "reviews.csv"
    main(["all", "--input", "reviews.csv", "--workdir", str(workdir),
          "--embedder", "tfidf", "--workers", "1"])
    main(["sample", "--workdir", str(workdir), "--eval-n", "10"])


def test_pipeline_output_matches_golden(tmp_path, monkeypatch, fake_llm, sync_pool):
    wd = tmp_path / "run"
    run_pipeline(wd, monkeypatch)
    if UPDATE:
        GOLDEN.mkdir(exist_ok=True)
        for name in COMPARED:
            shutil.copyfile(wd / name, GOLDEN / name)
    for name in COMPARED:
        got = normalize(name, (wd / name).read_text(encoding="utf-8-sig"))
        want = normalize(name, (GOLDEN / name).read_text(encoding="utf-8-sig"))
        assert got == want, f"{name} differs from the golden file"


def test_pipeline_made_real_clusters(tmp_path, monkeypatch, fake_llm, sync_pool):
    """Sanity check on the fixture itself: the golden run has several labeled groups."""
    wd = tmp_path / "run"
    run_pipeline(wd, monkeypatch)
    labels = json.loads((wd / "labels.json").read_text(encoding="utf-8"))
    assert len(labels) >= 5
    report = (wd / "report.md").read_text(encoding="utf-8")
    assert report.count("\n## ") == len(labels)


def test_help_matches_golden(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["idea_finder.py"])
    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    if UPDATE:
        (GOLDEN / "help.txt").write_text(out, encoding="utf-8")
    assert out.splitlines() == (GOLDEN / "help.txt").read_text(encoding="utf-8").splitlines()


def test_collect_jsonl_input_with_timestamp(tmp_path):
    """JSONL input with a 'timestamp' column used to crash (pandas turned it into dates)."""
    path = tmp_path / "in.jsonl"
    row = {"text": "The lid leaks every time it tips over in my bag on the way to work.",
           "rating": 1, "timestamp": 1588687728923, "product": "B01", "helpful_vote": 2}
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    main(["collect", "--input", str(path), "--workdir", str(tmp_path / "run")])
    saved = json.loads((tmp_path / "run" / "reviews.jsonl").read_text(encoding="utf-8"))
    assert saved["timestamp"] == 1588687728923   # still a plain number, as in the CSV path
    assert saved["product"] == "B01" and saved["helpful_vote"] == 2


def test_collect_gives_the_same_reviews_from_jsonl_and_csv(tmp_path):
    """The fixture CSV, converted to JSONL, collects to exactly the same reviews."""
    import pandas as pd
    df = pd.read_csv(INPUT_DIR / "reviews.csv", encoding="utf-8")
    jsonl = tmp_path / "reviews.jsonl"
    df.to_json(jsonl, orient="records", lines=True, force_ascii=False)
    main(["collect", "--input", str(INPUT_DIR / "reviews.csv"), "--workdir", str(tmp_path / "csv")])
    main(["collect", "--input", str(jsonl), "--workdir", str(tmp_path / "jsonl")])
    from_csv = (tmp_path / "csv" / "reviews.jsonl").read_text(encoding="utf-8").splitlines()
    from_jsonl = (tmp_path / "jsonl" / "reviews.jsonl").read_text(encoding="utf-8").splitlines()
    assert from_jsonl == from_csv
