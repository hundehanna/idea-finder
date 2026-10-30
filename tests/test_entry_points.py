"""The old command (python idea_finder.py) and the new one (python -m idea_finder) both work.

These run real subprocesses, the way a user types the commands.
Only 'collect' and '--help' are used here: neither calls the API.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "tests" / "golden"
INPUT_DIR = ROOT / "tests" / "fixtures" / "input"


def run(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    env = {**os.environ, "COLUMNS": "100", "PYTHONIOENCODING": "utf-8"}
    return subprocess.run([sys.executable, *args], cwd=cwd, env=env, capture_output=True,
                          text=True, encoding="utf-8")


def after_usage(text: str) -> list[str]:
    """Help text without the usage block (its wrapping depends on the program name)."""
    return text.split("\n\n", 1)[1].splitlines()


def test_old_script_help_is_unchanged():
    res = run(["idea_finder.py", "--help"], cwd=ROOT)
    assert res.returncode == 0, res.stderr
    assert res.stdout.splitlines() == (GOLDEN / "help.txt").read_text(encoding="utf-8").splitlines()


def test_module_help_has_same_content():
    res = run(["-m", "idea_finder", "--help"], cwd=ROOT)
    assert res.returncode == 0, res.stderr
    assert res.stdout.startswith("usage: python -m idea_finder")
    assert after_usage(res.stdout) == after_usage((GOLDEN / "help.txt").read_text(encoding="utf-8"))


def test_old_script_collect_gives_golden_reviews(tmp_path):
    res = run([str(ROOT / "idea_finder.py"), "collect", "--input", "reviews.csv",
               "--workdir", str(tmp_path)], cwd=INPUT_DIR)
    assert res.returncode == 0, res.stderr
    got = (tmp_path / "reviews.jsonl").read_text(encoding="utf-8").splitlines()
    assert got == (GOLDEN / "reviews.jsonl").read_text(encoding="utf-8").splitlines()
