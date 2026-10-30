"""Small helpers shared by the pipeline, the store and the CLI."""

from __future__ import annotations

import html
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def clean_text(s) -> str:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    s = html.unescape(str(s))
    s = re.sub(r"<br\s*/?>", " ", s, flags=re.I)
    s = re.sub(r"<[^>]+>", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def blank(x) -> bool:
    return x is None or (isinstance(x, float) and np.isnan(x)) or (isinstance(x, str) and not x.strip())


def short(s: str, n: int = 280) -> str:
    return s if len(s) <= n else s[: n - 1].rsplit(" ", 1)[0] + "…"


def require(path: Path, stage: str) -> None:
    if not path.exists():
        sys.exit(f"Missing {path.name}. Run the '{stage}' stage first.")
