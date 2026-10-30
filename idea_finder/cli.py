"""
Idea Finder: turn bad product reviews into a ranked list of unmet customer needs.

Pipeline (each stage saves its output in --workdir, so you can re-run any step):
  1. collect   Pull 1-3 star reviews for a category (Amazon Reviews 2023 dataset),
               or load your own CSV/JSONL of reviews or comments.
  2. extract   An LLM reads each review and pulls out the real problem(s),
               skipping shipping/seller/price noise.
  3. cluster   Embed the problem statements and group similar ones.
  4. label     An LLM names each group, explains the underlying need and
               suggests product directions.
  5. report    Score and rank the groups, write report.md and clusters.csv.
  6. sample    Write a random sample of extractions to a CSV you grade by hand
               (your eval set for the extraction step).

Setup:
  pip install anthropic numpy pandas scikit-learn huggingface_hub
  pip install sentence-transformers     # optional: better embeddings (else TF-IDF)
  export ANTHROPIC_API_KEY=sk-...

Examples:
  # small test run first (cheap), then scale up
  python idea_finder.py all --category Baby_Products --keyword "diaper bag" --n 200
  python idea_finder.py all --category Home_and_Kitchen --keyword lunch --n 2000

  # your own data: CSV/JSONL with a "text" column (optional: rating, title,
  # product, timestamp, helpful_vote)
  python idea_finder.py all --input my_reviews.csv --workdir runs/my_data

  # re-run only the later steps after tweaking settings
  python idea_finder.py cluster --workdir runs/baby --threshold 0.5
  python idea_finder.py label report --workdir runs/baby
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

from .config import CATEGORIES, STAGES
from .pipeline import (stage_cluster, stage_collect, stage_extract, stage_label,
                       stage_report, stage_sample)
from .utils import log


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None, prog: str | None = None) -> None:
    ap = argparse.ArgumentParser(prog=prog, description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stages", nargs="+", choices=STAGES + ["all"],
                    help="Which stage(s) to run. 'all' = collect extract cluster label report")
    ap.add_argument("--workdir", default=None, help="Folder for this run (default: runs/<category or input name>)")
    g = ap.add_argument_group("collect")
    g.add_argument("--category", help=f"Amazon category, e.g. Baby_Products. Known: {', '.join(CATEGORIES)}")
    g.add_argument("--input", help="Your own CSV/JSONL with a 'text' column instead of Amazon data")
    g.add_argument("--keyword", help="Only keep reviews mentioning any of these (comma-separated)")
    g.add_argument("--n", type=int, default=1000, help="How many reviews to keep (default 1000)")
    g.add_argument("--max-rating", type=float, default=3, help="Keep reviews at or below this rating (default 3)")
    g.add_argument("--min-chars", type=int, default=60, help="Skip reviews shorter than this")
    g.add_argument("--scan", type=int, default=1_000_000,
                   help="Max lines to scan from the Amazon file (0 = whole file; default 1,000,000)")
    g = ap.add_argument_group("extract / label")
    g.add_argument("--batch", type=int, default=10, help="Reviews per extraction call (default 10)")
    g.add_argument("--workers", type=int, default=4, help="Parallel API calls (default 4)")
    g.add_argument("--limit", type=int, default=0, help="Only extract the first N reviews (test runs)")
    g.add_argument("--max-clusters", type=int, default=25, help="Label at most this many clusters")
    g = ap.add_argument_group("cluster")
    g.add_argument("--embedder", choices=["auto", "st", "tfidf"], default="auto")
    g.add_argument("--threshold", type=float, default=None,
                   help="Cosine distance cutoff (default 0.45 sentence-transformers, 0.75 TF-IDF)")
    g.add_argument("--min-size", type=int, default=5, help="Smallest group to keep (default 5)")
    g = ap.add_argument_group("other")
    g.add_argument("--eval-n", type=int, default=50, help="Rows in eval_sample.csv")
    g.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    stages = ["collect", "extract", "cluster", "label", "report"] if "all" in args.stages else args.stages
    name = args.category or (Path(args.input).stem if args.input else "run")
    if args.keyword and not args.workdir:
        name += "_" + re.sub(r"[^a-z0-9]+", "-", args.keyword.lower()).strip("-")
    wd = Path(args.workdir or f"runs/{name}")
    wd.mkdir(parents=True, exist_ok=True)

    needs_llm = {"extract", "label"} & set(stages)
    if needs_llm and not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY first (needed for: " + ", ".join(sorted(needs_llm)) + ")")

    funcs = {"collect": stage_collect, "extract": stage_extract, "cluster": stage_cluster,
             "label": stage_label, "report": stage_report, "sample": stage_sample}
    for s in STAGES:
        if s in stages:
            t = time.time()
            log(f"== {s} ==")
            funcs[s](args, wd)
            log(f"   {s} took {time.time() - t:.0f}s")
