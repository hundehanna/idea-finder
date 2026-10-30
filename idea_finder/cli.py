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

from . import llm, store
from .config import CATEGORIES, DB_PATH, PREPARE_CATEGORIES, STAGES
from .pipeline import (prepare_category, stage_cluster, stage_collect, stage_extract, stage_label,
                       stage_report, stage_sample)
from .utils import log


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None, prog: str | None = None) -> None:
    """Entry point. 'prepare' is its own command; anything else runs pipeline stages as before."""
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] == "prepare":
        prepare_main(argv[1:], prog)
    else:
        run_stages(argv, prog)


def prepare_main(argv: list[str], prog: str | None = None) -> None:
    """Build the local review database. Reads the dataset only; makes no API calls."""
    ap = argparse.ArgumentParser(
        prog=f"{prog or Path(sys.argv[0]).name} prepare",
        description="Build the local review database used by the agent tools. Streams Amazon "
                    "Reviews 2023 from Hugging Face (or reads local files). No API cost.")
    ap.add_argument("--categories", default=",".join(PREPARE_CATEGORIES),
                    help="Comma-separated categories (default: %(default)s)")
    ap.add_argument("--per-category", type=int, default=5000, help="Reviews to keep per category (default 5000)")
    ap.add_argument("--scan", type=int, default=300_000,
                    help="Lines to read per category (0 = whole file; default 300,000)")
    ap.add_argument("--max-rating", type=float, default=3, help="Keep reviews at or below this rating (default 3)")
    ap.add_argument("--min-chars", type=int, default=60, help="Skip reviews shorter than this (default 60)")
    ap.add_argument("--db", default=str(DB_PATH), help="Database file (default: data/ideas.db)")
    ap.add_argument("--source-dir", default=None,
                    help="Read <dir>/<Category>.jsonl instead of streaming from Hugging Face")
    ap.add_argument("--force", action="store_true", help="Rebuild categories that are already in the database")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    conn = store.connect(Path(args.db))
    search = "FTS5" if store.fts_enabled(conn) else "LIKE fallback (FTS5 not available)"
    log(f"Database: {args.db} (keyword search: {search})")
    have = {c["category"]: c["reviews"] for c in store.category_counts(conn)}
    for category in [c.strip() for c in args.categories.split(",") if c.strip()]:
        if category in have and not args.force:
            log(f"{category}: already in the database ({have[category]:,} reviews). Use --force to rebuild it.")
            continue
        if category not in CATEGORIES and not args.source_dir:
            log(f"Note: '{category}' is not in the known list; trying anyway.")
        prepare_category(conn, category, args.per_category, args.scan, args.max_rating, args.min_chars,
                         args.seed, Path(args.source_dir) if args.source_dir else None,
                         replace=category in have)
    print_counts(conn)
    conn.close()


def print_counts(conn) -> None:
    """Print reviews per category as a small table."""
    print(f"\n{'Category':<24}{'Reviews':>9}{'Products':>10}{'Avg rating':>12}")
    for r in store.category_counts(conn):
        print(f"{r['category']:<24}{r['reviews']:>9,}{r['products']:>10,}{r['avg_rating']:>12.2f}")


def run_stages(argv=None, prog: str | None = None) -> None:
    """The original pipeline command: run one or more stages in a run folder."""
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
    llm.start_run(wd)  # every model call in this run is logged to <wd>/trace.jsonl

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
