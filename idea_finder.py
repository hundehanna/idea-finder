#!/usr/bin/env python3
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
import hashlib
import html
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

EXTRACT_MODEL = os.environ.get("EXTRACT_MODEL", "claude-haiku-5-5")
LABEL_MODEL = os.environ.get("LABEL_MODEL", "claude-sonnet-5-5")

HF_REPO = "McAuley-Lab/Amazon-Reviews-2023"
HF_PATH = "datasets/{repo}/raw/review_categories/{category}.jsonl"

# Category names in the Amazon Reviews 2023 dataset (data covers 1996 to Sep 2023).
CATEGORIES = [
    "All_Beauty", "Amazon_Fashion", "Appliances", "Arts_Crafts_and_Sewing",
    "Automotive", "Baby_Products", "Beauty_and_Personal_Care", "Books",
    "Cell_Phones_and_Accessories", "Clothing_Shoes_and_Jewelry", "Electronics",
    "Grocery_and_Gourmet_Food", "Handmade_Products", "Health_and_Household",
    "Home_and_Kitchen", "Industrial_and_Scientific", "Office_Products",
    "Patio_Lawn_and_Garden", "Pet_Supplies", "Sports_and_Outdoors",
    "Tools_and_Home_Improvement", "Toys_and_Games", "Video_Games",
]

STAGES = ["collect", "extract", "cluster", "label", "report", "sample"]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Stage 1: collect
# ---------------------------------------------------------------------------

def stage_collect(args, wd: Path) -> None:
    out = wd / "reviews.jsonl"
    keywords = [k.strip().lower() for k in (args.keyword or "").split(",") if k.strip()]
    rng = random.Random(args.seed)

    def keep(text: str, title: str, rating) -> bool:
        if len(text) < args.min_chars:
            return False
        if rating is not None and rating > args.max_rating:
            return False
        if keywords:
            hay = f"{title} {text}".lower()
            return any(k in hay for k in keywords)
        return True

    rows: list[dict] = []
    if args.input:
        path = Path(args.input)
        if path.suffix.lower() in (".jsonl", ".json"):
            df = pd.read_json(path, lines=path.suffix.lower() == ".jsonl")
        else:
            df = pd.read_csv(path, encoding="utf-8-sig")
        if "text" not in df.columns:
            sys.exit(f"{path} needs a 'text' column. Found: {list(df.columns)}")
        for _, r in df.iterrows():
            text = clean_text(r.get("text"))
            title = clean_text(r.get("title", ""))
            rating = r.get("rating")
            rating = None if rating is None or pd.isna(rating) else float(rating)
            if keep(text, title, rating):
                ts, hv, prod = r.get("timestamp"), r.get("helpful_vote"), r.get("product")
                rows.append({
                    "rating": rating, "title": title, "text": text,
                    "product": "" if blank(prod) else str(prod),
                    "timestamp": None if blank(ts) else (ts.item() if hasattr(ts, "item") else ts),
                    "helpful_vote": 0 if blank(hv) else int(hv),
                })
        rng.shuffle(rows)
        rows = rows[: args.n]
        source = str(path)
    else:
        if not args.category:
            sys.exit("Give --category (e.g. Baby_Products) or --input your_file.csv")
        if args.category not in CATEGORIES:
            log(f"Note: '{args.category}' is not in the known list; trying anyway.")
        from huggingface_hub import HfFileSystem
        fs = HfFileSystem()
        hf_path = HF_PATH.format(repo=HF_REPO, category=args.category)
        log(f"Streaming {hf_path} (scan limit: {args.scan or 'whole file'} lines)")
        seen = 0  # reservoir sampling over matching reviews
        with fs.open(hf_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if args.scan and i >= args.scan:
                    break
                if i and i % 200_000 == 0:
                    log(f"  scanned {i:,} lines, {seen:,} matching")
                r = json.loads(line)
                text, title = clean_text(r.get("text")), clean_text(r.get("title"))
                if not keep(text, title, r.get("rating")):
                    continue
                row = {
                    "rating": r.get("rating"), "title": title, "text": text,
                    "product": r.get("parent_asin") or r.get("asin") or "",
                    "timestamp": r.get("timestamp"),
                    "helpful_vote": r.get("helpful_vote", 0) or 0,
                }
                seen += 1
                if len(rows) < args.n:
                    rows.append(row)
                else:
                    j = rng.randrange(seen)
                    if j < args.n:
                        rows[j] = row
        source = f"Amazon Reviews 2023 / {args.category}"

    for i, r in enumerate(rows):
        r["id"] = i
    write_jsonl(out, rows)
    meta = {"source": source, "keywords": keywords, "max_rating": args.max_rating,
            "n_reviews": len(rows), "collected_at": datetime.now().isoformat(timespec="seconds")}
    (wd / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log(f"Saved {len(rows)} reviews to {out}")
    if len(rows) < 50:
        log("Warning: fewer than 50 reviews. Widen the keyword, raise --scan, or pick another category.")


# ---------------------------------------------------------------------------
# Stage 2: extract problems
# ---------------------------------------------------------------------------

EXTRACT_SYSTEM = """You read customer reviews to find unmet needs that a new product could solve.

For each review, list the real problems the customer ran into with the product or with the situation they use it in.
Rules:
- Write each problem as a general need in plain words, without brand names. Good: "Bag stays wet and starts to smell after beach trips". Bad: "The Acme bag is junk".
- One problem per item. A review can have zero, one or several.
- Only use what the review says. Do not guess or add problems.
- Skip complaints that are only about shipping, the seller, a late or wrong order, counterfeits, or price alone. If nothing else is left, return no problems and set noise=true.
- context: when or where it happens (e.g. "at the beach with kids"), if stated.
- who: who has the problem (e.g. "parent of a toddler"), if stated.
- workaround: anything the customer did to cope (e.g. "lines it with a trash bag"), if stated.
- wish: what they wanted instead, if stated.
Use an empty string for anything not stated."""

EXTRACT_TOOL = {
    "name": "record_problems",
    "description": "Record the problems found in each review.",
    "input_schema": {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "review_id": {"type": "integer"},
                        "noise": {"type": "boolean"},
                        "problems": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "problem": {"type": "string"},
                                    "context": {"type": "string"},
                                    "who": {"type": "string"},
                                    "workaround": {"type": "string"},
                                    "wish": {"type": "string"},
                                },
                                "required": ["problem", "context", "who", "workaround", "wish"],
                            },
                        },
                    },
                    "required": ["review_id", "noise", "problems"],
                },
            }
        },
        "required": ["results"],
    },
}


def _extract_batch(batch: list[dict]) -> list[dict]:
    parts = []
    for r in batch:
        stars = f"{r['rating']:.0f} stars" if r.get("rating") is not None else "no rating"
        parts.append(f"<review id=\"{r['id']}\" rating=\"{stars}\">\n"
                     f"Title: {r.get('title', '')}\n{short(r['text'], 2500)}\n</review>")
    user = "Reviews:\n\n" + "\n\n".join(parts)
    out = call_tool(EXTRACT_MODEL, EXTRACT_SYSTEM, user, EXTRACT_TOOL)
    valid = {r["id"] for r in batch}
    results, seen = [], set()
    for x in out.get("results", []):
        rid = x.get("review_id")
        if rid in valid and rid not in seen:  # ignore made-up or duplicate ids
            seen.add(rid)
            results.append(x)
    # Reviews the model skipped are not saved, so the next 'extract' run retries them.
    return results


def stage_extract(args, wd: Path) -> None:
    require(wd / "reviews.jsonl", "collect")
    reviews = read_jsonl(wd / "reviews.jsonl")
    if args.limit:
        reviews = reviews[: args.limit]
    out = wd / "extractions.jsonl"
    done = {x["review_id"] for x in read_jsonl(out)}
    todo = [r for r in reviews if r["id"] not in done]
    if not todo:
        log("All reviews already extracted.")
    else:
        batches = [todo[i: i + args.batch] for i in range(0, len(todo), args.batch)]
        log(f"Extracting {len(todo)} reviews in {len(batches)} calls to {EXTRACT_MODEL} "
            f"({len(done)} already done)")
        failed = 0
        with out.open("a", encoding="utf-8") as f, ThreadPoolExecutor(args.workers) as pool:
            futures = {pool.submit(_extract_batch, b): b for b in batches}
            for k, fut in enumerate(as_completed(futures), 1):
                try:
                    for res in fut.result():
                        f.write(json.dumps(res, ensure_ascii=False) + "\n")
                    f.flush()
                except Exception as e:  # keep going; failed batches are retried on re-run
                    failed += 1
                    log(f"  batch failed: {e}")
                if k % 10 == 0 or k == len(batches):
                    log(f"  {k}/{len(batches)} calls done")
        if failed:
            log(f"{failed} batches failed. Run 'extract' again to retry just those.")
        skipped = len(todo) - len({x["review_id"] for x in read_jsonl(out)} - done)
        if skipped > 0:
            log(f"{skipped} reviews were skipped by the model. Run 'extract' again to retry them.")

    # Flatten to one row per problem
    by_id = {r["id"]: r for r in reviews}
    problems = []
    for x in read_jsonl(out):
        r = by_id.get(x["review_id"])
        if r is None:
            continue
        for p in x.get("problems", []):
            if not p.get("problem", "").strip():
                continue
            problems.append({
                "pid": len(problems), "review_id": r["id"], "rating": r.get("rating"),
                "product": r.get("product", ""), "timestamp": r.get("timestamp"),
                "helpful_vote": r.get("helpful_vote", 0), **{k: p.get(k, "") for k in
                ("problem", "context", "who", "workaround", "wish")},
            })
    write_jsonl(wd / "problems.jsonl", problems)
    noise = sum(1 for x in read_jsonl(out) if x.get("noise"))
    log(f"{len(problems)} problems from {len(by_id)} reviews ({noise} reviews were only noise)")


# ---------------------------------------------------------------------------
# Stage 3: cluster
# ---------------------------------------------------------------------------

def embed(texts: list[str], how: str) -> tuple[np.ndarray, str]:
    if how in ("auto", "st"):
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2"))
            vecs = model.encode(texts, batch_size=64, show_progress_bar=len(texts) > 500,
                                normalize_embeddings=True)
            return np.asarray(vecs), "st"
        except Exception as e:
            if how == "st":
                raise
            log(f"sentence-transformers not available ({type(e).__name__}); using TF-IDF.")
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize
    tf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    X = tf.fit_transform(texts)
    k = max(2, min(100, X.shape[1] - 1, X.shape[0] - 1))
    vecs = TruncatedSVD(k, random_state=0).fit_transform(X)
    return normalize(vecs), "tfidf"


def stage_cluster(args, wd: Path) -> None:
    require(wd / "problems.jsonl", "extract")
    from sklearn.cluster import AgglomerativeClustering
    problems = read_jsonl(wd / "problems.jsonl")
    if len(problems) < 10:
        sys.exit("Too few problems to cluster. Collect more reviews.")
    if len(problems) > 15000:
        log("Warning: >15k problems is slow for this method. Consider a smaller --n.")
    texts = [p["problem"] for p in problems]
    log(f"Embedding {len(texts)} problem statements")
    vecs, used = embed(texts, args.embedder)
    threshold = args.threshold or (0.45 if used == "st" else 0.75)
    log(f"Clustering with {used} embeddings, cosine distance threshold {threshold}")
    labels = AgglomerativeClustering(
        n_clusters=None, distance_threshold=threshold, metric="cosine", linkage="average"
    ).fit_predict(vecs)

    # Renumber by size; small groups become -1 ("other")
    sizes = pd.Series(labels).value_counts()
    keep = [c for c in sizes.index if sizes[c] >= args.min_size]
    remap = {c: i for i, c in enumerate(keep)}
    for p, lab in zip(problems, labels):
        p["cluster"] = remap.get(int(lab), -1)
    write_jsonl(wd / "problems_clustered.jsonl", problems)
    np.save(wd / "embeddings.npy", vecs)
    other = sum(1 for p in problems if p["cluster"] == -1)
    top = ", ".join(str(int(sizes[c])) for c in keep[:10])
    log(f"{len(keep)} clusters with >= {args.min_size} problems; {other} problems in small groups")
    log(f"Largest cluster sizes: {top}")
    log("Tip: too many tiny clusters -> raise --threshold; one giant blob -> lower it.")


# ---------------------------------------------------------------------------
# Stage 4: label clusters
# ---------------------------------------------------------------------------

LABEL_SYSTEM = """You are a product strategist. You get a group of similar customer problems pulled from product reviews.
Describe the shared underlying need clearly and honestly. Base everything on the examples given; do not invent facts.
Judge whether a better-designed product could solve it (design gap), or whether it is mostly a quality-control defect or a one-brand issue (not a real opportunity)."""

LABEL_TOOL = {
    "name": "describe_cluster",
    "description": "Describe one group of customer problems.",
    "input_schema": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Short name, max 8 words"},
            "underlying_need": {"type": "string", "description": "1-2 sentences: what people actually need"},
            "who_and_when": {"type": "string", "description": "Who has this problem and in what situation"},
            "current_workarounds": {"type": "array", "items": {"type": "string"}},
            "why_products_fail": {"type": "string", "description": "Why current products don't solve it"},
            "product_directions": {"type": "array", "items": {"type": "string"},
                                   "description": "2-3 concrete product ideas that would address the need"},
            "opportunity_type": {"type": "string", "enum": ["design_gap", "quality_defect", "unclear"]},
        },
        "required": ["name", "underlying_need", "who_and_when", "current_workarounds",
                     "why_products_fail", "product_directions", "opportunity_type"],
    },
}


def _label_one(cid: int, items: list[dict], reviews: dict, seed: int) -> dict:
    rng = random.Random(seed + cid)
    sample = rng.sample(items, min(30, len(items)))
    lines = []
    for p in sample:
        extra = "; ".join(f"{k}: {p[k]}" for k in ("context", "who", "workaround", "wish") if p.get(k))
        lines.append(f"- {p['problem']}" + (f" ({extra})" if extra else ""))
    quotes = [short(reviews[p["review_id"]]["text"], 400) for p in sample[:4] if p["review_id"] in reviews]
    user = (f"This group has {len(items)} problems from "
            f"{len({p['product'] for p in items if p.get('product')}) or 'unknown'} different products.\n\n"
            "Problem statements (sample):\n" + "\n".join(lines) +
            "\n\nOriginal review excerpts:\n" + "\n".join(f'"{q}"' for q in quotes))
    return call_tool(LABEL_MODEL, LABEL_SYSTEM, user, LABEL_TOOL, max_tokens=2000)


def stage_label(args, wd: Path) -> None:
    require(wd / "problems_clustered.jsonl", "cluster")
    problems = read_jsonl(wd / "problems_clustered.jsonl")
    reviews = {r["id"]: r for r in read_jsonl(wd / "reviews.jsonl")}
    groups: dict[int, list[dict]] = {}
    for p in problems:
        if p["cluster"] >= 0:
            groups.setdefault(p["cluster"], []).append(p)
    cids = sorted(groups, key=lambda c: -len(groups[c]))[: args.max_clusters]
    path = wd / "labels.json"
    labels = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    def sig(c):  # fingerprint of a cluster's members, so re-clustering invalidates old labels
        return hashlib.md5(",".join(str(p["pid"]) for p in groups[c]).encode()).hexdigest()[:12]

    labels = {k: v for k, v in labels.items() if int(k) in groups and v.get("_sig") == sig(int(k))}
    todo = [c for c in cids if str(c) not in labels]
    log(f"Labeling {len(todo)} clusters with {LABEL_MODEL} ({len(cids) - len(todo)} cached)")
    with ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(_label_one, c, groups[c], reviews, args.seed): c for c in todo}
        for fut in as_completed(futures):
            c = futures[fut]
            try:
                labels[str(c)] = {**fut.result(), "_sig": sig(c)}
            except Exception as e:
                log(f"  cluster {c} failed: {e} (re-run 'label' to retry)")
    path.write_text(json.dumps(labels, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"Saved {len(labels)} cluster labels")


# ---------------------------------------------------------------------------
# Stage 5: report
# ---------------------------------------------------------------------------

# Scoring weights. Change these to match what you care about.
WEIGHTS = {
    "frequency": 0.35,   # how many problems are in the group
    "breadth": 0.25,     # how many different products share it (a category gap, not one bad product)
    "workarounds": 0.20, # people hacking their own fixes = strong buying signal
    "pain": 0.20,        # share of 1-star reviews
}
NOT_A_GAP_PENALTY = 0.5  # multiply score for clusters labeled quality_defect


def _year(ts) -> int | None:
    if ts is None or (isinstance(ts, float) and np.isnan(ts)):
        return None
    try:
        ts = float(ts)
        if ts > 1e11:  # milliseconds
            ts /= 1000
        return datetime.fromtimestamp(ts, tz=timezone.utc).year
    except (TypeError, ValueError):
        try:
            return pd.to_datetime(ts).year
        except Exception:
            return None


def stage_report(args, wd: Path) -> None:
    require(wd / "labels.json", "label")
    problems = read_jsonl(wd / "problems_clustered.jsonl")
    reviews = {r["id"]: r for r in read_jsonl(wd / "reviews.jsonl")}
    labels = json.loads((wd / "labels.json").read_text(encoding="utf-8"))
    meta = json.loads((wd / "meta.json").read_text(encoding="utf-8")) if (wd / "meta.json").exists() else {}
    df = pd.DataFrame(problems)
    df["has_workaround"] = df["workaround"].fillna("").str.strip().ne("")
    df["year"] = df["timestamp"].map(_year)
    latest_year = df["year"].dropna().max() if df["year"].notna().any() else None

    rows = []
    for cid_s, lab in labels.items():
        g = df[df["cluster"] == int(cid_s)]
        if g.empty:
            continue
        ratings = g["rating"].dropna()
        products = g["product"].replace("", np.nan).dropna()
        recent = None
        if latest_year is not None and g["year"].notna().any():
            recent = float((g["year"] >= latest_year - 1).mean())
        rows.append({
            "cluster": int(cid_s), "name": lab["name"], "type": lab["opportunity_type"],
            "problems": len(g), "share": len(g) / len(df),
            "products": products.nunique() if len(products) else None,
            "avg_rating": float(ratings.mean()) if len(ratings) else None,
            "pct_1star": float((ratings <= 1).mean()) if len(ratings) else None,
            "workaround_rate": float(g["has_workaround"].mean()),
            "recent_share": recent,
            "helpful_votes": int(g["helpful_vote"].fillna(0).sum()),
            "need": lab["underlying_need"], "who_when": lab["who_and_when"],
            "workarounds": lab["current_workarounds"], "why_fail": lab["why_products_fail"],
            "directions": lab["product_directions"],
        })
    if not rows:
        sys.exit("No labeled clusters to report.")
    res = pd.DataFrame(rows)

    def norm(col):
        s = res[col].fillna(0).astype(float)
        return s / s.max() if s.max() > 0 else s * 0

    res["score"] = (WEIGHTS["frequency"] * norm("problems") + WEIGHTS["breadth"] * norm("products")
                    + WEIGHTS["workarounds"] * norm("workaround_rate") + WEIGHTS["pain"] * norm("pct_1star"))
    res.loc[res["type"] == "quality_defect", "score"] *= NOT_A_GAP_PENALTY
    res["score"] = (100 * res["score"]).round(0)
    res = res.sort_values("score", ascending=False).reset_index(drop=True)

    csv_cols = ["score", "name", "type", "problems", "share", "products", "avg_rating",
                "pct_1star", "workaround_rate", "recent_share", "helpful_votes", "need",
                "who_when", "why_fail", "cluster"]
    out_csv = res[csv_cols].copy()
    out_csv["workarounds"] = res["workarounds"].map(" | ".join)
    out_csv["directions"] = res["directions"].map(" | ".join)
    out_csv.to_csv(wd / "clusters.csv", index=False, encoding="utf-8-sig")

    def fmt(x, pct=False, dec=1):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "n/a"
        return f"{100 * x:.0f}%" if pct else (f"{x:.{dec}f}" if isinstance(x, float) else str(x))

    md = ["# Idea Finder report", "",
          f"Source: {meta.get('source', 'unknown')}  ",
          f"Keywords: {', '.join(meta.get('keywords', [])) or 'none'}  ",
          f"Reviews analyzed: {len(reviews)} · Problems found: {len(df)} · "
          f"Groups reported: {len(res)}  ",
          f"Generated: {datetime.now():%Y-%m-%d %H:%M}", "",
          "**How to read this.** Score (0-100) mixes how often the problem shows up, how many "
          "different products share it, how often people invent workarounds, and how angry they "
          "are. A problem shared across many products is a gap in the whole category, not one "
          "bad product. Groups marked *quality_defect* are scored down: those are usually "
          "manufacturing issues, not new-product opportunities. Treat this as a list of leads "
          "to check with real people, not as proof.", ""]
    for i, r in res.iterrows():
        g = df[df["cluster"] == r["cluster"]]
        ex = g.sample(min(3, len(g)), random_state=args.seed)
        md += [f"## {i + 1}. {r['name']}  (score {r['score']:.0f})", "",
               f"*{r['type'].replace('_', ' ')}* · {r['problems']} problems "
               f"({fmt(r['share'], pct=True)} of all) · {fmt(r['products'])} products · "
               f"avg rating {fmt(r['avg_rating'])} · 1-star {fmt(r['pct_1star'], pct=True)} · "
               f"workarounds {fmt(r['workaround_rate'], pct=True)}"
               + (f" · last 2 yrs of data {fmt(r['recent_share'], pct=True)}" if r['recent_share'] is not None else ""),
               "", f"**Need:** {r['need']}", "", f"**Who / when:** {r['who_when']}", "",
               f"**Why current products fail:** {r['why_fail']}", ""]
        if r["workarounds"]:
            md += ["**What people do today:**"] + [f"- {w}" for w in r["workarounds"]] + [""]
        md += ["**Product directions to explore:**"] + [f"- {d}" for d in r["directions"]] + [""]
        md += ["**In their words:**"]
        for _, p in ex.iterrows():
            rv = reviews.get(p["review_id"])
            if rv:
                md.append(f"> {short(rv['text'], 260)}")
                md.append(">")
        if md[-1] == ">":
            md.pop()
        md.append("")
    (wd / "report.md").write_text("\n".join(md), encoding="utf-8")
    log(f"Wrote {wd / 'report.md'} and {wd / 'clusters.csv'}")
    print("\nTop groups:")
    for i, r in res.head(10).iterrows():
        print(f"  {i + 1:>2}. [{r['score']:>3.0f}] {r['name']}  ({r['problems']} problems, "
              f"{fmt(r['products'])} products, {r['type']})")


# ---------------------------------------------------------------------------
# Stage 6: sample for hand grading (eval set)
# ---------------------------------------------------------------------------

def stage_sample(args, wd: Path) -> None:
    require(wd / "extractions.jsonl", "extract")
    reviews = {r["id"]: r for r in read_jsonl(wd / "reviews.jsonl")}
    ex = [x for x in read_jsonl(wd / "extractions.jsonl") if x["review_id"] in reviews]
    rng = random.Random(args.seed)
    pick = rng.sample(ex, min(args.eval_n, len(ex)))
    rows = [{
        "review_id": x["review_id"], "rating": reviews[x["review_id"]].get("rating"),
        "review_text": reviews[x["review_id"]]["text"], "noise": x.get("noise"),
        "extracted_problems": "\n".join(f"- {p['problem']}" for p in x.get("problems", [])),
        "all_correct (y/n)": "", "made_up_problem (y/n)": "", "missed_problem": "", "notes": "",
    } for x in pick]
    path = wd / "eval_sample.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    log(f"Wrote {len(rows)} rows to {path}. Grade them, then fix the prompt where it goes wrong.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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


if __name__ == "__main__":
    main()
