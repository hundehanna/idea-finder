"""Local SQLite store: reviews with keyword search, an extraction cache, and embeddings.

The agent's tools read from here, so each tool call is fast and free.
'prepare' fills the reviews once; the extraction cache fills up as segments are analyzed.

Every function takes an open connection, so tests can use a temporary database.
A connection should stay in one thread; collect results first, then save them.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS reviews (
    id           TEXT PRIMARY KEY,   -- stable, e.g. BAB-3f9a1c2b7d
    category     TEXT NOT NULL,
    product      TEXT,
    rating       REAL,
    title        TEXT,
    text         TEXT NOT NULL,
    timestamp    INTEGER,            -- milliseconds since 1970, as in the dataset
    helpful_vote INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS reviews_by_category ON reviews (category, rating);

-- One row per review we have extracted, even when it had no problems (noise).
CREATE TABLE IF NOT EXISTS extractions (
    review_id      TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    model          TEXT NOT NULL,
    noise          INTEGER NOT NULL,
    created_at     TEXT NOT NULL,
    PRIMARY KEY (review_id, prompt_version)
);

CREATE TABLE IF NOT EXISTS problems (
    pid            INTEGER PRIMARY KEY,
    review_id      TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    problem        TEXT NOT NULL,
    context        TEXT,
    who            TEXT,
    workaround     TEXT,
    wish           TEXT
);
CREATE INDEX IF NOT EXISTS problems_by_review ON problems (review_id, prompt_version);

CREATE TABLE IF NOT EXISTS problem_embeddings (
    pid    INTEGER NOT NULL,
    model  TEXT NOT NULL,
    vector BLOB NOT NULL,            -- float32 numbers, scaled to length 1
    PRIMARY KEY (pid, model)
);

CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""

# A plain FTS5 table with its own copy of title and text. Its rowid is the
# reviews rowid, so a search result joins straight back to the review.
# 'porter' makes "leaking" match "leaks"; 'unicode61' handles accents.
FTS_SCHEMA = ("CREATE VIRTUAL TABLE IF NOT EXISTS reviews_fts "
              "USING fts5(title, text, tokenize='porter unicode61')")

REVIEW_COLUMNS = ["id", "category", "product", "rating", "title", "text", "timestamp", "helpful_vote"]


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def has_fts5(conn: sqlite3.Connection) -> bool:
    """True if this SQLite build has the FTS5 search extension."""
    try:
        conn.execute("CREATE VIRTUAL TABLE temp.fts5_check USING fts5(x)")
        conn.execute("DROP TABLE temp.fts5_check")
        return True
    except sqlite3.OperationalError:
        return False


def connect(path: Path, use_fts: bool = True) -> sqlite3.Connection:
    """Open (and create if needed) the database. Uses FTS5 when it is available."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    if use_fts and has_fts5(conn):
        conn.execute(FTS_SCHEMA)
    conn.commit()
    return conn


def fts_enabled(conn: sqlite3.Connection) -> bool:
    """True if this database has the FTS5 search table."""
    row = conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'reviews_fts'").fetchone()
    return row is not None


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value))
    conn.commit()


def get_meta(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else None


# ---------------------------------------------------------------------------
# Reviews
# ---------------------------------------------------------------------------

def review_id(category: str, user_id: str, asin: str, timestamp: object) -> str:
    """A stable ID, so cited IDs still work after the database is rebuilt."""
    digest = hashlib.sha1(f"{user_id}|{asin}|{timestamp}".encode("utf-8")).hexdigest()
    return f"{category[:3].upper()}-{digest[:10]}"


def add_reviews(conn: sqlite3.Connection, rows: list[dict]) -> int:
    """Insert reviews (skipping IDs we already have). Returns how many were new."""
    added = 0
    fts = fts_enabled(conn)
    with conn:  # one transaction: much faster than committing each row
        for r in rows:
            cur = conn.execute(
                f"INSERT OR IGNORE INTO reviews ({', '.join(REVIEW_COLUMNS)}) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [r.get(c) for c in REVIEW_COLUMNS])
            if cur.rowcount == 1:
                added += 1
                if fts:
                    conn.execute("INSERT INTO reviews_fts (rowid, title, text) VALUES (?, ?, ?)",
                                 (cur.lastrowid, r.get("title") or "", r["text"]))
    return added


def delete_category(conn: sqlite3.Connection, category: str) -> int:
    """Remove one category's reviews (used by 'prepare --force'). Returns how many were removed.

    Cached extractions are kept: review IDs are stable, so they still match after a rebuild.
    """
    with conn:
        if fts_enabled(conn):
            conn.execute("DELETE FROM reviews_fts WHERE rowid IN "
                         "(SELECT rowid FROM reviews WHERE category = ?)", (category,))
        return conn.execute("DELETE FROM reviews WHERE category = ?", (category,)).rowcount


def category_counts(conn: sqlite3.Connection) -> list[dict]:
    """Reviews per category, with product count and average rating."""
    rows = conn.execute("""
        SELECT category, COUNT(*) AS reviews, COUNT(DISTINCT product) AS products,
               ROUND(AVG(rating), 2) AS avg_rating
        FROM reviews GROUP BY category ORDER BY category""").fetchall()
    return [dict(r) for r in rows]


def get_reviews(conn: sqlite3.Connection, ids: list[str]) -> dict[str, dict]:
    """Full reviews by ID. Unknown IDs are left out."""
    if not ids:
        return {}
    marks = ", ".join("?" * len(ids))
    rows = conn.execute(f"SELECT * FROM reviews WHERE id IN ({marks})", list(ids)).fetchall()
    return {r["id"]: dict(r) for r in rows}


def query_words(query: str) -> list[str]:
    """The plain words in a search query, lowercased."""
    return re.findall(r"\w+", query.lower())


def _fts_query(words: list[str]) -> str:
    """Quote each word, so FTS5 treats it as text, not syntax (AND, NEAR, "*", ...)."""
    return " ".join(f'"{w}"' for w in words)


def _like_snippet(text: str, words: list[str], width: int = 80) -> str:
    """A short piece of text around the first matching word, for the LIKE fallback."""
    low = text.lower()
    hits = [low.find(w) for w in words if w in low]
    if not hits:
        return text[: 2 * width]
    start = max(0, min(hits) - width)
    piece = text[start: min(hits) + width]
    return ("…" if start > 0 else "") + piece + ("…" if start + len(piece) < len(text) else "")


def search_reviews(conn: sqlite3.Connection, query: str, category: str | None = None,
                   max_rating: float = 3, limit: int = 10) -> list[dict]:
    """Keyword search. Every word must appear. Returns short snippets with review IDs."""
    words = query_words(query)
    if not words:
        return []
    if fts_enabled(conn):
        rows = conn.execute("""
            SELECT r.id, r.category, r.product, r.rating, r.title,
                   snippet(reviews_fts, -1, '[', ']', '…', 24) AS snippet
            FROM reviews_fts JOIN reviews r ON r.rowid = reviews_fts.rowid
            WHERE reviews_fts MATCH ?
              AND (? IS NULL OR r.category = ?)
              AND (r.rating IS NULL OR r.rating <= ?)
            ORDER BY bm25(reviews_fts)
            LIMIT ?""", (_fts_query(words), category, category, max_rating, limit)).fetchall()
        return [dict(r) for r in rows]

    # Fallback without FTS5: slower, no ranking, no word stemming.
    where = " AND ".join("lower(coalesce(title, '') || ' ' || text) LIKE ?" for _ in words)
    rows = conn.execute(f"""
        SELECT id, category, product, rating, title, text FROM reviews
        WHERE {where} AND (? IS NULL OR category = ?) AND (rating IS NULL OR rating <= ?)
        ORDER BY helpful_vote DESC, id LIMIT ?""",
        [f"%{w}%" for w in words] + [category, category, max_rating, limit]).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["snippet"] = _like_snippet(d.pop("text"), words)
        out.append(d)
    return out


# ---------------------------------------------------------------------------
# Extraction cache
# ---------------------------------------------------------------------------

PROBLEM_FIELDS = ["problem", "context", "who", "workaround", "wish"]


def get_cached_extractions(conn: sqlite3.Connection, review_ids: list[str],
                           prompt_version: str) -> dict[str, dict]:
    """Saved extraction results for these reviews and this prompt version.

    Returns {review_id: {"noise": bool, "problems": [...]}} for the reviews we already have.
    """
    if not review_ids:
        return {}
    marks = ", ".join("?" * len(review_ids))
    done = conn.execute(
        f"SELECT review_id, noise FROM extractions WHERE prompt_version = ? AND review_id IN ({marks})",
        [prompt_version, *review_ids]).fetchall()
    out = {r["review_id"]: {"noise": bool(r["noise"]), "problems": []} for r in done}
    rows = conn.execute(
        f"SELECT * FROM problems WHERE prompt_version = ? AND review_id IN ({marks}) ORDER BY pid",
        [prompt_version, *review_ids]).fetchall()
    for r in rows:
        if r["review_id"] in out:
            out[r["review_id"]]["problems"].append({"pid": r["pid"], **{k: r[k] for k in PROBLEM_FIELDS}})
    return out


def save_extraction(conn: sqlite3.Connection, review_id: str, prompt_version: str, model: str,
                    noise: bool, problems: list[dict]) -> list[int]:
    """Save one review's extraction result. Returns the new problem IDs (pids).

    If this review and prompt version are already saved, nothing changes.
    """
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with conn:
        cur = conn.execute(
            "INSERT OR IGNORE INTO extractions (review_id, prompt_version, model, noise, created_at) "
            "VALUES (?, ?, ?, ?, ?)", (review_id, prompt_version, model, int(noise), now))
        if cur.rowcount == 0:
            return []
        pids = []
        for p in problems:
            cur = conn.execute(
                "INSERT INTO problems (review_id, prompt_version, problem, context, who, workaround, wish) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                [review_id, prompt_version] + [p.get(k, "") for k in PROBLEM_FIELDS])
            pids.append(cur.lastrowid)
    return pids


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

def save_embeddings(conn: sqlite3.Connection, pids: list[int], vectors: np.ndarray, model: str) -> None:
    """Store one vector per problem. Vectors are scaled to length 1 first."""
    vecs = np.asarray(vectors, dtype=np.float32)
    vecs = vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12)
    with conn:
        conn.executemany(
            "INSERT OR REPLACE INTO problem_embeddings (pid, model, vector) VALUES (?, ?, ?)",
            [(int(pid), model, v.tobytes()) for pid, v in zip(pids, vecs)])


def load_embeddings(conn: sqlite3.Connection, model: str) -> tuple[list[int], np.ndarray]:
    """All stored vectors for one embedding model, as (pids, matrix)."""
    rows = conn.execute("SELECT pid, vector FROM problem_embeddings WHERE model = ? ORDER BY pid",
                        (model,)).fetchall()
    if not rows:
        return [], np.zeros((0, 0), dtype=np.float32)
    return [r["pid"] for r in rows], np.vstack([np.frombuffer(r["vector"], dtype=np.float32) for r in rows])


def similar_problems(conn: sqlite3.Connection, query_vector: np.ndarray, model: str,
                     k: int = 10) -> list[dict]:
    """The k stored problems closest in meaning to the query vector, across all categories.

    A plain dot product over every vector. At a few thousand problems this takes milliseconds.
    """
    pids, matrix = load_embeddings(conn, model)
    if not pids:
        return []
    q = np.asarray(query_vector, dtype=np.float32).ravel()
    q = q / max(float(np.linalg.norm(q)), 1e-12)
    scores = matrix @ q
    top = np.argsort(-scores)[:k]
    by_pid = {pids[i]: float(scores[i]) for i in top}
    marks = ", ".join("?" * len(by_pid))
    rows = conn.execute(f"""
        SELECT p.pid, p.problem, p.review_id, r.category, r.product
        FROM problems p JOIN reviews r ON r.id = p.review_id
        WHERE p.pid IN ({marks})""", list(by_pid)).fetchall()
    out = [{**dict(r), "score": round(by_pid[r["pid"]], 4)} for r in rows]
    return sorted(out, key=lambda d: -d["score"])
