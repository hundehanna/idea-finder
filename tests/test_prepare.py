"""Tests for 'prepare': building the review store from raw review files (offline fixtures)."""

from __future__ import annotations

import json
from pathlib import Path

from idea_finder import store
from idea_finder.cli import main
from idea_finder.utils import clean_text

AMAZON = Path(__file__).parent / "fixtures" / "amazon"
CATS = ["Desk_Lamps", "Toy_Robots", "Travel_Mugs"]


def expected_ids(category: str, scan: int = 0) -> set[str]:
    """IDs of the reviews 'prepare' should keep: 1-3 stars and at least 60 characters."""
    lines = (AMAZON / f"{category}.jsonl").read_text(encoding="utf-8").splitlines()
    if scan:
        lines = lines[:scan]
    out = set()
    for line in lines:
        r = json.loads(line)
        if r["rating"] <= 3 and len(clean_text(r["text"])) >= 60:
            out.add(store.review_id(category, r["user_id"], r["asin"], r["timestamp"]))
    return out


def prepare(db: Path, *extra: str) -> None:
    main(["prepare", "--source-dir", str(AMAZON), "--categories", ",".join(CATS), "--db", str(db), *extra])


def ids_by_category(db: Path) -> dict[str, set[str]]:
    conn = store.connect(db)
    rows = conn.execute("SELECT id, category FROM reviews").fetchall()
    conn.close()
    out: dict[str, set[str]] = {}
    for r in rows:
        out.setdefault(r["category"], set()).add(r["id"])
    return out


def test_prepare_keeps_every_matching_review(tmp_path, capsys):
    prepare(tmp_path / "t.db")
    got = ids_by_category(tmp_path / "t.db")
    for cat in CATS:
        assert got[cat] == expected_ids(cat)
    out = capsys.readouterr().out
    assert "Category" in out and "Travel_Mugs" in out   # the count table is printed


def test_prepare_samples_the_same_reviews_every_time(tmp_path):
    prepare(tmp_path / "a.db", "--per-category", "10")
    prepare(tmp_path / "b.db", "--per-category", "10")
    a, b = ids_by_category(tmp_path / "a.db"), ids_by_category(tmp_path / "b.db")
    assert a == b
    assert all(len(a[cat]) == 10 and a[cat] <= expected_ids(cat) for cat in CATS)


def test_prepare_respects_scan_limit(tmp_path):
    prepare(tmp_path / "t.db", "--scan", "20")
    got = ids_by_category(tmp_path / "t.db")
    assert all(got.get(cat, set()) == expected_ids(cat, scan=20) for cat in CATS)


def test_rerun_skips_and_force_rebuilds(tmp_path, capsys):
    db = tmp_path / "t.db"
    prepare(db, "--per-category", "10")
    prepare(db)  # no --force: categories already there are skipped
    assert all(len(ids) == 10 for ids in ids_by_category(db).values())
    assert "already in the database" in capsys.readouterr().out
    prepare(db, "--force")
    got = ids_by_category(db)
    assert all(got[cat] == expected_ids(cat) for cat in CATS)
    conn = store.connect(db)
    assert store.search_reviews(conn, "cup holder")     # the search index was rebuilt too
    fts_rows = conn.execute("SELECT COUNT(*) FROM reviews_fts").fetchone()[0]
    assert fts_rows == sum(len(v) for v in got.values())


def test_prepare_records_its_settings(tmp_path):
    prepare(tmp_path / "t.db", "--per-category", "10")
    conn = store.connect(tmp_path / "t.db")
    info = json.loads(store.get_meta(conn, "prepare:Travel_Mugs"))
    assert info["kept"] == 10 and info["per_category"] == 10 and info["lines_scanned"] == 48
    assert info["matching"] == len(expected_ids("Travel_Mugs"))


def test_prepare_needs_no_api_key(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    prepare(tmp_path / "t.db", "--per-category", "5")
    assert ids_by_category(tmp_path / "t.db")
