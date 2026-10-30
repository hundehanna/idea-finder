"""Tests for the SQLite store: search, the extraction cache and embeddings."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
import pytest

from idea_finder import pipeline, store
from idea_finder.utils import clean_text

AMAZON = Path(__file__).parent / "fixtures" / "amazon"


def fixture_reviews() -> list[dict]:
    """All fixture reviews (every rating), in the store's format."""
    rows = []
    for path in sorted(AMAZON.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            rows.append({
                "id": store.review_id(path.stem, r["user_id"], r["asin"], r["timestamp"]),
                "category": path.stem, "product": r["parent_asin"], "rating": r["rating"],
                "title": clean_text(r["title"]), "text": clean_text(r["text"]),
                "timestamp": r["timestamp"], "helpful_vote": r["helpful_vote"],
            })
    return rows


@pytest.fixture(params=[True, False], ids=["fts5", "like"])
def db(request, tmp_path):
    """A filled database, once with FTS5 search and once with the LIKE fallback."""
    conn = store.connect(tmp_path / "test.db", use_fts=request.param)
    store.add_reviews(conn, fixture_reviews())
    yield conn
    conn.close()


def test_this_python_has_fts5(tmp_path):
    conn = store.connect(tmp_path / "x.db")
    assert store.has_fts5(conn) and store.fts_enabled(conn)


def test_review_id_is_stable_and_short():
    a = store.review_id("Baby_Products", "U1", "B0X", 1588687728923)
    assert a == store.review_id("Baby_Products", "U1", "B0X", 1588687728923)
    assert a != store.review_id("Baby_Products", "U1", "B0X", 1588687728924)
    assert re.fullmatch(r"BAB-[0-9a-f]{10}", a)


def test_add_reviews_skips_ids_we_have(db):
    assert store.add_reviews(db, fixture_reviews()) == 0
    counts = {c["category"]: c["reviews"] for c in store.category_counts(db)}
    assert counts == {"Desk_Lamps": 49, "Toy_Robots": 48, "Travel_Mugs": 48}


def test_search_finds_matching_low_star_reviews(db):
    hits = store.search_reviews(db, "cup holder", limit=50)
    assert len(hits) >= 5
    assert all(h["category"] == "Travel_Mugs" and h["rating"] <= 3 for h in hits)
    assert all("cup" in h["snippet"].lower() for h in hits)
    assert set(hits[0]) == {"id", "category", "product", "rating", "title", "snippet"}


def test_search_filters_by_category_and_rating(db):
    assert store.search_reviews(db, "cup holder", category="Desk_Lamps") == []
    assert store.search_reviews(db, "solid build", max_rating=3) == []      # only in 4-5 star reviews
    assert len(store.search_reviews(db, "solid build", max_rating=5)) >= 1


def test_fts_and_like_find_the_same_reviews(tmp_path):
    fts = store.connect(tmp_path / "a.db", use_fts=True)
    like = store.connect(tmp_path / "b.db", use_fts=False)
    for conn in (fts, like):
        store.add_reviews(conn, fixture_reviews())
    for q in ["cup holder", "cable", "clamp desk"]:
        ids_fts = {h["id"] for h in store.search_reviews(fts, q, limit=100)}
        ids_like = {h["id"] for h in store.search_reviews(like, q, limit=100)}
        assert ids_fts == ids_like, q


def test_fts_matches_other_word_forms(tmp_path):
    conn = store.connect(tmp_path / "a.db")
    store.add_reviews(conn, fixture_reviews())
    assert store.search_reviews(conn, "leaking")  # the reviews say "leaks" and "leaked"


@pytest.mark.parametrize("query", ["can't", '"lid', "AND OR NOT", "lid-leak", "NEAR(lid)", "***", ""])
def test_awkward_queries_do_not_crash(db, query):
    assert isinstance(store.search_reviews(db, query), list)


def test_search_handles_non_ascii(db):
    hits = store.search_reviews(db, "café")
    assert hits and all(h["title"] == "Déçu" for h in hits)


def test_snippet_of_a_long_review_is_clean_text(tmp_path):
    """A match in the middle of a long review gives a cut snippet with a plain '...' marker."""
    filler = " ".join(f"word{i}" for i in range(80))
    text = f"{filler} the mug does not fit my cup holder at all {filler}"
    for use_fts in (True, False):
        conn = store.connect(tmp_path / f"long_{use_fts}.db", use_fts=use_fts)
        store.add_reviews(conn, [{"id": "TES-1", "category": "Test", "rating": 1.0, "title": "",
                                  "text": text}])
        snippet = store.search_reviews(conn, "cup holder")[0]["snippet"]
        assert "�" not in snippet
        assert snippet.startswith("...") and snippet.endswith("...")
        assert len(snippet) < len(text) / 2


def test_search_is_fast(db):
    start = time.perf_counter()
    for _ in range(20):
        store.search_reviews(db, "lid leaks")
    assert (time.perf_counter() - start) / 20 < 0.1


def test_get_reviews(db):
    some = [r["id"] for r in fixture_reviews()[:3]]
    got = store.get_reviews(db, some + ["NOT-a-real-id"])
    assert set(got) == set(some)
    assert got[some[0]]["text"]


def test_extraction_cache_round_trip(db):
    a, b, c = [r["id"] for r in fixture_reviews()[:3]]
    p = {"problem": "Lid leaks", "context": "", "who": "", "workaround": "bag", "wish": ""}
    pids = store.save_extraction(db, a, "v1", "claude-haiku-5-5", False, [p, {**p, "problem": "Gets cold"}])
    store.save_extraction(db, b, "v1", "claude-haiku-5-5", True, [])
    cached = store.get_cached_extractions(db, [a, b, c], "v1")
    assert set(cached) == {a, b}                       # c was never extracted
    assert [x["problem"] for x in cached[a]["problems"]] == ["Lid leaks", "Gets cold"]
    assert [x["pid"] for x in cached[a]["problems"]] == pids
    assert cached[b] == {"noise": True, "problems": []}
    assert store.get_cached_extractions(db, [a, b], "v2") == {}   # new prompt version: a cache miss


def test_saving_twice_does_not_duplicate(db):
    a = fixture_reviews()[0]["id"]
    p = {"problem": "Lid leaks", "context": "", "who": "", "workaround": "", "wish": ""}
    assert store.save_extraction(db, a, "v1", "m", False, [p])
    assert store.save_extraction(db, a, "v1", "m", False, [p]) == []
    assert len(store.get_cached_extractions(db, [a], "v1")[a]["problems"]) == 1


def test_prompt_version_changes_with_the_prompt(monkeypatch):
    before = pipeline.extract_prompt_version()
    assert before == pipeline.extract_prompt_version()
    monkeypatch.setattr(pipeline, "EXTRACT_SYSTEM", pipeline.EXTRACT_SYSTEM + " Be brief.")
    assert pipeline.extract_prompt_version() != before


def test_embeddings_round_trip_and_nearest(db):
    ids = [r["id"] for r in fixture_reviews()[:3]]
    pids = []
    for rid, text in zip(ids, ["Lid leaks", "Gets cold fast", "Too loud"]):
        pids += store.save_extraction(db, rid, "v1", "m", False,
                                      [{"problem": text, "context": "", "who": "", "workaround": "", "wish": ""}])
    vecs = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 3]], dtype=np.float32)
    store.save_embeddings(db, pids, vecs, "test-model")
    got_pids, matrix = store.load_embeddings(db, "test-model")
    assert got_pids == pids
    assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0)
    hits = store.similar_problems(db, np.array([0.1, 0.9, 0.0]), "test-model", k=2)
    assert [h["problem"] for h in hits] == ["Gets cold fast", "Lid leaks"]
    assert hits[0]["review_id"] == ids[1] and hits[0]["score"] > hits[1]["score"]
    assert store.similar_problems(db, np.array([1.0, 0, 0]), "other-model") == []


def test_meta(db):
    store.set_meta(db, "prepare:Desk_Lamps", "{}")
    assert store.get_meta(db, "prepare:Desk_Lamps") == "{}"
    assert store.get_meta(db, "missing") is None
