"""Real article text instead of headline-only summaries. No network, no LLM."""
import dataclasses
import os
import sqlite3
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import collect
import compose
import config
import db
import geo
import pipeline
import render
import score
import tenant
from collect import Article

PUB = datetime(2026, 1, 1, tzinfo=timezone.utc)
BODY = ("Acme Steel will open a second rolling mill in Pune next year, adding 400 kt of "
        "annual capacity and 300 jobs, the company said on Tuesday.")


def _art(title="Acme opens new rolling mill", url="https://news.example.com/a", bucket="trigger",
         summary=None, **kw):
    return Article(title=title, url=url, source="S", published=PUB,
                   summary=title if summary is None else summary, query="q", bucket=bucket, **kw)


class _Resp:
    def __init__(self, text, status=200):
        self.text, self.status_code = text, status


def _mock_get(monkeypatch, html):
    import requests
    monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp(html))


def _no_bing(monkeypatch):
    monkeypatch.setattr(collect, "_fetch_bing", lambda *a, **k: [])


def test_page_og_description(monkeypatch):
    _no_bing(monkeypatch)
    _mock_get(monkeypatch, f'<html><head><meta property="og:description" '
                           f'content="{BODY.replace(",", "&#44;")}"></head><body></body></html>')
    out = collect.fetch_article_text(_art())
    assert out == BODY


def test_google_news_uses_best_bing_match(monkeypatch):
    import requests

    def boom(*a, **k):
        raise AssertionError("google news link must not be fetched directly")
    monkeypatch.setattr(requests, "get", boom)
    a = _art(url="https://news.google.com/rss/articles/CBMi123")

    def fake_bing(q, bucket, cutoff):
        assert q == a.title
        return [Article(title="Totally unrelated cricket result", url="https://bing.com/x",
                        source="B", published=PUB, summary="Cricket text " * 20),
                Article(title="Acme opens new rolling mill", url="https://bing.com/y",
                        source="B", published=PUB, summary=BODY)]
    monkeypatch.setattr(collect, "_fetch_bing", fake_bing)
    assert collect.fetch_article_text(a) == BODY


def test_unrelated_bing_result_ignored(monkeypatch):
    a = _art(url="https://news.google.com/rss/articles/CBMi123")
    monkeypatch.setattr(collect, "_fetch_bing", lambda *a_, **k: [
        Article(title="Something else entirely about football", url="https://bing.com/x",
                source="B", published=PUB, summary="Football text " * 20)])
    assert collect.fetch_article_text(a) == ""


def test_headline_only_text_discarded(monkeypatch):
    _no_bing(monkeypatch)
    a = _art()
    _mock_get(monkeypatch, f'<meta name="description" content="{a.title}">')
    assert collect.fetch_article_text(a) == ""


def test_fetch_never_raises(monkeypatch):
    import requests

    def boom(*a, **k):
        raise RuntimeError("down")
    monkeypatch.setattr(requests, "get", boom)
    monkeypatch.setattr(collect, "_fetch_bing", boom)
    assert collect.fetch_article_text(_art()) == ""


def test_enrich_texts_picks_by_score_and_replaces_only_longer(monkeypatch):
    arts = [_art(title=f"T{i}", url=f"https://x/{i}", ev_relevancy=1.0, ev_recency=i / 10)
            for i in range(1, 6)]
    arts.append(_art(title="Ind", url="https://x/ind", bucket="industry",
                     ev_relevancy=1.0, ev_recency=0.5))
    fetched = []

    def fake(a, timeout=8):
        fetched.append(a.title)
        return BODY if a.title != "T5" else "short"
    monkeypatch.setattr(collect, "_article_text_src",
                        lambda a, timeout: (fake(a), "page"))
    n = collect.enrich_texts(arts, per_bucket=2)
    # trigger: T5, T4 by ev score; industry: Ind
    assert sorted(fetched) == ["Ind", "T4", "T5"]
    by = {a.title: a for a in arts}
    assert by["T4"].summary == BODY and by["T4"].text_source == "page"
    assert by["Ind"].summary == BODY
    assert by["T5"].summary == "T5" and by["T5"].text_source == ""  # "short" not 40+ longer
    assert by["T1"].summary == "T1"
    assert n == 2


def test_enrich_texts_survives_raising_fetch(monkeypatch):
    def boom(a, timeout):
        raise RuntimeError("x")
    monkeypatch.setattr(collect, "_article_text_src", boom)
    assert collect.enrich_texts([_art(), _art(title="B", url="https://x/b")]) == 0


# --- pipeline: skipped in frozen replay, runs live ---------------------------------
@dataclasses.dataclass
class _P:
    value_prop: str = "vp"
    competitors: list = dataclasses.field(default_factory=lambda: ["Rival"])
    industries: list = dataclasses.field(default_factory=lambda: ["Steel"])
    trigger_queries: list = dataclasses.field(default_factory=lambda: ["q1"])
    industry_queries: list = dataclasses.field(default_factory=lambda: ["q2"])


LEAD = {"email": "a@acme.com", "company": "Acme", "name": "Sam", "website": "",
        "country": "India", "topic": ""}


def _setup(monkeypatch, enrich_calls):
    class PC:
        def get(self, *a, **k):
            return _P()
    monkeypatch.setattr(tenant, "ProfileCache", PC)
    monkeypatch.setattr(db, "get_conn", lambda: None)
    monkeypatch.setattr(db, "save_brief", lambda *a, **k: 1)
    monkeypatch.setattr(collect, "collect", lambda *a, **k: [_art(title="T1"), _art(title="T2")])
    monkeypatch.setattr(collect, "plan_searches", lambda *a, **k: ([("q", "trigger", (), "")], []))
    monkeypatch.setattr(collect, "enrich_texts", lambda arts, *a, **k: enrich_calls.append(1) or 3)
    monkeypatch.setattr(geo, "filter_for_lead", lambda arts, *a, **k: (arts, "country"))
    monkeypatch.setattr(score, "score_all", lambda arts, profile, topic="": arts)
    import brief_builder
    monkeypatch.setattr(brief_builder, "build_brief", lambda arts, profile, log=print, *,
                        market_topic="": ({"industry_outlook": []},
                                          {"customer": None, "competition": None}))
    monkeypatch.setattr(compose, "build_context", lambda *a, **k: {})
    monkeypatch.setattr(render, "render", lambda ctx, name: "<html></html>")


def _run(tmp_path):
    return pipeline.stage_generate(sqlite3.connect(":memory:"), [LEAD], kinds=["signal"],
                                   workers=1, run_id=1, outdir=str(tmp_path))


def test_pipeline_enriches_live_and_skips_frozen(monkeypatch, tmp_path):
    import json
    monkeypatch.delenv("CMI_EVIDENCE_DIR", raising=False)
    monkeypatch.setattr(config, "FETCH_ARTICLE_TEXT", True)
    calls = []
    _setup(monkeypatch, calls)
    first = _run(tmp_path / "a")[0]["evidence_path"]
    assert calls == [1]
    assert json.load(open(first, encoding="utf-8"))["texts_fetched"] == 3
    monkeypatch.setenv("CMI_EVIDENCE_DIR", os.path.dirname(first))
    rows = pipeline.stage_generate(sqlite3.connect(":memory:"), [LEAD], kinds=["signal"],
                                   workers=1, run_id=1, outdir=str(tmp_path / "b"))
    assert calls == [1]  # frozen replay did not fetch
    assert "texts_fetched" not in json.load(open(rows[0]["evidence_path"], encoding="utf-8"))


def test_pipeline_flag_off_and_failure_safe(monkeypatch, tmp_path):
    monkeypatch.delenv("CMI_EVIDENCE_DIR", raising=False)
    calls = []
    _setup(monkeypatch, calls)
    monkeypatch.setattr(config, "FETCH_ARTICLE_TEXT", False)
    _run(tmp_path / "a")
    assert calls == []
    monkeypatch.setattr(config, "FETCH_ARTICLE_TEXT", True)

    def boom(*a, **k):
        raise RuntimeError("x")
    monkeypatch.setattr(collect, "enrich_texts", boom)
    assert _run(tmp_path / "b")[0].get("evidence_path")


def test_text_source_roundtrips_in_pool_and_old_files_load(monkeypatch, tmp_path):
    import json
    assert "text_source" in pipeline._EVIDENCE_ART_FIELDS
    p = tmp_path / "x.evidence.json"
    p.write_text(json.dumps({"pool": [{"title": "T", "url": "u", "published": PUB.isoformat()}]}))
    monkeypatch.setenv("CMI_EVIDENCE_DIR", str(tmp_path))
    assert pipeline._load_frozen_pool("x.evidence.json")[0].text_source == ""
