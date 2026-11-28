"""Evidence file per brief + frozen-evidence mode. No network, no LLM."""
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
import dataclasses

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import collect
import compose
import config
import db
import filters
import geo
import pipeline
import render
import score
import tenant
from collect import Article

LEAD = {"email": "a@acme.com", "company": "Acme", "name": "Sam", "website": "",
        "country": "India", "topic": ""}
@dataclasses.dataclass
class _P:
    value_prop: str = "vp"
    competitors: list = dataclasses.field(default_factory=lambda: ["Rival"])
    industries: list = dataclasses.field(default_factory=lambda: ["Steel"])
    trigger_queries: list = dataclasses.field(default_factory=lambda: ["q1"])
    industry_queries: list = dataclasses.field(default_factory=lambda: ["q2"])


PROFILE = _P()


def _art(i):
    return Article(title=f"T{i}", url=f"http://x/{i}", source="S",
                   published=datetime(2026, 1, 1, tzinfo=timezone.utc),
                   summary="sum", query="q", bucket="trigger")


def _setup(monkeypatch, collect_calls):
    class PC:
        def get(self, *a, **k):
            return PROFILE
    monkeypatch.setattr(tenant, "ProfileCache", PC)
    monkeypatch.setattr(db, "get_conn", lambda: None)
    monkeypatch.setattr(config, "FETCH_ARTICLE_TEXT", False)   # no live article fetches
    monkeypatch.setattr(db, "save_brief", lambda *a, **k: 1)

    def fake_collect(*a, **k):
        collect_calls.append(1)
        return [_art(1), _art(2)]
    monkeypatch.setattr(collect, "collect", fake_collect)
    monkeypatch.setattr(collect, "plan_searches", lambda *a, **k: ([("q", "trigger", (), "")], []))
    monkeypatch.setattr(geo, "filter_for_lead", lambda arts, *a, **k: (arts, "country"))
    # Layer 1 drops two-word titles like "T1" as bare titles; this test is about evidence files.
    monkeypatch.setattr(filters, "apply_l1", lambda arts: (arts, []))
    monkeypatch.setattr(score, "score_all", lambda arts, profile, topic="": arts)
    import brief_builder

    def fake_build(arts, profile, log=print, *, market_topic=""):
        log("  skipping 'X' -- event is in Y")
        return ({"industry_outlook": [{"headline": "H", "source_url": "u", "source_name": "S",
                                       "date": "Jan 1"}]},
                {"customer": {"company": "C", "article": "T1", "url": "http://x/1",
                              "source": "S", "published": arts[0].published},
                 "competition": None})
    monkeypatch.setattr(brief_builder, "build_brief", fake_build)

    def fake_ctx(profile, brief, kind, market_topic="", log=print):
        log("ctx line")
        return {}
    monkeypatch.setattr(compose, "build_context", fake_ctx)
    monkeypatch.setattr(render, "render", lambda ctx, name: "<html></html>")


def _run(tmp_path):
    return pipeline.stage_generate(sqlite3.connect(":memory:"), [LEAD], kinds=["signal"],
                                   workers=1, run_id=1, outdir=str(tmp_path))


def test_evidence_written(monkeypatch, tmp_path):
    monkeypatch.delenv("CMI_EVIDENCE_DIR", raising=False)
    _setup(monkeypatch, [])
    rows = _run(tmp_path)
    p = rows[0]["evidence_path"]
    assert p.endswith(".evidence.json") and os.path.exists(p)
    ev = json.load(open(p, encoding="utf-8"))
    assert {"lead", "searches", "pool", "used", "build_log", "passed_to_builder"} <= set(ev)
    assert len(ev["pool"]) == 2 and ev["pool"][0]["eligible"] in (True, False)
    assert "Potential Customer" in ev["used"] and "Industry Outlook" in ev["used"]
    assert ev["build_log"] and ev["compose_log"]


def test_frozen_mode_skips_collect(monkeypatch, tmp_path):
    monkeypatch.delenv("CMI_EVIDENCE_DIR", raising=False)
    calls = []
    _setup(monkeypatch, calls)
    first = _run(tmp_path / "a")[0]["evidence_path"]
    assert calls == [1]
    monkeypatch.setenv("CMI_EVIDENCE_DIR", os.path.dirname(first))
    rows = pipeline.stage_generate(sqlite3.connect(":memory:"), [LEAD], kinds=["signal"],
                                   workers=1, run_id=1, outdir=str(tmp_path / "b"))
    assert calls == [1]  # collect not called again
    ev = json.load(open(rows[0]["evidence_path"], encoding="utf-8"))
    assert [r["title"] for r in ev["pool"]] == ["T1", "T2"]
