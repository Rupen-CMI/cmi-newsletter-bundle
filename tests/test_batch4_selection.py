"""Potential Customer selection: would_buy, buyers only, lead-country tie; low-confidence skips rivals."""
from datetime import datetime, timezone
from types import SimpleNamespace

import brief_builder
import free_research
import llm
from collect import Article

NOW = datetime.now(timezone.utc)


def _art(title, summary="", **kw):
    return Article(title=title, url="https://x.example/" + title[:8].replace(" ", ""),
                   source="Wire", published=NOW, summary=summary, bucket="trigger", **kw)


def _prof(market="United States", **kw):
    return SimpleNamespace(company="EVSeller", short="EVS", market=market, tagline="t",
                           recipient="R", value_prop="EV chargers", industries=["Retail"],
                           competitors=[], **kw)


def test_triage_stores_would_buy(monkeypatch):
    a = _art("Acme Retail opens 40 stores")
    reply = {"items": [{"index": 0, "company_name": "Acme Retail", "relationship": "buyer",
                        "event_country": "United States", "would_buy": " fast chargers "}]}
    monkeypatch.setattr(llm, "chat_json", lambda *x, **k: reply)
    rel, cands = brief_builder.triage_triggers([a], _prof())
    assert rel[0] == ("Acme Retail", "buyer")
    assert cands[0].would_buy == "fast chargers"


def _run_customer(monkeypatch, arts, rel, profile=None):
    monkeypatch.setattr(brief_builder, "triage_triggers", lambda t, p: (rel, arts))
    monkeypatch.setattr(brief_builder, "_rank_by_evidence", lambda t: t)
    monkeypatch.setattr(free_research, "enrich_company", lambda *a, **k: {})
    monkeypatch.setattr(brief_builder, "_write_card", lambda *a, **k: {"company": a[1]})
    logs = []
    out = brief_builder.build_customer(arts, profile or _prof(), log=logs.append)
    return out, logs


def test_buyer_without_would_buy_skipped(monkeypatch):
    a = _art("Acme Retail opens stores in Texas, United States"); a.would_buy = ""
    a.event_country = "United States"
    (card, art, _), logs = _run_customer(monkeypatch, [a], {0: ("Acme Retail", "buyer")})
    assert card is None and any("no product it would buy" in l for l in logs)


def test_other_never_chosen(monkeypatch):
    a = _art("Pope Packaging expands plant in United States"); a.would_buy = "x"
    a.event_country = "United States"
    (card, art, _), _ = _run_customer(monkeypatch, [a], {0: ("Pope Packaging", "other")})
    assert card is None and art is None


def test_foreign_unknown_country_skipped(monkeypatch):
    a = _art("Lionfish raises pre-seed in Singapore", "Singapore startup funding")
    a.would_buy = "chargers"; a.event_country = "unknown"
    (card, _, _), logs = _run_customer(monkeypatch, [a], {0: ("Lionfish", "buyer")})
    assert card is None and any("not in the lead's country" in l for l in logs)


def test_in_country_buyer_chosen(monkeypatch):
    a = _art("Acme Retail opens 40 stores across the United States")
    a.would_buy = "fast chargers"; a.event_country = "unknown"
    (card, art, _), _ = _run_customer(monkeypatch, [a], {0: ("Acme Retail", "buyer")})
    assert card["company"] == "Acme Retail" and art is a


def test_low_confidence_skips_competition(monkeypatch):
    monkeypatch.setattr(brief_builder, "build_customer", lambda *a, **k: (None, None, []))

    def boom(*a, **k):
        raise AssertionError("build_competition must not run")
    monkeypatch.setattr(brief_builder, "build_competition", boom)
    monkeypatch.setattr(brief_builder, "triage_and_synthesize_industry", lambda *a, **k: [])
    logs = []
    brief, _ = brief_builder.build_brief([], _prof(profile_confidence="low"), log=logs.append)
    assert not brief.get("competition_outlook")
    assert any("low-confidence" in str(l) for l in logs)
