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


def test_triage_uses_trained_prompt_and_returns_labels_only(monkeypatch):
    import json
    a = _art("Acme Retail opens 40 stores")
    seen = {}

    def fake(system, user, **k):
        seen["s"], seen["u"] = system, user
        return {"items": [{"index": 0, "company_name": "Acme Retail", "relationship": "buyer"}]}
    monkeypatch.setattr(llm, "chat_json", fake)
    rel, cands = brief_builder.triage_triggers([a], _prof())
    assert rel[0] == ("Acme Retail", "buyer")
    trained = json.loads(open("training_data/v4/test_triage.jsonl", encoding="utf-8").readline())
    head = trained["system"].split("SELLER:")[0]
    tail = trained["system"].split("STEP 1")[1]
    assert seen["s"].startswith(head) and seen["s"].endswith(tail)
    assert seen["u"].endswith(trained["user"][trained["user"].index("Return JSON"):])
    assert "| SCORE: " in seen["u"] and "| TRIGGER: " in seen["u"]


def _run_customer(monkeypatch, arts, rel, pick=(0, "United States", "fast chargers"), profile=None):
    monkeypatch.setattr(brief_builder, "triage_triggers", lambda t, p: (rel, arts[:10]))
    monkeypatch.setattr(brief_builder, "_pick_buyer", lambda buyers, p: pick)
    monkeypatch.setattr(brief_builder, "_rank_by_evidence", lambda t: t)
    monkeypatch.setattr(free_research, "enrich_company", lambda *a, **k: {})
    monkeypatch.setattr(brief_builder, "_write_card", lambda *a, **k: {"company": a[1]})
    logs = []
    out = brief_builder.build_customer(arts, profile or _prof(), log=logs.append)
    return out, logs


def test_no_card_when_model_picks_none(monkeypatch):
    a = _art("Acme Retail opens stores in Texas, United States")
    (card, art, _), logs = _run_customer(monkeypatch, [a], {0: ("Acme Retail", "buyer")}, pick=(-1, "", ""))
    assert card is None and any("no buyer fits" in l for l in logs)


def test_other_never_chosen(monkeypatch):
    a = _art("Pope Packaging expands plant in United States")
    (card, art, _), _ = _run_customer(monkeypatch, [a], {0: ("Pope Packaging", "other")})
    assert card is None and art is None


def test_foreign_unknown_country_skipped(monkeypatch):
    a = _art("Lionfish raises pre-seed in Singapore", "Singapore startup funding")
    (card, _, _), logs = _run_customer(monkeypatch, [a], {0: ("Lionfish", "buyer")},
                                       pick=(0, "unknown", "chargers"))
    assert card is None and any("skipped" in l for l in logs)


def test_in_country_buyer_chosen(monkeypatch):
    a = _art("Acme Retail opens 40 stores across the United States")
    (card, art, _), _ = _run_customer(monkeypatch, [a], {0: ("Acme Retail", "buyer")},
                                      pick=(0, "unknown", "fast chargers"))
    assert card["company"] == "Acme Retail" and art is a and a.would_buy == "fast chargers"


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
