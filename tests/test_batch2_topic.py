"""Report briefs: on-topic matching, topic geo widening, no off-topic fallback."""
from datetime import datetime, timezone
from types import SimpleNamespace

import brief_builder
import collect
import geo
import llm
import score
from collect import Article

NOW = datetime.now(timezone.utc)


def _art(title, summary="", **kw):
    return Article(title=title, url="https://x.example/" + title[:8].replace(" ", ""),
                   source="Wire", published=NOW, summary=summary, bucket="industry", **kw)


def test_topic_subject_drops_years_and_country():
    assert collect.topic_subject("Bridesmaid Gown Market 2025-2032") == "Bridesmaid Gown"
    assert collect.topic_subject("Spain Data Center Market", "ES") == "Data Center"


def test_topic_tokens_drop_digits_and_drop_set():
    assert score.topic_tokens("Spain Data Center 2025") == ["spain", "data", "center"]
    assert score.topic_tokens("Spain Data Center 2025", {"spain"}) == ["data", "center"]


def test_topic_relevance_adjacent_pair():
    toks = score.topic_tokens("Monoclonal Antibody Diagnostic Reagents Market")
    a = _art("New Monoclonal Antibodies Target Conserved Glycoprotein")
    assert score.topic_relevance(a, toks) >= 0.6
    toks = score.topic_tokens("Cloud Seeding Equipment Market")
    assert score.topic_relevance(_art("Yurika launches sovereign cloud service for Australia"), toks) < 0.6
    assert score.topic_relevance(_art("Cloud seeding trial boosts rainfall"), toks) >= 0.6


SWISS = {"name": "Switzerland", "iso2": "CH"}
TOK = score.topic_tokens("Cloud Seeding Equipment Market")


def _scopes(arts):
    return {a.title: a.geo_scope for a in arts}


def test_topic_geo_region_then_global():
    de = _art("Germany expands cloud seeding programme")
    off = _art("Germany opens new car factory")
    kept, scope = geo.filter_topic_for_lead([de, off], SWISS, TOK)
    assert scope == "region" and _scopes(kept) == {de.title: "region"}
    jp = _art("Japan tests cloud seeding drones")
    kept, scope = geo.filter_topic_for_lead([jp, off], SWISS, TOK)
    assert scope == "global" and _scopes(kept) == {jp.title: "global"}


def test_topic_geo_country_enough():
    ch = [_art("Swiss firm trials cloud seeding"), _art("Cloud seeding in Switzerland grows")]
    kept, scope = geo.filter_topic_for_lead(ch, SWISS, TOK)
    assert scope == "country" and {a.geo_scope for a in kept} == {"country"}


def test_build_brief_no_on_topic_skips_section(monkeypatch):
    monkeypatch.setattr(brief_builder, "build_customer", lambda *a, **k: (None, None, []))
    monkeypatch.setattr(brief_builder, "build_competition", lambda *a, **k: (None, None))
    called = []
    monkeypatch.setattr(brief_builder, "triage_and_synthesize_industry",
                        lambda cands, *a, **k: called.append(cands) or [])
    a = _art("Seller's unrelated news"); a.topic_rel = 0.0
    prof = SimpleNamespace(company="S", short="S", market="Global", tagline="t", recipient="R")
    brief, _ = brief_builder.build_brief([a], prof, log=lambda *x: None,
                                         market_topic="Cloud Seeding Equipment Market")
    assert called == [[]] and not brief.get("industry_outlook")


def test_cands_block_location_only_when_outside():
    a, b, c = _art("A"), _art("B", geo_scope="region"), _art("C", geo_scope="country")
    txt = brief_builder._cands_block([a, b, c]).split("\n[")
    assert "LOCATION: outside" not in txt[0]
    assert "LOCATION: outside" in txt[1]
    assert "LOCATION: outside" not in txt[2]


def test_translate_topic_failure_returns_empty(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("down")
    monkeypatch.setattr(llm, "chat_json", boom)
    assert collect.translate_topic("Cloud Seeding", "Polish") == ""
