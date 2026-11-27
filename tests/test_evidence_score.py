"""Evidence scores (evidence_score.py): relevancy and recency per article, annotation only.

No network, no LLM; the embedding model is switched off so the keyword fallback is deterministic.
"""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

import brief_builder as bb
import evidence_score as es
import relevance
import score
from collect import Article


@pytest.fixture(autouse=True)
def _no_model():
    relevance.set_encoder(False)
    yield
    relevance.reset()


def _profile(**kw):
    base = dict(company="Cook Medical", short="Cook", market="Ireland",
                competitors=["Stryker", "Medtronic"], value_prop="sells surgical devices to hospitals",
                industries=["Hospitals", "Medical Device Manufacturers"],
                industry_keywords={"Hospitals": ["hospital", "surgeon"]},
                products=[], competitor_aliases={}, local_competitor_names={})
    base.update(kw)
    return SimpleNamespace(**base)


def _art(title, bucket, query, days=1, summary=""):
    return Article(title=title, url=f"http://x/{title}", source="S",
                   published=datetime.now(timezone.utc) - timedelta(days=days),
                   summary=summary, query=query, bucket=bucket, fingerprint=title)


@pytest.mark.parametrize("bucket,half", [("trigger", 30), ("industry", 45), ("competitor", 60), ("market", 30)])
def test_recency_half_life(bucket, half):
    assert es.recency(bucket, 0) == pytest.approx(1.0)
    assert es.recency(bucket, half) == pytest.approx(0.5)
    assert es.recency(bucket, 2 * half) == pytest.approx(0.25)


def test_annotates_recency_and_age():
    a = _art("Hospital opens new surgical wing in Dublin", "trigger", "hospital expansion Ireland", days=30)
    es.score_evidence([a], _profile())
    assert a.ev_age_days == pytest.approx(30, abs=0.2)
    assert a.ev_recency == pytest.approx(0.5, abs=0.01)


@pytest.mark.parametrize("bucket,query,good,bad", [
    ("trigger", "hospital expansion Ireland",
     "Dublin hospital announces expansion with new surgical wing", "Football club wins league title"),
    ("industry", "hospital procurement regulation Ireland",
     "New hospital procurement regulation reshapes buying", "Celebrity chef opens a bakery"),
    ("competitor", "Stryker Ireland",
     "Stryker launches new surgical robot", "Local council repaves the main road"),
])
def test_on_use_case_beats_off_topic(bucket, query, good, bad):
    g, b = _art(good, bucket, query), _art(bad, bucket, query)
    es.score_evidence([g, b], _profile())
    assert g.ev_relevancy > b.ev_relevancy
    assert 0.0 <= b.ev_relevancy <= g.ev_relevancy <= 1.0


def test_reason_string_content():
    c = _art("Stryker launches new surgical robot", "competitor", "Stryker robot Ireland")
    t = _art("Dublin hospital secures funding for new wing", "trigger", "hospital funding Ireland")
    n = _art("Weather turns cold", "competitor", "Stryker robot Ireland")
    es.score_evidence([c, t, n], _profile())
    assert "names rival Stryker" in c.ev_reason and "query match" in c.ev_reason
    assert "trigger: funding" in t.ev_reason
    assert "no rival named" in n.ev_reason


def test_country_words_not_counted_in_query_match():
    a = _art("Hospital expansion announced", "trigger", "hospital expansion Ireland")
    es.score_evidence([a], _profile(market="Global"))
    assert "query match 2/2" in a.ev_reason


def test_score_all_unchanged_by_annotation(monkeypatch):
    prof = _profile()
    arts = [_art("Stryker launches new surgical robot", "competitor", "Stryker Ireland", days=3),
            _art("Dublin hospital secures funding for new wing", "trigger", "hospital funding Ireland", days=1),
            _art("Weather turns cold", "industry", "hospital Ireland", days=40)]
    ref = [Article(**{k: getattr(a, k) for k in ("title", "url", "source", "published", "summary",
                                                 "query", "bucket", "fingerprint")}) for a in arts]
    monkeypatch.setattr(es, "score_evidence", lambda articles, profile: articles)
    base = score.score_all(ref, prof)
    monkeypatch.undo()
    out = score.score_all(arts, prof)
    assert [a.title for a in out] == [a.title for a in base]
    assert [a.composite for a in out] == [a.composite for a in base]
    assert all(a.ev_reason for a in out)


def test_never_raises(monkeypatch):
    def boom(_t):
        raise RuntimeError("x")
    monkeypatch.setattr(relevance, "embed", boom)
    a = _art("Stryker launches new surgical robot", "competitor", "Stryker Ireland")
    assert es.score_evidence([a], _profile()) == [a]
    assert es.score_evidence([], _profile()) == []


def test_cands_block_and_single_signal_carry_scores():
    a = _art("Stryker launches new surgical robot", "competitor", "Stryker Ireland", days=3)
    es.score_evidence([a], _profile())
    block = bb._cands_block([a])
    assert "FIT:" in block and "AGE: 3 days" in block
    assert "SCORE:" not in block
    assert "FIT (0 to 1)" in bb._SCORE_NOTE


# Root-cause fixes from the manual review
def test_event_regex_trigger_and_competitor_move():
    t = _art("Navitas Solar Announces INR 10,000 Crore Investment", "trigger", "solar manufacturing")
    c = _art("Rockwell Automation ramps up India manufacturing capacity", "competitor", "Rockwell Automation")
    es.score_evidence([t, c], _profile(competitors=["Rockwell Automation"]))
    assert "no buying event" not in t.ev_reason
    assert "no move" not in c.ev_reason


def test_rival_matches_country_suffixed_and_first_token():
    p = _profile(competitors=["Siemens Industry India", "Rockwell Automation India"], market="India")
    assert es._named_rival(p, "Siemens launches Eigen Engineering Agent in India") == "Siemens Industry India"
    assert es._named_rival(p, "Cisco and Rockwell Automation launch push") == "Rockwell Automation India"
    p2 = _profile(competitors=["General Electric India"])
    assert es._named_rival(p2, "General Motors expands") == ""


@pytest.mark.parametrize("title,why", [
    ("Asia-Pacific Industrial Automation Market Size, Share and Industry Trends 2035", "SEO/report"),
    ("India Home Automation System Market Size, Share,Trends, Growth Analysis Report, 2032", "SEO/report"),
    ("Stocks to buy or hold: ABB India, Siemens", "stock noise"),
    ("Honeywell forms death cross on the charts", "stock noise"),
])
def test_junk_capped(title, why):
    a = _art(title, "competitor", "ABB India")
    es.score_evidence([a], _profile(competitors=["ABB India"]))
    assert a.ev_relevancy <= 0.15 and f"capped: {why}" in a.ev_reason


def test_industry_term_right_boundary():
    a = _art("Factory automation spending rises", "industry", "automation")
    es.score_evidence([a], _profile(industry_keywords={"Auto": ["auto"]}, industries=[]))
    assert "no buyer-industry term" in a.ev_reason


def test_trigger_half_life_is_30():
    assert es.recency("trigger", 30) == pytest.approx(0.5)


# ── scorer fixes: buying-event words, stemming, stock/explainer, approval caps ──
@pytest.mark.parametrize("title", [
    "Oswal Pumps to set up 1.2 GW TOPCon solar cell factory in India",
    "India's Oswal Pumps Plans 1.2 GW TOPCon Cell Plant",
    "Amazon to pump $7 billion in renewable energy projects in India by 2030",
    "India Approves Two Electronics Manufacturing Clusters in Tamil Nadu",
    "Centre clears 31 more ECMS projects; approved investment crosses Rs 69,500 crore",
    "Wheels India gets nod to raise 400cr",
])
def test_trigger_event_words_score_full_bucket_check(title):
    a = _art(title, "trigger", "solar module manufacturer investment")
    chk, why = es._bucket_check(a, _profile(), set(), title.lower())
    assert chk == 1.0, why


def test_company_name_pumps_is_not_an_event():
    a = _art("Oswal Pumps shares slide", "trigger", "q")
    assert es._bucket_check(a, _profile(), set(), a.title.lower())[0] == 0.0


def test_query_terms_stemmed():
    a = _art("Wheels India gets nod to raise 400cr", "trigger", "auto component supplier raises funding India")
    es.score_evidence([a], _profile(market="India"))
    assert "query match 1/5" in a.ev_reason       # "raises" now finds "raise"
    assert es._stem_rx("funding").search("fundraising round")
    assert es._stem_rx("raises").search("will raise money")


def test_report_page_capped_but_approval_headline_is_not():
    rep = _art("India Canned Motor Pumps Market Size, Share,Trends, Growth Analysis Report, 2030", "trigger", "q")
    ok = _art("India Approves USD-297 Million Critical Mineral Processing Scheme", "trigger", "q")
    ok.source = "Spherical Insights"       # score.is_unsafe_to_cite rejects this source outright
    ok2 = _art("The Indian government has approved two new Electronics Manufacturing Clusters", "trigger", "q")
    ok2.source = "Instagram"
    assert es._junk(rep) == "SEO/report page"
    assert es._junk(ok) == "" and es._junk(ok2) == ""
    assert score.is_unsafe_to_cite(ok) and score.is_unsafe_to_cite(ok2)    # score.py unchanged


def test_stock_comparison_and_explainer():
    s = _art("Siemens Limited vs GE Vernova: Which MNC Stock Is Better", "trigger", "q")
    assert es._junk(s) == "stock noise"
    for t in ("Why investors are looking beyond solar modules", "How PLI is helping Indian brands go global"):
        a = _art(t, "trigger", "q")
        assert es._bucket_check(a, _profile(), set(), t.lower())[0] == 0.0


def test_topic_fit_uses_bare_query_when_better(monkeypatch):
    import numpy as np
    # rows: [use-case text, bare query, article]; article matches only the bare query
    monkeypatch.setattr(relevance, "embed", lambda texts: np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]))
    a = _art("Acme to set up plant", "trigger", "plant setup")
    es.score_evidence([a], _profile())
    assert "topic fit 1.00" in a.ev_reason
