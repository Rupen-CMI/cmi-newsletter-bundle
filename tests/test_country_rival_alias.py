"""Rival presence check, outlet->country map, model event_country, competitor aliases,
and the relevance-gated fallback pool. No network, no LLM, no real DB."""
import sqlite3
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest import mock

import pytest

import brief_builder as bb
import config
import db
import geo
import rivals
import score
import source_country
import tenant
from collect import Article

KR = config.get_country("South Korea")


def _art(title, *, source="Wire", bucket="competitor", summary="", url=None, **kw):
    a = Article(title=title, url=url or f"http://x/{abs(hash(title))}", source=source,
                published=datetime.now(timezone.utc), summary=summary, query="q",
                bucket=bucket, fingerprint=title)
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def _profile(**kw):
    base = dict(company="Seller", short="Seller", value_prop="widgets for labs.",
                industries=["Pharma"], competitors=["Societe Acme Instruments"],
                market="South Korea", lead_type="seller", buyer_functions=[], recipient="Sam",
                products=[{"name": "Zeta Widget", "application": "x"}],
                competitor_aliases={"Societe Acme Instruments": ["Acme Instruments", "ACMI"]},
                local_competitor_names={})
    base.update(kw)
    return SimpleNamespace(**base)


@pytest.fixture(autouse=True)
def _clean():
    rivals.reset_memory()
    source_country.reset_memory()
    yield
    rivals.reset_memory()
    source_country.reset_memory()


@pytest.fixture
def mem_db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    db.init_db(conn)
    return conn


# -- 1. rival presence ---------------------------------------------------------
def test_no_country_rival_kept_when_cached_active(mem_db):
    db.put_rival_presence(mem_db, "Societe Acme Instruments", "KR", True, 5)
    a = _art("Acme Instruments posts record quarterly results")
    kept, _ = geo.filter_for_lead([a], KR, profile=_profile(), conn=mem_db,
                                  presence_fetch=mock.Mock(side_effect=AssertionError("no fetch")))
    assert a in kept


def test_no_country_rival_dropped_when_cached_inactive(mem_db):
    db.put_rival_presence(mem_db, "Societe Acme Instruments", "KR", False, 0)
    a = _art("Acme Instruments posts record quarterly results")
    kept, _ = geo.filter_for_lead([a], KR, profile=_profile(), conn=mem_db)
    assert a not in kept


def test_other_country_rival_article_dropped_even_if_active(mem_db):
    db.put_rival_presence(mem_db, "Societe Acme Instruments", "KR", True, 5)
    a = _art("Acme Instruments opens plant in Germany")
    kept, _ = geo.filter_for_lead([a], KR, profile=_profile(), conn=mem_db)
    assert a not in kept


def test_rule_only_for_competitor_bucket(mem_db):
    db.put_rival_presence(mem_db, "Societe Acme Instruments", "KR", True, 5)
    a = _art("Acme Instruments posts record quarterly results", bucket="trigger")
    kept, _ = geo.filter_for_lead([a], KR, profile=_profile(), conn=mem_db)
    assert a not in kept


def test_presence_counts_alias_hits_and_caches(mem_db):
    def fetch(query, locale):
        if "Korea" in query:
            return [_art("ACMI wins Korea hospital tender", url="u1"),
                    _art("Acme Instruments expands in South Korea", url="u2"),
                    _art("Acme Instruments expands in South Korea", url="u2"),   # duplicate
                    _art("Acme Instruments in Japan", url="u3")]                 # wrong country
        return []
    f = mock.Mock(side_effect=fetch)
    p = _profile()
    assert rivals.rival_active(p, "Societe Acme Instruments", KR, "KR", conn=mem_db, fetch=f)
    row = db.get_rival_presence(mem_db, "Societe Acme Instruments", "KR")
    assert row["active"] is True and row["hits"] == 2
    calls = f.call_count
    rivals.reset_memory()                                   # force the DB path
    assert rivals.rival_active(p, "Societe Acme Instruments", KR, "KR", conn=mem_db, fetch=f)
    assert f.call_count == calls                            # served from the table


def test_presence_inactive_under_two_hits_and_in_memory_without_conn():
    f = mock.Mock(return_value=[_art("Acme Instruments expands in South Korea", url="u1")])
    p = _profile()
    assert not rivals.rival_active(p, "Societe Acme Instruments", KR, "KR", conn=None, fetch=f)
    n = f.call_count
    assert not rivals.rival_active(p, "Societe Acme Instruments", KR, "KR", conn=None, fetch=f)
    assert f.call_count == n


def test_presence_ttl_expires(mem_db):
    db.put_rival_presence(mem_db, "x", "KR", True, 3)
    mem_db.execute("UPDATE rival_presence SET checked_at='2020-01-01T00:00:00+00:00'")
    assert db.get_rival_presence(mem_db, "x", "KR") is None


# -- 2a. outlet -> country map -------------------------------------------------
def _src_arts(src, n, text="Seoul shipyard orders cranes"):
    return [_art(f"{text} {i}", source=src, bucket="industry") for i in range(n)]


def test_source_learned_at_threshold_and_used_by_geo(mem_db):
    source_country.learn(_src_arts("Daily Seoul Gazette", 5), mem_db)
    assert source_country.learned_map(mem_db) == {"daily seoul gazette": "KR"}
    assert db.get_source_countries(mem_db)["daily seoul gazette"]["n"] == 5
    nogeo = _art("Cranes order lifts shipyard", source="Daily Seoul Gazette", bucket="industry")
    kept, _ = geo.filter_for_lead([nogeo], KR, conn=mem_db)
    assert nogeo in kept


def test_source_not_learned_below_threshold():
    source_country.learn(_src_arts("Tiny Paper", 4), None)
    assert source_country.learned_map() == {}
    mixed = _src_arts("Mixed Wire", 3) + [_art("Tokyo port news", source="Mixed Wire"),
                                          _art("Tokyo port news 2", source="Mixed Wire")]
    source_country.learn(mixed, None)
    assert "mixed wire" not in source_country.learned_map()   # 3/5 = 60% < 70%


def test_source_of_other_country_treated_as_that_country():
    source_country.learn(_src_arts("Daily Sentinel", 6, text="Brazilian factory"), None)
    assert source_country.learned_map() == {"daily sentinel": "BR"}
    a = _art("Factory orders rise", source="Daily Sentinel", bucket="trigger")
    kept, _ = geo.filter_for_lead([a], KR)
    assert a not in kept


def test_source_map_learns_incrementally(mem_db):
    source_country.learn(_src_arts("Inc Paper", 3), mem_db)
    assert source_country.learned_map(mem_db) == {}
    source_country.reset_memory()
    source_country.learn(_src_arts("Inc Paper", 3), mem_db)
    assert source_country.learned_map(mem_db) == {"inc paper": "KR"}


# -- 2b. event_country from triage ---------------------------------------------
def test_triage_stores_event_country_and_customer_skips_foreign():
    p = _profile(market="South Korea", industries=["Pharma"], competitors=[])
    cands = [_art("Alpha plant expands", bucket="trigger", trigger_type="expansion"),
             _art("Beta plant expands", bucket="trigger", trigger_type="expansion")]
    items = [{"index": 0, "company_name": "Alpha Foods", "relationship": "buyer",
              "event_country": "Germany", "would_buy": "pharma packaging"},
             {"index": 1, "company_name": "Beta Foods", "relationship": "buyer",
              "event_country": "South Korea", "would_buy": "pharma packaging"}]
    picked = {}
    with mock.patch.object(bb.llm, "chat_json", return_value={"items": items}), \
            mock.patch.object(bb, "_write_card",
                              side_effect=lambda a, n, *r: picked.update(n=n) or {"company_name": n}), \
            mock.patch.object(bb.free_research, "enrich_company", return_value={}), \
            mock.patch.object(bb.tenant, "_overrides_for", return_value={}):
        bb.build_customer(cands, p, log=lambda *_: None)
    assert cands[0].event_country == "Germany"
    assert picked["n"] == "Beta Foods"


def test_unknown_event_country_not_skipped():
    p = _profile(market="South Korea", competitors=[])
    a = _art("Alpha plant expands", bucket="trigger")
    a.event_country = "unknown"
    assert not bb._foreign_event(a, p)
    a.event_country = "South Korea"
    assert not bb._foreign_event(a, p)
    a.event_country = "Germany"
    assert bb._foreign_event(a, p)


def test_industry_foreign_event_picks_next():
    p = _profile(market="South Korea")
    c0 = _art("Berlin story", bucket="industry")
    c1 = _art("Seoul story", bucket="industry")
    c2 = _art("Seoul story two", bucket="industry")
    replies = [{"best_index": 0, "headline": "H0", "body_html": "B0", "event_country": "Germany"},
               {"best_index": 0, "headline": "H1", "body_html": "B1", "event_country": "South Korea"}]
    with mock.patch.object(bb, "_chat_guarded", side_effect=replies):
        out = bb.triage_and_synthesize_industry([c0, c1, c2], p)
    assert out[0]["headline"] == "H1" and out[0]["source_url"] == c1.url


# -- 3. aliases ----------------------------------------------------------------
def test_old_profile_dict_loads_without_aliases():
    d = {"company": "S", "short": "S", "recipient": "x", "tagline": "t", "value_prop": "v",
         "industries": [], "competitors": ["Acme"], "industry_keywords": {},
         "trigger_queries": [], "industry_queries": []}
    assert tenant.Profile.from_dict(d).competitor_aliases == {}


def test_aliases_parsed_and_merged_with_local_names():
    out = tenant._parse_competitor_aliases(
        {"Societe Acme Instruments": ["Acme Instruments", "ACMI", "societe acme instruments"],
         "Unlisted Co": ["X"]},
        ["Societe Acme Instruments"], {"Societe Acme Instruments": "local-form"})
    assert out == {"Societe Acme Instruments": ["Acme Instruments", "ACMI", "local-form"]}


def test_alias_used_in_attribution_english_headline_matches_french_rival():
    p = _profile()
    a = _art("Acme Instruments unveils new analyser")
    judged = []
    with mock.patch.object(bb, "_competitive_move_check",
                           side_effect=lambda art, comp, prof: judged.append(comp) or True), \
            mock.patch.object(bb, "_write_card", side_effect=lambda a_, n, *r: {"company_name": n}), \
            mock.patch.object(bb.llm, "chat_json", return_value={}),             mock.patch.object(bb.free_research, "enrich_company", return_value={}):
        out = bb.build_competition([a], [], p, log=lambda *_: None)
    assert judged == ["Societe Acme Instruments"]
    assert out[0]["company_name"] == "Societe Acme Instruments"


def test_short_alias_needs_word_boundary():
    p = _profile()
    assert tenant.rival_named_in(p, "Societe Acme Instruments", "ACMI profits climb")
    assert not tenant.rival_named_in(p, "Societe Acme Instruments", "acmicron announces chip")


def test_is_self_ignores_competitor_aliases():
    p = _profile(company="Seller Corp", short="Seller")
    assert not bb.is_self("Acme Instruments", p)


# -- 4. fallback pool ----------------------------------------------------------
def test_fallback_excludes_low_relevance_and_unsafe():
    good = _art("Relevant", bucket="trigger", seller_rel=0.25)
    nomodel = _art("No model score", bucket="trigger", seller_rel=None)
    low = _art("Unrelated national headline", bucket="trigger", seller_rel=0.02)
    assert score.fallback_pool([good, nomodel, low]) == [good, nomodel]
    with mock.patch.object(score, "is_unsafe_to_cite", side_effect=lambda a: a is good):
        assert score.fallback_pool([good, nomodel, low]) == [nomodel]
