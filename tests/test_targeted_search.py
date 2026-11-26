"""Targeted search: when: operator, buyer-name harvest, staffing trigger-query drop.

No network, no LLM: feedparser.parse is monkeypatched.
"""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from urllib.parse import unquote_plus

import pytest

import collect
import config
import tenant
from collect import Article


def _profile(**kw):
    base = dict(company="Multifly", competitors=["Siemens"], competitor_aliases={},
                local_competitor_names={}, trigger_queries=[], industry_queries=[],
                competitor_query_seeds=[], local_queries=[], products=[], market="India",
                short="Multifly", aliases=[])
    base.update(kw)
    return SimpleNamespace(**base)


def _art(title, days=1, bucket="trigger"):
    return Article(title=title, url="http://x/" + title, source="S", bucket=bucket,
                   published=datetime.now(timezone.utc) - timedelta(days=days),
                   fingerprint=title)


def test_when_operator_in_url_only(monkeypatch):
    urls = []

    def fake_parse(url):
        urls.append(url)
        e = SimpleNamespace(title="Acme opens plant - Mint", link="http://a",
                            published_parsed=datetime.now(timezone.utc).timetuple(), summary="")
        return SimpleNamespace(entries=[e], status=200, bozo=False)

    monkeypatch.setattr(collect.feedparser, "parse", fake_parse)
    cutoff = datetime.now(timezone.utc) - timedelta(days=182)
    loc = ("en-IN", "IN", "IN:en")
    out = collect._fetch_one("textile plant India", "trigger", cutoff, loc, 182)
    assert "when:182d" in unquote_plus(urls[-1])
    assert out and out[0].query == "textile plant India"
    collect._fetch_one("textile plant India", "trigger", cutoff, loc)
    assert "when:" not in unquote_plus(urls[-1])


def test_collect_passes_when_days_per_bucket(monkeypatch):
    calls = []
    monkeypatch.setattr(collect, "_fetch_one", lambda q, b, c, loc, w=0: calls.append((b, w)) or [])
    monkeypatch.setattr(config, "SEARCH_WHEN_OPERATOR", True)
    monkeypatch.setattr(config, "HARVEST_BUYERS", 0)
    p = _profile(trigger_queries=["a b"], industry_queries=["c d"], competitors=["Rival"])
    collect.collect(p, config.get_country("India"))
    assert {w for _b, w in calls} == {182}
    calls.clear()
    monkeypatch.setattr(config, "SEARCH_WHEN_OPERATOR", False)
    collect.collect(p, config.get_country("India"))
    assert {w for _b, w in calls} == {0}


HEADLINES = {
    "Oswal Pumps to set up 1.2 GW TOPCon solar cell factory in India": "Oswal Pumps",
    "Home appliances maker Novamax to invest ₹200 crore in new manufacturing facility in DMIC": "Novamax",
    "Shakti Pumps (India) Limited Invests ₹10 Crore in Solar Unit": "Shakti Pumps Limited",
    "Amazon to pump $7 billion in renewable energy projects in India by 2030": "Amazon",
    "Bhagwati Products to invest Rs. 1300 crore in display and precision moulding": "Bhagwati Products",
    "India's Oswal Pumps Plans 1.2 GW TOPCon Cell Plant": "Oswal Pumps",
}


@pytest.mark.parametrize("title,name", list(HEADLINES.items()))
def test_harvest_names(title, name):
    assert collect.harvest_buyers([_art(title)], _profile(), None, 6) == [name]


@pytest.mark.parametrize("title", [
    "Centre clears 31 more ECMS projects; approved investment crosses Rs 69,500 crore",
    "Cabinet Approves ₹2,814 Crore Scheme for rare earth magnets",
    "India Approves Two Electronics Manufacturing Clusters in Tamil Nadu",
    "Tamil Nadu to invest in new clusters",
    "Siemens to invest in new plant",        # profile competitor
    "Multifly to set up new line",           # the seller
])
def test_harvest_rejects(title):
    assert collect.harvest_buyers([_art(title)], _profile(), None, 6) == []


def test_harvest_ranks_by_count_then_recency_and_limits():
    arts = [_art("Alpha Corp to invest big", 5), _art("Beta Works to set up plant", 1),
            _art("Beta Works opens new unit", 2), _art("Gamma Ltd to build plant", 0.5)]
    assert collect.harvest_buyers(arts, _profile(), None, 2) == ["Beta Works", "Gamma Ltd"]
    assert collect.harvest_buyers(arts, _profile(), None, 0) == []


def test_harvest_skips_competitor_bucket_and_already_searched():
    assert collect.harvest_buyers([_art("Zeta Corp to invest", bucket="competitor")], _profile(), None, 6) == []
    p = _profile(trigger_queries=["Zeta Corp expansion"])
    assert collect.harvest_buyers([_art("Zeta Corp to invest")], p, None, 6) == []


def test_collect_second_pass_and_disable(monkeypatch):
    calls = []

    def fake(q, b, c, loc, w=0):
        calls.append(q)
        return [_art("Zeta Corp to invest 100 crore in plant")] if q == "seed" else []

    monkeypatch.setattr(collect, "_fetch_one", fake)
    monkeypatch.setattr(config, "HARVEST_BUYERS", 6)
    p = _profile(trigger_queries=["seed"], competitors=[])
    collect.collect(p, None)
    assert '"Zeta Corp"' in calls and collect.LAST_HARVESTED == ["Zeta Corp"]
    calls.clear()
    monkeypatch.setattr(config, "HARVEST_BUYERS", 0)
    collect.collect(p, None)
    assert calls == ["seed"] and collect.LAST_HARVESTED == []


def test_staffing_trigger_queries_dropped():
    qs = ["pharma plant expansion", "solar module investment", "food unit new line",
          "mining new operation", "textile company hires new executives", "CEO appointed at steel firm"]
    p = _profile(trigger_queries=list(qs))
    ch = tenant.tidy_profile(p, website="", country=None)
    assert p.trigger_queries == qs[:4] and len(ch["staffing_triggers"]) == 2


def test_staffing_kept_when_too_few_remain():
    qs = ["pharma plant expansion", "textile company hires new executives", "steel ceo change"]
    p = _profile(trigger_queries=list(qs))
    tenant.tidy_profile(p, website="", country=None)
    assert p.trigger_queries == qs
