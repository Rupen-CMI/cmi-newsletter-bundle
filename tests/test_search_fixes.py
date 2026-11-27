"""Search-chain fixes from the 2026-11-26 audit (session_scratch/bench/search_chain_audit.md)."""
from datetime import datetime, timezone
from types import SimpleNamespace

import collect as C
import config
import tenant
from collect import Article

NOW = datetime.now(timezone.utc)


def _art(title, bucket="trigger"):
    return Article(title=title, url="https://x.example/" + str(abs(hash(title))), source="Wire",
                   published=NOW, summary="", bucket=bucket, fingerprint=str(hash(title)))


def _profile(**kw):
    base = dict(company="SellerCo", short="SellerCo", recipient="r", tagline="t",
                value_prop="sells widgets", industries=["Manufacturers"], competitors=[],
                industry_keywords={}, trigger_queries=[], industry_queries=[])
    base.update(kw)
    return tenant.Profile.from_dict(base)


# ── 1. story merging works for every script ──────────────────────────────────
def test_non_latin_headlines_are_kept_and_not_merged_on_a_year():
    a = _art("HDC현대산업개발, 천안 부성6지구서 고소")
    b = _art("HDC현대산업개발, 용산 정비창 개발 착수")
    c = _art("السفر الفاخر في دبي 2026")
    d = _art("افتتاح منتجع جديد في دبي 2026")
    out = C._collapse_near_duplicates([a, b, c, d])
    assert len(out) == 4


def test_latin_duplicates_still_merge():
    a = _art("Spain relaxes initial proposal for data-center rules")
    b = _art("Spain relaxes initial proposal for data center rules")
    assert len(C._collapse_near_duplicates([a, b])) == 1


# ── 2. competitor queries: quoted names, no "rival + our product", no dead aliases ─
def test_rival_queries_quoted_and_dead_aliases_skipped():
    p = _profile(competitors=["HDC Hyundai Development Company"],
                 competitor_aliases={"HDC Hyundai Development Company": ["HDC현대산업개발", "HDC"]},
                 products=[{"name": "MAILab"}])
    qs = [q for q, *_ in C.plan_searches(p, config.get_country("South Korea"))[0]]
    assert '"HDC Hyundai Development Company" Korea' in qs
    # "HDC" is the name shortened (same results, audit) and the Hangul form returns nothing
    # on the English edition: both skipped
    assert not any("MAILab" in q or "현대" in q or q == '"HDC" Korea' for q in qs)


def test_rival_query_uses_product_category_to_disambiguate():
    p = _profile(competitors=["FLO", "ChargePoint"], product_category="EV charging")
    qs = [q for q, *_ in C.plan_searches(p, config.get_country("Canada"))[0]]
    assert '"FLO" EV charging Canada' in qs
    assert '"FLO" Canada' not in qs


def test_empty_category_rival_query_falls_back_to_plain_name(monkeypatch):
    p = _profile(company="QuantumEV", competitors=["FLO"], product_category="EV charging")
    calls = []

    def fake(query, bucket, cutoff, locale, when_days=0):
        calls.append(query)
        if query == '"FLO" EV charging Canada':
            return []                       # too specific: nothing
        if query == '"FLO" Canada':
            return [_art("FLO opens new EV charger factory", bucket="competitor")]
        return []

    monkeypatch.setattr(C, "_fetch_one", fake)
    arts = C.collect(p, config.get_country("Canada"))
    assert '"FLO" Canada' in calls
    assert any(a.bucket == "competitor" and "FLO" in a.title for a in arts)


def test_category_rival_query_with_results_is_not_retried(monkeypatch):
    p = _profile(company="QuantumEV", competitors=["FLO"], product_category="EV charging")
    calls = []

    def fake(query, bucket, cutoff, locale, when_days=0):
        calls.append(query)
        if query == '"FLO" EV charging Canada':
            return [_art("FLO launches home charger", bucket="competitor")]
        return []

    monkeypatch.setattr(C, "_fetch_one", fake)
    C.collect(p, config.get_country("Canada"))
    assert '"FLO" Canada' not in calls


# ── 3. local queries go to a bucket by event word ────────────────────────────
def test_local_queries_split_by_event_word():
    p = _profile(local_queries=["Börsengang Schweiz", "Privatbank Markt", "Familienunternehmen"])
    jobs, _ = C.plan_searches(p, config.get_country("Switzerland"))
    buckets = {q: b for q, b, _, tag in jobs if tag}
    assert buckets["Börsengang Schweiz"] == "trigger"
    assert buckets["Privatbank Markt"] == "industry"


# ── 4. harvested names ───────────────────────────────────────────────────────
def test_harvest_rejects_places_fragments_and_keeps_real_names():
    arts = [_art(t) for t in (
        "South Korea plans $100m investment in US lithium project",
        "7 Best Retail Stocks for 2026 and How to Invest",
        "ICE Arrests Man in Boston Days After He Wins $1 Million Lottery Prize",
        "Türkiye plans new rail line",
        "Tokyo Tatemono plans Shibuya redevelopment",
        "Nestlé Purina invests €55 million in Italian plant",
        "Purina invests in new line",
    )]
    names = C.harvest_buyers(arts, _profile(), config.get_country("Japan"), 10)
    assert "Tokyo Tatemono" in names
    assert not {"South Korea", "How", "Boston Days After He", "Türkiye"} & set(names)
    assert sum("Purina" in n for n in names) == 1


# ── 6. topic searches ────────────────────────────────────────────────────────
def test_topic_acronyms_and_fewer_endings():
    assert C.topic_subject("Phytochemical Api Market") == "Phytochemical API"
    assert C.topic_subject("Gcc Outbound Tourism Market") == "GCC Outbound Tourism"
    assert C.topic_queries("Cable Glands Market") == ["Cable Glands", "Cable Glands industry",
                                                       "Cable Glands expansion investment"]


def test_bing_market_follows_the_lead_country():
    assert C._bing_market(config.get_country("Spain")) == "es-ES"
    assert C._bing_market(config.get_country("United Kingdom")) == "en-GB"
    assert C._bing_market(config.get_country("Global")) == ""


# ── 7. short country word, GCC neighbours ────────────────────────────────────
def test_short_country_word_and_gcc_neighbour_kept():
    p = _profile(trigger_queries=["train order"], industry_queries=["Saudi outbound travel"],
                 value_prop="hotel representation across the GCC")
    tr = [q for q, *_ in C.plan_searches(p, config.get_country("Turkey"))[0]]
    assert "train order Türkiye" in tr
    ae = [q for q, *_ in C.plan_searches(p, config.get_country("United Arab Emirates"))[0]]
    assert any(q.startswith("Saudi outbound travel") for q in ae)


# ── 5. rival matching reads non-Latin names ──────────────────────────────────
def test_non_latin_rival_name_matches():
    assert tenant.name_matches("엠디엠그룹", "[디벨로퍼 인사이드] 엠디엠그룹, 홈플러스 개발")
    assert tenant.name_matches("東京センチュリー", "東京センチュリーの子会社が航空機を発注")


# ── one fetch per search per run ─────────────────────────────────────────────
def test_repeated_search_is_fetched_once(monkeypatch):
    calls = []

    def fake(query, bucket, cutoff, locale, when_days=0):
        calls.append(query)
        return [Article(title="Acme opens plant", url="u", source="s", published=NOW,
                        query=query, bucket=bucket)]
    monkeypatch.setattr(C, "_fetch_one_uncached", fake)
    C._FETCH_CACHE.clear()
    loc = ("en-GB", "GB", "GB:en")
    a = C._fetch_one("glasshouse expansion UK", "trigger", NOW, loc, 182)
    b = C._fetch_one("glasshouse expansion UK", "competitor", NOW, loc, 182)
    assert calls == ["glasshouse expansion UK"]
    assert a[0].bucket == "trigger" and b[0].bucket == "competitor" and a[0] is not b[0]
    C._FETCH_CACHE.clear()
