"""Local-language search, embedding relevance and event dedupe. No network, no LLM, no
real embedding model (a fake encoder stands in)."""
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

import collect
import config
import relevance
import score
import tenant

NOW = datetime.now(timezone.utc)

# Concept lexicon for the fake encoder: words in either language map to one dimension, so
# the same story in English and Polish gets (near-)identical vectors.
_CONCEPTS = [
    {"lidl", "store", "sklep", "krakow", "krakowie", "opens", "otwiera"},       # 0 retail opening
    {"poultry", "drob", "eksport", "export"},                                  # 1 poultry
    {"kospi", "stock", "market", "gielda", "shares"},                          # 2 markets
    {"shop", "fittings", "fixtures", "retail", "retailers", "wyposazenie"},    # 3 seller topic
    {"army", "defence", "wojsko", "ammo"},                                     # 4 defence
]


class FakeEncoder:
    def embed(self, texts):
        for t in texts:
            toks = {w.strip(".,:;'\"()").lower() for w in t.split()}
            v = np.zeros(len(_CONCEPTS) + 1, dtype="float32")
            for i, group in enumerate(_CONCEPTS):
                v[i] = len(toks & group)
            v[-1] = 0.01                      # never a zero vector
            yield v


@pytest.fixture(autouse=True)
def _reset_relevance():
    yield
    relevance.reset()


@pytest.fixture
def fake_model():
    relevance.set_encoder(FakeEncoder())


def _art(title, bucket="trigger", source="Local Daily", summary="", published=None, url=None):
    return collect.Article(title=title, url=url or f"https://x.test/{abs(hash(title))}",
                           source=source, published=published or NOW - timedelta(days=2),
                           summary=summary, bucket=bucket, query="q",
                           fingerprint=collect._fingerprint(title, title))


def _profile(**kw):
    base = dict(company="Acme", short="AC", recipient="Sam", tagline="t",
                value_prop="Acme sells shop fittings and fixtures to retailers.",
                industries=["Retail"], competitors=["Rival One"],
                industry_keywords={"Retail": ["retail"]},
                trigger_queries=["retailer expansion"], industry_queries=["retail trends"],
                market="Poland")
    base.update(kw)
    return tenant.Profile(**base)


# ── 1. profile ───────────────────────────────────────────────────────────────
class TestProfileLocalFields:
    def test_old_dict_loads_with_defaults(self):
        old = {"company": "Acme", "short": "AC", "recipient": "Sam", "tagline": "t",
               "value_prop": "v", "industries": [], "competitors": [],
               "industry_keywords": {}, "trigger_queries": [], "industry_queries": []}
        p = tenant.Profile.from_dict(old)
        assert p.local_language == "" and p.local_queries == [] and p.local_competitor_names == {}

    def test_from_company_asks_for_and_parses_local_fields(self, monkeypatch):
        monkeypatch.setattr(config, "LOCAL_NEWS_EDITIONS",
                            {"Poland": ("pl", "PL", "PL:pl")}, raising=False)
        monkeypatch.setattr(config, "LOCAL_LANGUAGE_NAMES", {"Poland": "Polish"}, raising=False)
        seen = {}

        def fake_chat(system, user, **kw):
            if "news-search queries" in system:
                seen["qsys"], seen["quser"] = system, user
                return {"trigger_queries": ["retail expansion"],
                        "industry_queries": ["retail trends"],
                        "local_queries": ["nowy sklep", "wyposażenie sklepów"]}
            seen["user"] = user
            return {"value_prop": "Sells shop fittings.", "target_industries": ["Retail"],
                    "competitors": [{"name": "Rival One", "their_product": "shop fittings"}],
                    "industry_keywords": {"Retail": ["retail"]},
                    "local_competitor_names": {"rival one": "Rywal Jeden", "Ghost": "Duch"}}

        monkeypatch.setattr(tenant.llm, "chat_json", fake_chat)
        p = tenant.Profile.from_company("Acme Fixtures", country_name="Poland")
        assert "Polish" in seen["user"] and "local_queries" not in seen["user"]
        assert "Polish" in seen["quser"] and "local_queries" in seen["quser"]
        assert "2 to 4 words" in seen["quser"]
        assert p.local_language == "Polish"
        assert p.local_queries == ["nowy sklep", "wyposażenie sklepów"]
        # only competitors that are actually in the final list are kept, keyed by English name
        assert p.local_competitor_names == {"Rival One": "Rywal Jeden"}

    def test_english_market_has_no_local_fields(self, monkeypatch):
        monkeypatch.setattr(config, "LOCAL_NEWS_EDITIONS", {"Canada": None}, raising=False)
        monkeypatch.setattr(config, "LOCAL_LANGUAGE_NAMES", {}, raising=False)
        seen = {}
        monkeypatch.setattr(tenant.llm, "chat_json",
                            lambda s, u, **kw: seen.setdefault("user", u) and {
                                "local_queries": ["x"], "target_industries": ["Retail"]})
        p = tenant.Profile.from_company("Acme Fixtures", country_name="Canada")
        assert "local_queries" not in seen["user"]
        assert p.local_language == "" and p.local_queries == []


# ── 2. local pass in collect() ───────────────────────────────────────────────
class TestLocalPass:
    @pytest.fixture
    def jobs(self, monkeypatch):
        monkeypatch.setattr(config, "LOCAL_NEWS_EDITIONS",
                            {"Poland": ("pl", "PL", "PL:pl")}, raising=False)
        calls = []

        def fake_fetch(query, bucket, cutoff, locale, when_days=0):
            calls.append((query, bucket, tuple(locale), cutoff))
            return []

        monkeypatch.setattr(collect, "_fetch_one", fake_fetch)
        return calls

    def test_local_jobs_use_local_edition_and_queries(self, jobs):
        prof = _profile(local_language="Polish",
                        local_queries=["nowy sklep", "otwarcie galerii", "wyposażenie sklepów",
                                       "centra handlowe"],
                        local_competitor_names={"Rival One": "Rywal Jeden"})
        collect.collect(prof, config.get_country("Poland"))
        en = [j for j in jobs if j[2] != ("pl", "PL", "PL:pl")]
        loc = {j[0]: j for j in jobs if j[2] == ("pl", "PL", "PL:pl")}
        assert en, "English pass still runs"
        assert set(loc) == {"nowy sklep", "otwarcie galerii", "wyposażenie sklepów",
                            "centra handlowe", "Rywal Jeden"}
        assert loc["nowy sklep"][1] == "trigger" and loc["otwarcie galerii"][1] == "trigger"
        assert loc["wyposażenie sklepów"][1] == "industry"
        assert loc["Rywal Jeden"][1] == "competitor"
        # competitor jobs keep the long competitor cutoff
        assert loc["Rywal Jeden"][3] < loc["nowy sklep"][3]

    def test_local_query_naming_foreign_country_is_dropped(self, jobs):
        prof = _profile(local_language="Polish", local_queries=["rynek Niemcy Germany", "nowy sklep"])
        collect.collect(prof, config.get_country("Poland"))
        loc_q = [j[0] for j in jobs if j[2] == ("pl", "PL", "PL:pl")]
        assert "nowy sklep" in loc_q and "rynek Niemcy Germany" not in loc_q

    def test_no_local_pass_for_english_market(self, jobs, monkeypatch):
        monkeypatch.setattr(config, "LOCAL_NEWS_EDITIONS", {"Poland": None}, raising=False)
        collect.collect(_profile(local_queries=["nowy sklep"]), config.get_country("Poland"))
        assert all(j[0] != "nowy sklep" for j in jobs)

    def test_collect_topic_local_pass_only_with_translation(self, jobs, monkeypatch):
        monkeypatch.setattr(collect, "_fetch_bing", lambda *a, **k: [])
        poland = config.get_country("Poland")
        collect.collect_topic("Metal Furniture Market", poland)
        assert all(j[2] != ("pl", "PL", "PL:pl") for j in jobs)     # no translation -> skipped
        jobs.clear()
        collect.collect_topic("Metal Furniture Market", poland, local_topic="meble metalowe")
        assert any(j[0] == "meble metalowe" and j[2] == ("pl", "PL", "PL:pl") for j in jobs)


# ── 2b. blocklist ────────────────────────────────────────────────────────────
class TestBlocklist:
    def test_blocklisted_source_dropped_in_both_passes(self, monkeypatch):
        monkeypatch.setattr(config, "SOURCE_BLOCKLIST", ["indexbox", "stocktickers.example"],
                            raising=False)
        monkeypatch.setattr(config, "LOCAL_NEWS_EDITIONS",
                            {"Poland": ("pl", "PL", "PL:pl")}, raising=False)

        def fake_fetch(query, bucket, cutoff, locale, when_days=0):
            return [_art(f"Story from seo farm {query}", source="IndexBox"),
                    _art(f"Ticker {query}", url="https://stocktickers.example/a", source="Some"),
                    _art(f"Real report on {query}", source="Rzeczpospolita")]

        monkeypatch.setattr(collect, "_fetch_one", fake_fetch)
        monkeypatch.setattr(collect, "_fetch_bing", lambda *a, **k: [])
        prof = _profile(local_language="Polish", local_queries=["nowy sklep"])
        out = collect.collect(prof, config.get_country("Poland"))
        assert out and all(a.source == "Rzeczpospolita" for a in out)
        out_t = collect.collect_topic("Metal Furniture Market", config.get_country("Poland"),
                                      local_topic="meble")
        assert out_t and all(a.source == "Rzeczpospolita" for a in out_t)

    def test_no_blocklist_attribute_is_fine(self, monkeypatch):
        monkeypatch.delattr(config, "SOURCE_BLOCKLIST", raising=False)
        assert collect._drop_blocked([_art("x", source="IndexBox")])


# ── 3. relevance ─────────────────────────────────────────────────────────────
class TestRelevance:
    def test_fallback_when_model_missing(self):
        relevance.set_encoder(None)
        arts = [_art("Lidl opens store in Krakow")]
        assert relevance.seller_relevance(arts, _profile()) is None
        scored = score.score_all(arts, _profile())
        assert scored[0].seller_rel is None
        assert scored[0].relevancy >= 0.35          # bucket floor still applies

    def test_encoder_exception_never_crashes(self):
        class Boom:
            def embed(self, texts):
                raise RuntimeError("no")
        relevance.set_encoder(Boom())
        assert relevance.seller_relevance([_art("x")], _profile()) is None

    def test_seller_text_covers_value_prop_products_buyers(self):
        t = relevance.seller_text(_profile(products=[{"name": "Rack X", "application": "shelving"}]))
        assert "shop fittings" in t and "Rack X (shelving)" in t and "Retail" in t

    def test_embedding_replaces_floor(self, fake_model):
        on = _art("Retailers want new shop fittings and fixtures")
        off = _art("Kospi stock market closes higher")
        scored = score.score_all([on, off], _profile())
        by = {a.title: a for a in scored}
        assert by[on.title].seller_rel > by[off.title].seller_rel
        assert by[on.title].relevancy > by[off.title].relevancy
        assert by[off.title].relevancy < 0.25       # no longer lifted by the bucket floor


# ── 4. event dedupe ──────────────────────────────────────────────────────────
class TestEventDedupe:
    def test_same_event_two_languages_collapses(self, fake_model):
        en = _art("Lidl opens store in Krakow", summary="short")
        pl = _art("Lidl otwiera sklep w Krakowie", summary="A longer Polish lead paragraph " * 4)
        other = _art("Poultry export grows")
        out = collect._collapse_events([en, pl, other])
        titles = {a.title for a in out}
        assert len(out) == 2
        assert pl.title in titles and en.title not in titles     # longest summary wins
        assert other.title in titles

    def test_without_model_only_token_overlap_applies(self):
        relevance.set_encoder(None)
        en = _art("Lidl opens store in Krakow")
        pl = _art("Lidl otwiera sklep w Krakowie")
        assert len(collect._collapse_events([en, pl])) == 2

    def test_bucket_preference_beats_summary(self, fake_model):
        ind = _art("Lidl opens store in Krakow", bucket="industry", summary="x" * 300)
        trig = _art("Lidl otwiera sklep w Krakowie", bucket="trigger")
        out = collect._collapse_events([ind, trig])
        assert [a.bucket for a in out] == ["trigger"]


# ── 5. eligible() relevance gate ─────────────────────────────────────────────
class TestEligibleGate:
    def _pool(self, sims):
        arts = []
        for i, s in enumerate(sims):
            a = _art(f"Story {i}")
            a.composite, a.seller_rel = 0.6, s
            arts.append(a)
        return arts

    def test_drops_low_relevance_in_pool(self):
        sims = [0.02, 0.03, 0.05] + [0.3] * 9
        kept = score.eligible(self._pool(sims))
        assert len(kept) == 9 and all(a.seller_rel >= 0.3 for a in kept)

    def test_keeps_everything_when_pool_is_uniformly_related(self):
        kept = score.eligible(self._pool([0.25, 0.3, 0.35, 0.4] * 3))
        assert len(kept) == 12

    def test_skipped_without_embeddings(self):
        arts = self._pool([None] * 12)
        assert len(score.eligible(arts)) == 12

    def test_small_pool_not_gated(self):
        assert len(score.eligible(self._pool([0.0, 0.5, 0.5]))) == 3
