"""Query/geo/validation fixes for the "South Korea lead got an India story" bug.

Trace being locked down: the LLM-written profile query "food safety standards India"
had the lead's country appended by collect() ("... India South Korea"), so
geo.filter_for_lead's no-geo anchor rule read the article as anchored to South Korea;
the headline ("FSSAI sanctions prosecution against Bisleri") named no country, so
regions.article_countries found none and nothing else stopped it.

  collect.collect    drops profile-written queries that name a foreign country,
                     and searches the new competitor_query_seeds field as
                     competitor-bucket jobs
  geo.filter_for_lead  a query anchors only when it names the lead country AND
                     names no other country
  validate_brief.mask_entities  masks the lead's own proper nouns before the
                     banned-word check

All collection calls are mocked at collect._fetch_one -- no network.
"""
from datetime import datetime, timezone
from types import SimpleNamespace

import collect as C
import config
import geo
import validate_brief as vb
from collect import Article


def _art(title, *, query="", bucket="industry", summary="", source="S",
         url="http://x"):
    return Article(title=title, url=url, source=source,
                   published=datetime.now(timezone.utc), summary=summary,
                   query=query, bucket=bucket, fingerprint=title)


def _profile(**kw):
    base = dict(company="Bisleri International", competitors=["Rival Water Co"],
                trigger_queries=[], industry_queries=[],
                competitor_query_seeds=[], products=[])
    base.update(kw)
    return SimpleNamespace(**base)


def _korea():
    return config.get_country("South Korea")


def _record_jobs(monkeypatch):
    """Replace the network fetch with a recorder; return the list it fills."""
    calls = []

    def fake_fetch(query, bucket, cutoff, locale, when_days=0):
        calls.append({"query": query, "bucket": bucket, "cutoff": cutoff})
        return []

    monkeypatch.setattr(C, "_fetch_one", fake_fetch)
    return calls


# ── collect.collect: foreign-country profile queries ─────────────────────────
class TestForeignQueryDropped:
    def test_query_naming_another_country_is_not_searched(self, monkeypatch):
        calls = _record_jobs(monkeypatch)
        p = _profile(industry_queries=["food safety standards India",
                                       "bottled water safety"])
        C.collect(p, _korea())
        queries = [c["query"] for c in calls]
        assert not any("food safety standards India" in q for q in queries)
        assert "bottled water safety Korea" in queries

    def test_lead_country_query_is_kept(self, monkeypatch):
        calls = _record_jobs(monkeypatch)
        p = _profile(industry_queries=["water quality India"])
        C.collect(p, config.get_country("India"))
        # collect() country-suffixes every query (pre-existing behaviour), so the
        # kept query is the suffixed form -- the point is that it was not dropped.
        assert any(c["query"].startswith("water quality India")
                   for c in calls), calls

    def test_global_lead_keeps_every_query(self, monkeypatch):
        calls = _record_jobs(monkeypatch)
        p = _profile(trigger_queries=["water recall India"],
                     industry_queries=["food safety standards India"],
                     competitor_query_seeds=["bottled water recall"])
        C.collect(p, config.COUNTRIES["Global"])
        queries = [c["query"] for c in calls]
        assert "water recall India" in queries
        assert "food safety standards India" in queries
        assert "bottled water recall" in queries

    def test_dropped_queries_are_counted(self, monkeypatch):
        _record_jobs(monkeypatch)
        p = _profile(trigger_queries=["plant investment Germany"],
                     industry_queries=["food safety standards India"])
        C.collect(p, _korea())
        dropped = {q for q, _isos in C.LAST_DROPPED_QUERIES}
        assert dropped == {"plant investment Germany", "food safety standards India"}

    def test_dropped_queries_are_logged(self, monkeypatch):
        _record_jobs(monkeypatch)
        lines = []
        p = _profile(industry_queries=["food safety standards India"])
        C.collect(p, _korea(), log=lines.append)
        assert len(lines) == 1
        assert "food safety standards India" in lines[0]
        assert "IN" in lines[0]


# ── collect.collect: competitor_query_seeds ───────────────────────────────────
class TestCompetitorQuerySeeds:
    def test_seeds_become_competitor_jobs_with_the_country_suffix(self, monkeypatch):
        calls = _record_jobs(monkeypatch)
        p = _profile(competitor_query_seeds=["bottled water recall",
                                             "mineral water plant expansion"])
        C.collect(p, _korea())
        seed_calls = [c for c in calls if c["bucket"] == "competitor"
                      and "bottled water recall" in c["query"]]
        assert seed_calls, calls
        assert seed_calls[0]["query"] == "bottled water recall Korea"
        assert any(c["query"] == "mineral water plant expansion Korea"
                   for c in calls)

    def test_seeds_use_the_long_competitor_cutoff(self, monkeypatch):
        calls = _record_jobs(monkeypatch)
        p = _profile(competitor_query_seeds=["bottled water recall"])
        C.collect(p, _korea())
        call = next(c for c in calls if c["query"] == "bottled water recall Korea")
        age_days = (datetime.now(timezone.utc) - call["cutoff"]).days
        # every bucket now looks back ~182 days.
        assert age_days >= 150, age_days

    def test_seeds_naming_a_foreign_country_are_dropped(self, monkeypatch):
        calls = _record_jobs(monkeypatch)
        p = _profile(competitor_query_seeds=["water plant expansion India"])
        C.collect(p, _korea())
        assert "water plant expansion India" not in [c["query"] for c in calls]


# ── geo.filter_for_lead: the no-geo anchor rule ──────────────────────────────
class TestAnchorRuleRejectsForeignQuery:
    def test_query_naming_india_and_korea_does_not_anchor_a_korean_lead(self):
        """The live bug: collect() appended " South Korea" to an India query, and the
        anchor rule matched the appended word alone."""
        a = _art("FSSAI sanctions prosecution against Bisleri",
                 query="food safety standards India South Korea")
        kept, _ = geo.filter_for_lead([a], _korea())
        assert kept == []

    def test_query_naming_only_the_lead_country_anchors(self):
        a = _art("FSSAI sanctions prosecution against Bisleri",
                 query="food safety standards South Korea")
        kept, _ = geo.filter_for_lead([a], _korea())
        assert kept == [a]

    def test_demonym_only_query_still_anchors(self):
        a = _art("Water regulator tightens labelling rules",
                 query="food safety standards korean")
        kept, _ = geo.filter_for_lead([a], _korea())
        assert kept == [a]


# ── validate_brief.mask_entities ─────────────────────────────────────────────
class TestMaskEntities:
    def _waters(self):
        return SimpleNamespace(
            company="Waters",
            products=[{"name": "Empower", "application": "chromatography software"},
                      {"name": "Empower chromatography software"}],
            competitors=["Thermo Fisher"])

    def test_product_name_is_masked_but_the_generic_word_is_not(self):
        masked = vb.mask_entities("The Empower 3 software suite runs the assay.",
                                  self._waters())
        assert "Empower" not in masked
        assert vb.banned_word_warnings(masked) == []

        generic = vb.mask_entities("We empower every lab team to move faster.",
                                   self._waters())
        assert any("'empower'" in w for w in vb.banned_word_warnings(generic))

    def test_whole_phrase_is_masked_before_the_bare_brand_word(self):
        masked = vb.mask_entities("Empower chromatography software is validated.",
                                  self._waters())
        assert "Empower" not in masked
        assert masked.count("[entity]") == 1

    def test_company_and_competitor_names_are_masked(self):
        p = SimpleNamespace(company="Robust Systems",
                            products=[], competitors=["Unlock Labs"])
        masked = vb.mask_entities("Robust Systems competes with Unlock Labs.", p)
        assert "Robust" not in masked and "Unlock" not in masked
        assert vb.banned_word_warnings(masked) == []

    def test_unrelated_banned_words_still_flagged(self):
        masked = vb.mask_entities("Empower 3 is a robust, seamless suite.",
                                  self._waters())
        flagged = " ".join(vb.banned_word_warnings(masked))
        assert "robust" in flagged and "seamless" in flagged

    def test_no_profile_fields_is_a_no_op(self):
        text = "The quick brown fox."
        assert vb.mask_entities(text, SimpleNamespace()) == text
